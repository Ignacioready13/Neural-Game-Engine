"""Stage 5 — train the steering core (Milestone 1).

Trains ``steer.SteeringModel`` (readout -> structured warp -> constrained
residual) on bulk_v1 with **hold-last, deployment-consistent** windows:

  * the window's future actions are the held last action (no new click inside the
    horizon), so the held action we feed at play time is exactly the action the
    real clip is responding to;
  * the supervision target is the *cumulative horizontal image shift* over the
    next 1..H steps, taken from the precomputed phase-correlation table
    (``shift_table.py``) — the same quantity the Gate G1 shift metrics measure.

Two ingredients beyond plain supervised training matter a lot in practice:

  * **DAgger-style roll-in** (``dagger_prob``): for part of every batch the model
    first rolls itself forward a few steps on the real click history, and the
    readout is then supervised on *those* frames. Without it the closed loop is
    bistable: if the model fails to start turning, its own frames show no turn, the
    readout keeps predicting "no turn", and the turn never happens (measured:
    closed-loop shift ratio collapses from 0.88 to 0.32). The roll-in teaches the
    readout to answer from the click + the frame content even when its own past
    frames are wrong.
  * **Mirror augmentation**: flip the frames, negate ``actionX``/``deltaActionX``
    and the shift targets (the world is left/right symmetric; the bulk balance is
    50/50).

Run (from /neural):

    ./memcap.sh --mem 14G -- .venv/bin/python train_steer.py --name steer64 \
        --packed data/bulk_v1_64x48 --max-steps 6000 --batch 64
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import PackedDataset
from shift_table import load_or_compute, load_target_table
from steer import (SteeringModel, click_features, count_parameters,
                   save_steer_checkpoint)
from train import _set_seed

EXTRA_HIST = 8          # frames of action history before the window (for since-click)
HORIZONS = tuple(range(1, 11))
DAGGER_KMAX = 12        # max self-rollout steps used in a DAgger roll-in


DEFAULT_CONFIG = {
    "name": "steer64",
    "packed": "data/bulk_v1_64x48",
    "K": 4,
    "H": 10,
    "base": 32,
    "mode": "warp_resid",     # "warp_resid" | "warp_only"
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "max_steps": 6000,
    "seed": 0,
    "mixed_precision": "bf16",
    "val_every": 500,
    "val_windows": 2048,
    "mirror_prob": 0.5,
    "dagger_prob": 0.5,
    "dagger_kmax": DAGGER_KMAX,
    "cf_prob": 0.3,          # counterfactual "click the other door" augmentation
    "spatial_click": True,   # Gaussian click-column channels into the readout encoder
    "w_cum": 1.0,
    "w_sign": 0.5,
    "w_inc1": 0.2,
    "w_resid": 1e-3,
    "target": "heading",      # supervision target: "heading" (denoised) | "phase"
    "max_windows": None,      # cap candidates (debug)
    "resume": None,           # checkpoint path to continue training from
    "frame_model": None,      # flow checkpoint whose frames the DAgger roll-in uses
    "out": "runs",
}


def global_prefix(table: dict) -> np.ndarray:
    """Global cumulative-sum prefix of the per-step shift table.

    Differences are only ever taken inside one episode, so a single global cumsum
    is enough to read off any within-episode cumulative shift.
    """
    return np.concatenate([[0.0], np.cumsum(table["dx"].astype(np.float64))])


def cum_targets(P: np.ndarray, gt: np.ndarray, horizons=HORIZONS) -> np.ndarray:
    """Cumulative shift over each horizon, for global start frames ``gt``."""
    h = np.asarray(horizons)
    return (P[gt[:, None] + h[None, :]] - P[gt[:, None]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Candidate windows + targets (vectorized per episode)
# --------------------------------------------------------------------------- #

def fit_click_geometry(packed: PackedDataset, sidecar, table: dict, K: int, H: int,
                       split: str = "train", max_n: int = 40000, cache: bool = True) -> dict:
    """Fit the engine-rule geometry ``cum_m ~ geom_m * screenX(target door)``.

    The engine resolves a click to the reachable doorway nearest to it in screen
    space and turns toward it, so the *counterfactual* target for "the player
    clicked that other door" is well predicted by the clicked door's ``screenX``.
    Fitting one coefficient per horizon (a training-only use of the sidecar) gives
    a target for the counterfactual augmentation that forces the readout to use
    the click instead of reading the direction off the frames.
    """
    from analysis import load_analysis
    path = os.path.join(os.path.dirname(os.path.abspath(packed.path)), "_cache",
                        os.path.basename(packed.path) + "_click_geom.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cache and os.path.exists(path):
        z = np.load(path)
        return {"coef": z["coef"], "sign_acc": float(z["sign_acc"]), "n": int(z["n"])}
    if sidecar is None:
        sidecar = load_analysis(packed)
    P = global_prefix(table)
    cand = build_candidates(packed, table, split, K, H)
    rng = np.random.default_rng(0)
    idx = np.arange(len(cand["ep"]))
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, max_n, replace=False))
    off = {e["episode_index"]: e["frame_offset"] for e in packed.episodes}
    X, C = [], []
    for i in idx.tolist():
        e = int(cand["ep"][i]); t = int(cand["gt"][i] - off[e])
        ax = float(packed.labels[off[e] + t]["actionX"])
        ahc = float(packed.labels[off[e] + t]["actionHasClicked"])
        if np.isnan(ax) or ahc < 0.5:
            continue
        cs = [c for c in sidecar.candidates_at(e, t) if c["reachable"]]
        if not cs:
            continue
        near = min(cs, key=lambda c: abs(c["screenX"] - ax))
        X.append(near["screenX"])
        C.append(cand["cum"][i])
    X = np.array(X); C = np.array(C)
    coef = np.zeros(H, np.float64)
    for m in range(H):
        A = np.stack([X, np.ones_like(X)], 1)
        sol, *_ = np.linalg.lstsq(A, C[:, m], rcond=None)
        coef[m] = sol[0]
    sign_acc = float((np.sign(coef[-1] * X) == np.sign(C[:, -1])).mean())
    print(f"click geometry: coef(H) {coef[-1]:.2f} px per screenX unit, "
          f"sign agreement {sign_acc:.3f} (n={len(X)})", flush=True)
    if cache:
        np.savez_compressed(path, coef=coef, sign_acc=sign_acc, n=len(X))
    return {"coef": coef, "sign_acc": sign_acc, "n": len(X)}


def build_candidates(packed: PackedDataset, table: dict, split: str, K: int, H: int,
                     max_windows: int | None = None, sidecar=None,
                     geom: dict | None = None) -> dict:
    """All hold-last-consistent, safe (episode, start) windows with shift targets.

    Returns arrays ``ep``, ``start``, ``gt`` (global last-context frame),
    ``cum`` [n, H], ``weight`` [n], ``onset`` [n], ``kmax`` [n] (how many
    self-rollout steps keep the supervision window action-consistent).
    """
    dx = table["dx"].astype(np.float64)
    eps, starts, gts, cums, weights, onsets, kmaxs = [], [], [], [], [], [], []
    doors, alts = [], []
    for e in packed.episodes_by_split[split]:
        L = e["length"]
        off = e["frame_offset"]
        if L < K + H + 1:
            continue
        A = np.nan_to_num(np.asarray(packed.labels["actionX"][off: off + L],
                                     dtype=np.float64), nan=0.0)
        C = np.nan_to_num(np.asarray(packed.labels["actionHasClicked"][off: off + L],
                                     dtype=np.float64), nan=0.0)
        U = np.asarray(packed.labels["viewUnsafe"][off: off + L], dtype=np.int64)
        change = np.zeros(L, dtype=np.int64)
        change[1:] = ((A[1:] != A[:-1]) | (C[1:] != C[:-1])).astype(np.int64)
        cchange = np.concatenate([[0], np.cumsum(change)])
        cunsafe = np.concatenate([[0], np.cumsum(U)])
        P = np.concatenate([[0.0], np.cumsum(dx[off: off + L])])
        ch_idx = np.where(change == 1)[0]

        starts_e = np.arange(0, L - (K + H) + 1, dtype=np.int64)
        if len(starts_e) == 0:
            continue
        t = starts_e + K - 1                       # last context frame
        # safe window over [s, s+K+H)
        safe = (cunsafe[starts_e + K + H] - cunsafe[starts_e]) == 0
        # no new click among the future held actions a_{t+1} .. a_{t+H-1}
        consistent = (cchange[t + H] - cchange[t + 1]) == 0
        keep = safe & consistent
        s = starts_e[keep]
        t = t[keep]
        if len(s) == 0:
            continue
        cum = np.stack([P[t + m] - P[t] for m in HORIZONS], axis=1)   # [n, H]
        # how far we may roll in and still supervise from the held action
        nxt = np.searchsorted(ch_idx, t + 1, side="left")
        safe_nxt = np.clip(nxt, 0, max(0, len(ch_idx) - 1))
        next_change = np.where(nxt < len(ch_idx), ch_idx[safe_nxt], L)
        kmax = np.clip(next_change - t - H, 0, DAGGER_KMAX)
        onset = (change[np.clip(t - 2, 0, L - 1)] | change[np.clip(t - 1, 0, L - 1)]
                 | change[t]).astype(bool)
        w = np.ones(len(s), dtype=np.float64)
        w[onset] = 4.0
        w[np.abs(cum[:, H - 1]) >= 1.0] = np.maximum(w[np.abs(cum[:, H - 1]) >= 1.0], 2.0)
        door_x = np.full(len(s), np.nan, np.float64)
        alt_x = np.full(len(s), np.nan, np.float64)
        if sidecar is not None:
            ei = e["episode_index"]
            for j in range(len(s)):
                cs = [c for c in sidecar.candidates_at(ei, int(t[j]))
                      if c["reachable"]]
                if not cs:
                    continue
                ax = A[t[j]]
                near = min(cs, key=lambda c: abs(c["screenX"] - ax))
                door_x[j] = near["screenX"]
                others = [c["screenX"] for c in cs if c["doorwayId"] != near["doorwayId"]]
                if others:
                    alt_x[j] = max(others, key=lambda v: abs(v - ax))
        eps.append(np.full(len(s), e["episode_index"], dtype=np.int64))
        starts.append(s)
        gts.append(off + t)
        doors.append(door_x)
        alts.append(alt_x)
        cums.append(cum)
        weights.append(w)
        onsets.append(onset)
        kmaxs.append(kmax)
    out = {
        "ep": np.concatenate(eps), "start": np.concatenate(starts),
        "gt": np.concatenate(gts),
        "cum": np.concatenate(cums).astype(np.float32),
        "weight": np.concatenate(weights).astype(np.float32),
        "onset": np.concatenate(onsets),
        "kmax": np.concatenate(kmaxs).astype(np.int64),
        "door_x": np.concatenate(doors).astype(np.float32) if doors else None,
        "alt_x": np.concatenate(alts).astype(np.float32) if alts else None,
    }
    if max_windows and len(out["ep"]) > max_windows:
        idx = np.linspace(0, len(out["ep"]) - 1, max_windows).astype(int)
        for k in out:
            out[k] = out[k][idx]
    return out


# --------------------------------------------------------------------------- #
# Batch building
# --------------------------------------------------------------------------- #

def _batch_windows(packed, cand, idx, cfg, device, rng, extra_hist=0):
    """Materialize a batch: context, action history (mirrored), targets."""
    K = cfg["K"]
    B = len(idx)
    ep = cand["ep"][idx]
    st = cand["start"][idx]
    n_hist = EXTRA_HIST + K + extra_hist
    H, W = packed.height, packed.width
    ctx = np.empty((B, K, H, W, 3), np.uint8)
    hist = np.empty((B, n_hist, 2), np.float32)
    for i in range(B):
        e = int(ep[i]); s = int(st[i])
        rec = packed.episodes[e]
        off, L = rec["frame_offset"], rec["length"]
        lo = max(0, s - EXTRA_HIST)
        hi = min(L, s + K + extra_hist)
        a = packed.action_vectors(packed.labels[off + lo: off + hi])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        if len(a) < n_hist:
            pad = n_hist - len(a)
            a = np.concatenate([np.repeat(a[:1], pad, axis=0), a], axis=0)
        hist[i] = a
        ctx[i] = packed.images[off + s: off + s + K]
    ctx_t = torch.from_numpy(np.ascontiguousarray(
        ctx.transpose(0, 1, 4, 2, 3))).float().to(device) / 127.5 - 1.0
    if cfg["mirror_prob"] > 0:
        flip = rng.random(B) < cfg["mirror_prob"]
        if flip.any():
            fi = torch.from_numpy(np.where(flip)[0]).to(device)
            ctx_t[fi] = torch.flip(ctx_t[fi], dims=[3])
            hist[flip, :, 0] *= -1.0
    hist_t = torch.from_numpy(np.ascontiguousarray(hist)).to(device)
    return ctx_t, hist_t


def _features_slice(hist_t, start_off, K):
    """Click features for the K frames starting at history offset ``start_off``."""
    return click_features(hist_t)[:, start_off: start_off + K]


@torch.no_grad()
def _roll_in(model, ctx, hist_t, ks, K, frame_model=None):
    """DAgger roll-in: run the model forward ``k`` steps per window (no grad).

    Returns ``(ctx_syn, feats_syn)`` where ``ctx_syn`` is the K-frame context after
    the roll-in (the real context where ``k == 0``) and ``feats_syn`` the click
    features aligned to it.
    """
    buf = ctx.clone()
    kmax = int(ks.max()) if len(ks) else 0
    for j in range(kmax):
        active = ks > j
        if not bool(active.any()):
            break
        feats = _features_slice(hist_t, EXTRA_HIST + j, K)
        src = frame_model if frame_model is not None else model
        nxt = src.predict(buf, features=feats, seed=(j * 17 + 3) if frame_model else None)
        if isinstance(nxt, tuple):
            nxt = nxt[0]
        new = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
        buf = torch.where(active.view(-1, 1, 1, 1, 1), new, buf)
    allf = click_features(hist_t)                      # [B, T, F]
    offs = ((EXTRA_HIST + ks).view(-1, 1)
            + torch.arange(K, device=hist_t.device).view(1, -1))
    feats = torch.gather(allf, 1, offs[:, :, None].expand(-1, -1, allf.shape[-1]))
    return buf, feats


def _loss(model, ctx, feats, tgt, cfg):
    pred, meta = model.predict(ctx, features=feats, return_shift=True)
    inc = meta["inc"]                       # [B, H] per-step shifts
    cum = torch.cumsum(inc, dim=1)
    Hn = tgt.shape[1]
    w = torch.arange(1, Hn + 1, device=tgt.device, dtype=torch.float32)
    w = w / w.sum()
    l_cum = ((cum - tgt).abs() * w).mean()
    s_t = tgt[:, Hn - 1]
    m = s_t.abs() > 0.5
    l_sign = torch.zeros((), device=tgt.device)
    if m.any():
        y = (s_t[m] > 0).float()
        l_sign = F.binary_cross_entropy_with_logits(meta["sign_logit"][m], y)
    l_inc1 = (inc[:, 0] - tgt[:, 0]).abs().mean()
    l_resid = torch.zeros((), device=tgt.device)
    if model.use_resid:
        l_resid = (pred - meta["warped"]).abs().mean()
    total = (cfg["w_cum"] * l_cum + cfg["w_sign"] * l_sign
             + cfg["w_inc1"] * l_inc1 + cfg["w_resid"] * l_resid)
    return total, {"l_cum": float(l_cum.detach()), "l_sign": float(l_sign.detach()),
                   "l_inc1": float(l_inc1.detach()), "l_resid": float(l_resid.detach())}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def closed_loop_metric(model, packed, cand, P, K, steps=40, n=48, device="cuda"):
    """Closed-loop (own frames, real clicks) commanded-shift ratio and sign vs real."""
    sel = np.where(cand["onset"] & (np.abs(cand["cum"][:, -1]) >= 2.0))[0]
    if len(sel) == 0:
        return {"closed_sign": None, "closed_ratio": None, "closed_n": 0}
    if len(sel) > n:
        sel = sel[np.linspace(0, len(sel) - 1, n).astype(int)]
    ctx, hist_t = _batch_windows(packed, cand, sel, {"K": K, "mirror_prob": 0.0},
                                 device, np.random.default_rng(0),
                                 extra_hist=steps + 2)
    buf = ctx
    acc = torch.zeros(len(sel), device=device)
    for j in range(steps):
        feats = _features_slice(hist_t, EXTRA_HIST + j, K)
        out, info = model.predict(buf, features=feats, return_shift=True)
        acc = acc + info["dx"].float()
        buf = torch.cat([buf[:, 1:], out[:, None]], dim=1)
    gt = cand["gt"][sel]
    real = torch.from_numpy((P[gt + steps] - P[gt]).astype(np.float32)).to(device)
    m = real.abs() >= 2.0
    if not bool(m.any()):
        return {"closed_sign": None, "closed_ratio": None, "closed_n": 0}
    sign = float((torch.sign(acc[m]) == torch.sign(real[m])).float().mean())
    ratio = float(acc[m].abs().mean() / real[m].abs().mean())
    return {"closed_sign": sign, "closed_ratio": ratio, "closed_n": int(m.sum())}


@torch.no_grad()
def evaluate(model, packed, cand, P, K, cfg, device, use_bf16):
    model.eval()
    n = len(cand["ep"])
    batch = min(256, cfg["batch_size"] * 2)
    rng = np.random.default_rng(12345)
    cum_pred = np.empty((n, cfg["H"]), np.float32)
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        ctx, hist_t = _batch_windows(packed, cand, idx, {**cfg, "mirror_prob": 0.0},
                                     device, rng)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            _, meta = model.predict(ctx, features=_features_slice(hist_t, EXTRA_HIST, K),
                                    return_shift=True)
            cum = torch.cumsum(meta["inc"].float(), dim=1)
        cum_pred[i: i + len(idx)] = cum.float().cpu().numpy()
    tgt = cand["cum"]
    h = cfg["H"]
    w = np.arange(1, h + 1, dtype=np.float32); w /= w.sum()
    l_cum = float((np.abs(cum_pred - tgt) * w).mean())
    s_t, s_p = tgt[:, -1], cum_pred[:, -1]
    m = np.abs(s_t) > 0.5
    sign_acc = float((np.sign(s_p[m]) == np.sign(s_t[m])).mean()) if m.any() else float("nan")
    ratio = (float(np.abs(s_p[m]).mean() / max(1e-6, np.abs(s_t[m]).mean()))
             if m.any() else float("nan"))
    cl = closed_loop_metric(model, packed, cand, P, K, steps=40, device=device)
    model.train()
    return {"val_cum_l1": l_cum, "val_sign_acc": sign_acc, "val_ratio": ratio,
            "val_loss": l_cum, **cl}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train(config: dict) -> dict:
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"
    rng = np.random.default_rng(config["seed"])
    K, H = config["K"], config["H"]

    packed = PackedDataset(config["packed"], in_memory=False)
    table = load_target_table(packed, config["packed"], config.get("target", "heading"))
    P = global_prefix(table)
    from analysis import load_analysis
    sidecar = load_analysis(packed)
    geom = fit_click_geometry(packed, sidecar, table, K, H)
    print("building train candidates ...", flush=True)
    train_cand = build_candidates(packed, table, "train", K, H, config["max_windows"],
                                  sidecar=sidecar, geom=geom)
    print(f"train windows: {len(train_cand['ep'])} "
          f"(onset {int(train_cand['onset'].sum())}, "
          f"with alternate door {int(np.isfinite(train_cand['alt_x']).sum())})",
          flush=True)
    val_cand = build_candidates(packed, table, "val", K, H, config["val_windows"],
                                sidecar=sidecar, geom=geom)

    model = SteeringModel(K=K, H=H, base=config["base"], mode=config["mode"],
                          spatial_click=config.get("spatial_click", True)).to(device)
    frame_model = None
    if config.get("frame_model"):
        # stack-level DAgger: the roll-in frames come from the *generator* pipeline,
        # so the readout is trained on the frames it will actually see at play time.
        from flow import FlowSteerModel
        fp = torch.load(config["frame_model"], map_location=device, weights_only=False)
        frame_model = FlowSteerModel(**fp["model_config"]).to(device)
        frame_model.load_state_dict(fp["state_dict"])
        frame_model.eval()
        for prm in frame_model.parameters():
            prm.requires_grad_(False)
        print(f"DAgger roll-in frames come from {config['frame_model']}", flush=True)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    start_step = 0
    if config.get("resume"):
        ck = torch.load(config["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {config['resume']} at step {start_step} "
              f"(model weights only; the optimizer state is not kept)", flush=True)
    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**config, "n_params": n_params, "device": device,
                   "n_train_windows": len(train_cand["ep"]),
                   "resumed_from": config.get("resume")}, f, indent=1)

    p = train_cand["weight"].astype(np.float64)
    cdf = np.cumsum(p / p.sum()); cdf[-1] = 1.0
    kmax_arr = train_cand["kmax"]
    dk = int(config["dagger_kmax"])
    log, best, step, t0 = [], float("inf"), int(start_step), time.time()
    running, seen = 0.0, 0
    while step < config["max_steps"]:
        idx = np.clip(np.searchsorted(cdf, rng.random(config["batch_size"]), side="right"),
                      0, len(p) - 1)
        ctx, hist_t = _batch_windows(packed, train_cand, idx, config, device, rng,
                                     extra_hist=dk + 2)
        # counterfactual "click the other door": same frames, click replaced, and the
        # target taken from the fitted engine-rule geometry. This is what forces the
        # readout to use the click instead of reading the direction off the frames.
        alt = train_cand["alt_x"][idx]
        cf = (rng.random(len(idx)) < config["cf_prob"]) & np.isfinite(alt)
        coef = torch.from_numpy(geom["coef"].astype(np.float32)).to(device)
        ks = np.zeros(len(idx), dtype=np.int64)
        use_d = (rng.random(len(idx)) < config["dagger_prob"]) & ~cf
        if use_d.any():
            ks[use_d] = rng.integers(1, dk + 1, size=int(use_d.sum()))
        ks = np.minimum(ks, kmax_arr[idx])
        tgt = cum_targets(P, train_cand["gt"][idx] + ks)
        if cf.any():
            cfx = torch.from_numpy(alt[cf].astype(np.float32)).unsqueeze(1).to(device)
            tgt[cf] = (cfx * coef.unsqueeze(0)).cpu().numpy()
            sel = torch.from_numpy(np.where(cf)[0]).to(device)
            tail = hist_t[sel].clone()
            tail[:, EXTRA_HIST + K - 1:, 0] = cfx
            tail[:, EXTRA_HIST + K - 1:, 1] = 1.0
            hist_t[sel] = tail
        tgt = torch.from_numpy(np.ascontiguousarray(tgt)).to(device)
        if ks.max() > 0:
            ctx_syn, feats = _roll_in(model, ctx, hist_t,
                                      torch.from_numpy(ks).to(device), K,
                                      frame_model=frame_model)
        else:
            ctx_syn, feats = ctx, _features_slice(hist_t, EXTRA_HIST, K)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            loss, parts = _loss(model, ctx_syn, feats, tgt, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        running += float(loss.item()); seen += 1; step += 1
        if step % config["val_every"] == 0 or step == config["max_steps"]:
            tr = running / seen; running, seen = 0.0, 0
            v = evaluate(model, packed, val_cand, P, K, config, device, use_bf16)
            rec = {"step": step, "train_loss": tr, "elapsed_s": time.time() - t0, **v}
            log.append(rec)
            print(f"step {step} train {tr:.4f} val_cum {v['val_cum_l1']:.4f} "
                  f"sign {v['val_sign_acc']:.3f} ratio {v['val_ratio']:.3f} "
                  f"| closed sign {v['closed_sign']} ratio {v['closed_ratio']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if v["val_loss"] < best:
                best = v["val_loss"]
                save_steer_checkpoint(os.path.join(out_dir, "best.pt"), model, config,
                                      best, step)
            if step % (5 * config["val_every"]) == 0 or step == config["max_steps"]:
                save_steer_checkpoint(os.path.join(out_dir, "last.pt"), model, config,
                                      v["val_loss"], step)
    dt = time.time() - t0
    result = {"name": config["name"], "n_params": n_params, "steps": step,
              "best_val_cum_l1": best, "seconds": dt, "steps_per_second": step / dt,
              "log": log[-1] if log else {}, "n_train_windows": len(train_cand["ep"])}
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the Stage 5 steering core (M1)")
    for k, dv in DEFAULT_CONFIG.items():
        if k == "out":
            continue
        if isinstance(dv, bool):
            t = lambda s: s.lower() in ("1", "true", "yes")  # noqa: E731
        elif isinstance(dv, int):
            t = int
        elif isinstance(dv, float):
            t = float
        else:
            t = str
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None, type=t)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config-json", default=None)
    args = ap.parse_args(argv)
    config = dict(DEFAULT_CONFIG)
    if args.config_json:
        with open(args.config_json) as f:
            config.update(json.load(f))
    config.update({k: v for k, v in vars(args).items()
                   if v is not None and k != "config_json"})
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
