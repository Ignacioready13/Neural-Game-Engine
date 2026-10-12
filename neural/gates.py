"""Stage 5 — the gate runner (G1-G4) and the Milestone metrics.

One command evaluates a Stage 5 checkpoint against the brief's gates:

    ./memcap.sh --mem 14G -- .venv/bin/python gates.py --gates g1 \
        --packed data/bulk_v1_64x48 --checkpoint runs/steer64/best.pt

Everything is measured on val only (the test split is touched once, in
Milestone 4, by passing ``--split test``). The gate definitions follow brief
§3 exactly; each function returns a dict of numbers plus the pass/fail booleans
against the stated thresholds, and the CLI prints a table.

Steering is measured the way the Stage 4 report says it must be: with the
phase-correlation horizontal image shift, the click counterfactual (both doors),
the horizon shift ratio, and the sustained-steering ratio — never by one-step
pixel error alone.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis import load_analysis
from dataset import PackedDataset, frames_to_tensor
from eval_harness import model_predict, rollout_v2
from metrics import (black_frac, color_stats, estimate_shift_x, frame_change,
                     laplacian_var, saturated_frac, seed_diversity, sharpness)
from model import PersistenceModel
from probe_click import _select_contexts
from shift_table import load_target_table
from steer import SteeringModel, click_features
from train_steer import build_candidates, global_prefix

_DOOR_NEAR = 2.0
_NEAR_SPLIT = 0.5
TEAL = "  "


# --------------------------------------------------------------------------- #
# Model loading (both Stage 5 kinds)
# --------------------------------------------------------------------------- #

def load_model(path: str, device: str = "cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    kind = payload.get("kind", "baseline")
    if kind == "steering":
        model = SteeringModel(**payload["model_config"]).to(device)
    elif kind == "flow":
        from flow import FlowSteerModel
        model = FlowSteerModel(**payload["model_config"]).to(device)
    else:
        from model import load_checkpoint
        return load_checkpoint(path, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# --------------------------------------------------------------------------- #
# Context selection
# --------------------------------------------------------------------------- #

def near_door_contexts(packed, sidecar, K, H, n_min=300, max_per_ep=2, split="val"):
    """Contexts with a close reachable door on each side (probe_click rule)."""
    ctxs = _select_contexts(packed, sidecar, K, H, max_per_ep=max_per_ep, split=split)
    eps = {c[0] for c in ctxs}
    return ctxs, len(eps)


def onset_windows(packed, table, K, n_min=300, max_per_ep=2, split="val",
                  min_turn_px=2.0, only_strict=False):
    """Val windows with a click at the last context step and a real turn ahead."""
    cand = build_candidates(packed, table, split, K, 10, None)
    keep = cand["onset"] & (np.abs(cand["cum"][:, -1]) >= min_turn_px)
    if only_strict:
        keep &= _strict_onset(packed, cand, K)
    ep, st = cand["ep"][keep], cand["start"][keep]
    seen, out = {}, []
    for e, s in zip(ep.tolist(), st.tolist()):
        if seen.get(e, 0) >= max_per_ep:
            continue
        seen[e] = seen.get(e, 0) + 1
        out.append((e, s))
    return out


def _strict_onset(packed, cand, K):
    """True where the click change event is exactly at the last context frame."""
    out = np.zeros(len(cand["ep"]), dtype=bool)
    for i in range(len(cand["ep"])):
        e = int(cand["ep"][i]); t = int(cand["gt"][i]) - packed.episodes[e]["frame_offset"]
        rec = packed.episodes[e]
        lab = packed.labels[rec["frame_offset"] + max(0, t - 1): rec["frame_offset"] + t + 1]
        a = np.nan_to_num(packed.action_vectors(lab), nan=0.0) if len(lab) else None
        if a is None or len(a) < 2:
            continue
        out[i] = bool(abs(a[1, 0] - a[0, 0]) > 1e-6 or abs(a[1, 1] - a[0, 1]) > 1e-6)
    return out


def sustained_windows(packed, table, K, split="val", steps=40, min_turn_px=2.0,
                      max_per_ep=2, n=300):
    """Onset windows with **no further click** inside the rollout (one clean turn).

    A retarget inside the 40 steps makes "the real clip's cumulative shift" a sum of
    two turns, so the sign/ratio comparison is not a single-turn test. These windows
    are the fair subset; the unrestricted number is reported alongside.
    """
    cand = build_candidates(packed, table, split, K, 10, None)
    ok = cand["onset"] & (np.abs(cand["cum"][:, -1]) >= min_turn_px)
    ep, st = cand["ep"][ok], cand["start"][ok]
    seen, out = {}, []
    for e, s in zip(ep.tolist(), st.tolist()):
        rec = packed.episodes[e]
        if s + K + steps + 1 > rec["length"]:
            continue
        # one clean turn: no further click anywhere inside the rollout
        t = s + K - 1
        a = np.nan_to_num(packed.action_vectors(
            packed.labels[rec["frame_offset"] + t: rec["frame_offset"] + t + steps + 1]),
            nan=0.0)
        if len(a) > 1:
            d = np.abs(np.diff(a[:, 0])) + np.abs(np.diff(a[:, 1]))
            if (d > 1e-6).any():
                continue
        if seen.get(e, 0) >= max_per_ep:
            continue
        seen[e] = seen.get(e, 0) + 1
        out.append((e, s))
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Rollout helpers
# --------------------------------------------------------------------------- #

def _u8(t):
    t = t.float().detach().cpu()
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)
    else:
        t = t.permute(1, 2, 0)
    u = (t * 0.5 + 0.5) * 255.0
    return np.clip(u.numpy(), 0, 255).astype(np.uint8)


def _cum_shift(seq_u8, block: int = 8):
    """Cumulative content shift of a sequence, measured in blocks.

    A single phase correlation over a long rollout aliases once the accumulated
    shift approaches half the frame width (a 40-step turn can reach ~40 px on a
    64 px-wide frame, where the estimator wraps) — measured: a commanded +37 px
    summed shift read as +3 px. Summing per-step estimates avoids the aliasing but
    adds the estimator's ~0.4 px noise 40 times (sigma*sqrt(40) ~ 2.5 px). Blocks of
    ``block`` steps keep each estimate well inside the alias-free range while
    cutting the noise to ``sqrt(n/block)`` terms.
    """
    n = len(seq_u8) - 1
    blocks = []
    i = 0
    while i < n:
        j = min(i + block, n)
        blocks.append(estimate_shift_x(seq_u8[i], seq_u8[j]))
        i = j
    per_step = np.array([estimate_shift_x(seq_u8[k], seq_u8[k + 1])
                         for k in range(n)])
    return float(sum(blocks)), per_step


def _real_seq(packed, episode, start, K, steps):
    e = packed.episodes[episode]
    off = e["frame_offset"]
    lo = start + K - 1
    hi = min(lo + steps + 1, e["length"])
    return np.asarray(packed.images[off + lo: off + hi])


def _rollout_seq(packed, model, episode, start, K, steps, override=None, seed=None,
                 device="cuda"):
    gen, info = rollout_v2(packed, model, episode, start, K, steps,
                           action_override=override, seed=seed, device=device)
    last_ctx = np.asarray(packed.images[packed.episodes[episode]["frame_offset"]
                                        + start + K - 1])[None]
    return np.concatenate([last_ctx, _u8(gen)], axis=0), info


# --------------------------------------------------------------------------- #
# G1 — the steering core
# --------------------------------------------------------------------------- #

COUNTERFACTUAL_EPS = 0.5     # px: a turn counts as "went that way" above this
_P_CACHE = {}                # heading-derived cumulative-shift prefix (set by run_g1)


def g1_counterfactual(packed, model, sidecar, K, H, ctxs, device="cuda",
                      seeds=(None,)):
    """Left / right / no-click counterfactual, two ways.

    * ``tf`` — **one forward pass** on the real context with the click injected at
      ``a_{K-1}``: the model's own predicted cumulative shift over H steps
      (``inc.sum()``). This is the exact analogue of ``probe_click.py`` for a model
      whose motion is an explicit parameter, and it is free of closed-loop feedback
      confounds.
    * ``cl`` — the same counterfactual rolled closed-loop on the model's own frames,
      measured as the phase-correlation image shift (what the player would see).

    Two "both-correct" readings are reported for each:
      ``directions``  = left turns left AND right turns right (each beyond
                        ``COUNTERFACTUAL_EPS`` px) — the convention already used by
                        ``eval_harness.counterfactual_eval``, and the one the gate's
                        "left door vs right door" comparison is about;
      ``ordered``     = ``L > hold > R`` — the stricter ordering against the no-click
                        baseline (fails on contexts where the player is *already*
                        turning, since "hold" then is itself a real turn).
    """
    res = {s: {"tf": {"left": [], "right": [], "hold": []},
               "cl": {"left": [], "right": [], "hold": []}, "meta": []} for s in seeds}
    for e, s, xl, xr, dl, dr in ctxs:
        e_pk = packed.episodes[e]
        off = e_pk["frame_offset"]
        lo = max(0, s - 8)
        hist = np.nan_to_num(packed.action_vectors(
            packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)
        held = hist[-1].copy()
        ctx = frames_to_tensor(
            np.asarray(packed.images[off + s: off + s + K]))[None].to(device)
        for seed in seeds:
            for cond, x, clicked in (("left", xl, 1.0), ("right", xr, 1.0),
                                     ("hold", float(held[0]), float(held[1]))):
                h = hist.copy()
                h[-1] = [x, clicked]
                feats = click_features(
                    torch.from_numpy(h)[None].float().to(device))[:, -K:]
                with torch.no_grad():
                    out = model_predict(model, ctx, None, feats, seed,
                                        return_shift=True)
                if isinstance(out, tuple) and isinstance(out[1], dict):
                    res[seed]["tf"][cond].append(
                        float(out[1]["inc"].sum(dim=1).reshape(-1)[0].item()))
                ov = torch.tensor([[[x, clicked]] * K], dtype=torch.float32).to(device)
                seq, _ = _rollout_seq(packed, model, e, s, K, H, override=ov,
                                      seed=seed, device=device)
                res[seed]["cl"][cond].append(estimate_shift_x(seq[0], seq[-1]))
            res[seed]["meta"].append((e, s, min(dl, dr)))

    def _block(r, tag):
        L = np.array(r[tag]["left"]); R = np.array(r[tag]["right"])
        Hd = np.array(r[tag]["hold"])
        eps = COUNTERFACTUAL_EPS
        d = {
            "mean_left_px": float(L.mean()), "mean_right_px": float(R.mean()),
            "mean_hold_px": float(Hd.mean()),
            "left_correct": float((L > eps).mean()),
            "right_correct": float((R < -eps).mean()),
            "both_correct_directions": float(((L > eps) & (R < -eps)).mean()),
            "left_beats_hold": float((L > Hd).mean()),
            "hold_beats_right": float((Hd > R).mean()),
            "both_correct_ordered": float(((L > Hd) & (Hd > R)).mean()),
            "left_gt_right": float((L > R).mean()),
        }
        return d

    out = {}
    for seed in seeds:
        r = res[seed]
        d = {"n_contexts": len(r["tf"]["left"]),
             "n_episodes": len({m[0] for m in r["meta"]}),
             "tf": _block(r, "tf"), "cl": _block(r, "cl")}
        near = np.array([m[2] < _NEAR_SPLIT for m in r["meta"]])
        for lbl, m in (("near", near), ("far", ~near)):
            if m.sum():
                L = np.array(r["tf"]["left"])[m]; R = np.array(r["tf"]["right"])[m]
                d[f"both_correct_directions_{lbl}"] = float(
                    ((L > COUNTERFACTUAL_EPS) & (R < -COUNTERFACTUAL_EPS)).mean())
                d[f"n_{lbl}"] = int(m.sum())
        out[str(seed)] = d
    first = out[str(seeds[0])]
    out["pass_both_correct_near_0.70"] = bool(
        first.get("both_correct_directions_near", 0.0) >= 0.70)
    return out


def g1_shift_ratio(packed, model, table, K, H, windows, device="cuda", seeds=(None,)):
    """Predicted-vs-real cumulative shift ratio at the H-step horizon."""
    preds, reals = [], []
    for e, s in windows:
        seq, _ = _rollout_seq(packed, model, e, s, K, H, seed=seeds[0], device=device)
        real = _real_seq(packed, e, s, K, H)
        preds.append(estimate_shift_x(seq[0], seq[-1]))
        reals.append(estimate_shift_x(real[0], real[-1]))
    preds, reals = np.array(preds), np.array(reals)
    m = np.abs(reals) >= 1.0
    ratio = float(np.abs(preds[m]).mean() / max(1e-6, np.abs(reals[m]).mean())) if m.any() else None
    return {
        "n": int(len(preds)),
        "mean_abs_pred_px": float(np.abs(preds).mean()),
        "mean_abs_real_px": float(np.abs(reals).mean()),
        "shift_ratio_h": ratio,
        "sign_agreement": float((np.sign(preds[m]) == np.sign(reals[m])).mean()) if m.any() else None,
        "pass_ratio_0.5": bool(ratio is not None and ratio >= 0.5),
    }


def g1_sustained(packed, model, K, steps, windows, device="cuda", seeds=(None,),
                 min_real_px=5.0):
    """Closed loop with the true clicks: sign agreement, median ratio, overshoot.

    The cumulative image shift is measured two ways: ``net`` — one phase
    correlation between the first and last frame (the robust estimate of the
    accumulated shift, and the one the gate uses), and ``sum`` — the sum of the
    per-step estimates (the harness's action-following form, noisier because 40
    estimates of ~0.5 px are summed). Both are reported; the gate uses ``net``.
    """
    rows = []
    cmd_rows = []
    for e, s in windows:
        seq, info = _rollout_seq(packed, model, e, s, K, steps, seed=seeds[0],
                                 device=device)
        real = _real_seq(packed, e, s, K, steps)
        n = min(len(seq), len(real))
        if info.get("dx"):
            # the model's *commanded* shift (the warp it applies) vs the real
            # heading-derived cumulative shift: noise-free, and independent of how
            # much the stochastic content dilutes a phase-correlation estimate
            g = packed.episodes[e]["frame_offset"] + s + K - 1
            P = _P_CACHE.get("P")
            cmd_rows.append((float(sum(info["dx"][:steps])),
                             float(P[g + steps] - P[g])))
        m_sum, mcurve = _cum_shift(seq[:n])
        r_sum, rcurve = _cum_shift(real)
        m_net = estimate_shift_x(seq[0], seq[n - 1])
        r_net = estimate_shift_x(real[0], real[n - 1])
        q = max(1, n // 4)
        over = float(np.abs(mcurve[-q:]).mean() - np.abs(rcurve[-q:]).mean())
        rows.append((m_sum, r_sum, m_net, r_net, over))

    def _agg(idx_m, idx_r, tag, thr):
        pairs = [(r[idx_m], r[idx_r]) for r in rows if abs(r[idx_r]) >= thr]
        if not pairs:
            return {"n": 0}
        pred = np.array([p[0] for p in pairs]); real = np.array([p[1] for p in pairs])
        ratios = np.abs(pred) / np.maximum(1e-6, np.abs(real))
        sa = np.sign(pred) == np.sign(real)
        return {
            "n": int(len(pairs)),
            "sign_agreement": float(sa.mean()),
            "median_ratio": float(np.median(ratios)),
            "mean_ratio": float(ratios.mean()),
            f"pass_sign_0.8_{tag}": bool(sa.mean() >= 0.8),
            f"pass_ratio_band_{tag}": bool(0.6 <= np.median(ratios) <= 1.4),
        }

    # The per-step phase-correlation estimator has a ~0.4 px noise floor, and 40 of
    # them are summed, so the statistic is only meaningful where the real turn is
    # clearly above that floor. Report the primary threshold plus a sweep.
    out = _agg(0, 1, "sum", min_real_px)
    out["net_estimate"] = _agg(2, 3, "net", min_real_px)
    out["by_threshold"] = {str(t): _agg(0, 1, "sum", float(t)) for t in (2, 5, 10, 15)}
    out["overshoot_px_per_step_tail"] = float(np.mean([r[4] for r in rows])) if rows else None
    if cmd_rows:
        pred = np.array([c[0] for c in cmd_rows]); real = np.array([c[1] for c in cmd_rows])
        m = np.abs(real) >= min_real_px
        if m.any():
            out["commanded"] = {
                "n": int(m.sum()),
                "sign_agreement": float((np.sign(pred[m]) == np.sign(real[m])).mean()),
                "median_ratio": float(np.median(np.abs(pred[m]) / np.maximum(1e-6, np.abs(real[m])))),
                "mean_ratio": float(np.abs(pred[m]).mean() / np.abs(real[m]).mean()),
            }
    return out


def g1_readout(packed, model, sidecar, table, K, H, windows, device="cuda"):
    """Readout direction sign accuracy on click-onset windows (P4 style).

    Truth = the sign of the real net turn over H steps (from the denoised heading
    table). Oracle = sign of the reachable door nearest to the click — the engine's
    resolution rule, i.e. the ceiling. Also reported: oracle-vs-truth, which is how
    well the engine rule itself predicts the real turn at these windows.
    """
    core = model if isinstance(model, SteeringModel) else getattr(model, "core", None)
    if core is None:
        return {"note": "model has no steering readout"}
    P = global_prefix(table)
    off = {e["episode_index"]: e["frame_offset"] for e in packed.episodes}
    pr, orc, tru = [], [], []
    for e, s in windows:
        t = s + K - 1
        g = off[e] + t
        lo = max(0, s - 8)
        hist = np.nan_to_num(packed.action_vectors(
            packed.labels[off[e] + lo: off[e] + t + 1]), nan=0.0).astype(np.float32)
        ctx = frames_to_tensor(
            np.asarray(packed.images[off[e] + s: off[e] + t + 1]))[None].to(device)
        feats = click_features(
            torch.from_numpy(hist)[None].float().to(device))[:, -K:]
        with torch.no_grad():
            _, meta = core.predict(ctx, features=feats, return_shift=True)
            pred = float(meta["inc"].sum(dim=1).item())
        truth = float(P[g + H] - P[g])
        click_x = float(hist[-1, 0])
        reach = [c for c in sidecar.candidates_at(e, t) if c["reachable"]]
        door = np.nan
        if reach:
            near = min(reach, key=lambda c: abs(c["screenX"] - click_x))
            # the engine turns the view *toward* that door, so the content shift the
            # readout predicts (positive = content right = view turned left) is the
            # opposite of the door's screenX sign
            door = -np.sign(near["screenX"])
        if abs(truth) < 1.0:
            continue
        pr.append(np.sign(pred)); orc.append(door); tru.append(np.sign(truth))
    pr, orc, tru = np.array(pr), np.array(orc), np.array(tru)
    ok = np.isfinite(orc)
    out = {
        "n": int(len(pr)),
        "readout_vs_truth_sign_acc": float((pr == tru).mean()) if len(pr) else None,
        "readout_vs_oracle_sign_acc": float((pr[ok] == orc[ok]).mean()) if ok.any() else None,
        "oracle_vs_truth_sign_acc": float((orc[ok] == tru[ok]).mean()) if ok.any() else None,
    }
    return out


def run_g1(packed, model, sidecar, table, K, H, device="cuda", steps=40,
           seeds=(None,), n_ctx=300, split="val"):
    _P_CACHE["P"] = global_prefix(table)
    ctxs, n_ep = near_door_contexts(packed, sidecar, K, H, split=split)
    ctxs = ctxs[:n_ctx]
    wins = onset_windows(packed, table, K, split=split)[:n_ctx]
    swins = sustained_windows(packed, table, K, split=split, steps=steps,
                              n=min(n_ctx, 300))
    out = {
        "n_contexts": len(ctxs), "n_context_episodes": n_ep,
        "n_onset_windows": len(wins), "n_sustained_windows": len(swins),
        "counterfactual": g1_counterfactual(packed, model, sidecar, K, H, ctxs,
                                            device, seeds),
        "shift_ratio": g1_shift_ratio(packed, model, table, K, H, wins, device, seeds),
        "sustained": g1_sustained(packed, model, K, steps, swins, device, seeds),
        "sustained_all_onsets": g1_sustained(packed, model, K, steps, wins[:200],
                                             device, seeds),
        "readout": g1_readout(packed, model, sidecar, table, K, H, wins[:300], device),
    }
    cf = out["counterfactual"][str(seeds[0])]
    sus = out["sustained"]
    out["passes"] = {
        "G1_counterfactual_both_directions_near>=0.70":
            bool(cf.get("both_correct_directions_near", 0) >= 0.70),
        "G1_shift_ratio>=0.5": out["shift_ratio"]["pass_ratio_0.5"],
        "G1_sustained_sign>=0.8": bool(sus.get("pass_sign_0.8_sum", False)),
        "G1_sustained_ratio_band": bool(sus.get("pass_ratio_band_sum", False)),
    }
    return out


# --------------------------------------------------------------------------- #
# Quality / stability metrics (used by G2, G3 and the test-split run)
# --------------------------------------------------------------------------- #

def frame_stats(u8: np.ndarray) -> dict:
    """Sharpness / Laplacian variance / colour / collapse statistics for one frame."""
    cs = color_stats(u8)
    return {
        "sharpness": sharpness(u8),
        "laplacian_var": laplacian_var(u8),
        "mean_rgb": cs["mean_rgb"],
        "std_rgb": cs["std_rgb"],
        "mean": cs["mean_all"],
        "black_frac": black_frac(u8),
        "saturated_frac": saturated_frac(u8),
    }


def _mean_stats(list_of_stats):
    if not list_of_stats:
        return {}
    keys = list_of_stats[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in list_of_stats], dtype=np.float64)
        out[k] = [float(v) for v in vals.mean(axis=0)] if vals.ndim > 1 else float(vals.mean())
    return out


def _sample_windows(packed, split, K, steps, n, seed=0, min_change=None):
    ep, start = packed.enumerate_windows(split, K, 1, require_safe=True)
    keep = [(int(e), int(s)) for e, s in zip(ep.tolist(), start.tolist())
            if s + K + steps + 2 < packed.episodes[int(e)]["length"]]
    if len(keep) <= n:
        return keep
    idx = np.linspace(0, len(keep) - 1, n).astype(int)
    return [keep[i] for i in idx]


def reveal_and_steady_windows(packed, split, K, n=24, device="cpu"):
    """Split val windows into "something new revealed" vs "steady walking"."""
    ep, start = packed.enumerate_windows(split, K, 2, require_safe=True)
    cand = np.arange(len(ep))
    rng = np.random.default_rng(0)
    if len(cand) > 4000:
        cand = np.sort(rng.choice(cand, 4000, replace=False))
    scored = []
    for i in cand.tolist():
        e, s = int(ep[i]), int(start[i])
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        a = np.asarray(packed.images[off + s + K - 1])
        b = np.asarray(packed.images[off + s + K])
        scored.append((frame_change(a, b), e, s))   # metrics accept uint8 frames
    scored.sort(key=lambda t: -t[0])
    hi = scored[:n]
    lo = scored[len(scored) // 2: len(scored) // 2 + n]
    return hi, lo


def quality_metrics(packed, model, K, windows, steps=1, device="cuda", seeds=(0, 1, 2),
                    baseline_ckpt=None):
    """One-step quality: the generator vs real frames (and vs the Stage 4 baseline)."""
    baseline = None
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        baseline, _ = load_model(baseline_ckpt, device)
    real_s, gen_s, base_s, div, l1_gen, l1_base = [], [], [], [], [], []
    for e, s in windows:
        real = _real_seq(packed, e, s, K, steps)
        gt = frames_to_tensor(np.asarray(real)[1:2])
        real_s.append(frame_stats(np.asarray(real[1])))
        samples = []
        for seed in seeds:
            seq, _ = _rollout_seq(packed, model, e, s, K, steps, seed=seed, device=device)
            g = frames_to_tensor(np.asarray(seq)[1:2])
            samples.append(g)
            gen_s.append(frame_stats(seq[1]))
            l1_gen.append(float(torch.nn.functional.l1_loss(g, gt)) / 2.0)
        div.append(seed_diversity(samples))
        if baseline is not None:
            from eval_harness import features_from_history
            off = packed.episodes[e]["frame_offset"]
            h = np.nan_to_num(packed.action_vectors(
                packed.labels[off + max(0, s - 8): off + s + K]), nan=0.0).astype(np.float32)
            ctx = frames_to_tensor(np.asarray(
                packed.images[off + s: off + s + K]))[None].to(device)
            a = torch.from_numpy(h[-K:])[None].to(device)
            bp = model_predict(baseline, ctx, a)
            bp = bp[0] if isinstance(bp, tuple) else bp
            base_s.append(frame_stats(_u8(bp)))
            l1_base.append(float(torch.nn.functional.l1_loss(bp[None].cpu(), gt)) / 2.0)
    rs = _mean_stats(real_s); gs = _mean_stats(gen_s)
    out = {"n": len(windows), "real": rs, "generated": gs,
           "sharpness_ratio_gen_over_real": gs["sharpness"] / max(1e-9, rs["sharpness"]),
           "seed_diversity": float(np.mean(div)),
           "l1_gen": float(np.mean(l1_gen))}
    if base_s:
        bs = _mean_stats(base_s)
        out["baseline"] = bs
        out["sharpness_ratio_gen_over_baseline"] = gs["sharpness"] / max(1e-9, bs["sharpness"])
        out["l1_baseline"] = float(np.mean(l1_base))
    return out


def diversity_reveal_vs_steady(packed, model, K, device="cuda", seeds=(0, 1, 2, 3),
                               n=24, split="val"):
    hi, lo = reveal_and_steady_windows(packed, split, K, n)
    out = {}
    for tag, wins in (("reveal", hi), ("steady", lo)):
        divs = []
        for _, e, s in wins:
            samples = []
            for sd in seeds:
                seq, _ = _rollout_seq(packed, model, e, s, K, 1, seed=sd, device=device)
                samples.append(frames_to_tensor(np.asarray(seq)[1:2])[0])
            divs.append(seed_diversity(samples))
        out[tag] = float(np.mean(divs))
    out["n"] = len(hi)
    out["ratio_reveal_over_steady"] = out["reveal"] / max(1e-9, out["steady"])
    out["pass_reveal_gt_steady"] = bool(out["reveal"] > out["steady"])
    return out


def rollout_quality(packed, model, K, steps=60, n=30, seed=0, device="cuda",
                    every=5, action_source="true", baseline_ckpt=None, split="val"):
    """Closed-loop rollout statistics vs horizon (sharpness, colour, motion, change)."""
    clips = _sample_windows(packed, split, K, steps, n)
    baseline = None
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        baseline, _ = load_model(baseline_ckpt, device)
    base_sharp = []
    curves = {k: [] for k in ("sharpness", "mean", "black_frac", "saturated_frac",
                              "frame_change", "motion", "shift")}
    real_curves = {k: [] for k in ("sharpness", "mean", "shift", "frame_change")}
    for e, s in clips:
        seq, _ = _rollout_seq(packed, model, e, s, K, steps, seed=seed, device=device)
        real = _real_seq(packed, e, s, K, steps)
        if baseline is not None:
            # the Stage 4 deterministic model rolled the same way: its frames are the
            # "blurry average" the sharpness comparison is meant to expose
            rec = packed.episodes[e]
            off = rec["frame_offset"]
            acts = np.nan_to_num(packed.action_vectors(
                packed.labels[off: off + rec["length"]]), nan=0.0).astype(np.float32)
            bf = frames_to_tensor(np.asarray(
                packed.images[off + s: off + s + K])).to(device)
            for t in range(min(steps, rec["length"] - s - K)):
                from eval_harness import features_from_history
                bp = model_predict(baseline, bf[-K:].unsqueeze(0),
                                   torch.from_numpy(np.ascontiguousarray(
                                       acts[s + t: s + t + K]))[None].to(device))
                bp = bp[0] if isinstance(bp, tuple) else bp
                bf = torch.cat([bf, bp], dim=0)
            for j in range(K, len(bf), every):
                base_sharp.append(sharpness(_u8(bf[j])))
        n_use = min(len(seq), len(real))
        for j in range(1, n_use, every):
            st = frame_stats(seq[j])
            curves["sharpness"].append(st["sharpness"])
            curves["mean"].append(st["mean"])
            curves["black_frac"].append(st["black_frac"])
            curves["saturated_frac"].append(st["saturated_frac"])
            curves["frame_change"].append(frame_change(seq[j], seq[j - 1]))
            curves["motion"].append(abs(estimate_shift_x(seq[j - 1], seq[j])))
            curves["shift"].append(estimate_shift_x(seq[0], seq[j]))
            rst = frame_stats(real[j])
            real_curves["sharpness"].append(rst["sharpness"])
            real_curves["mean"].append(rst["mean"])
            real_curves["shift"].append(estimate_shift_x(real[0], real[j]))
            real_curves["frame_change"].append(frame_change(real[j], real[j - 1]))
    out = {
        "n": len(clips),
        "sharpness_ratio": float(np.mean(curves["sharpness"]) / max(1e-9, np.mean(real_curves["sharpness"]))),
        "mean_colour_gap": float(np.mean(curves["mean"]) - np.mean(real_curves["mean"])),
        "mean_black_frac": float(np.mean(curves["black_frac"])),
        "mean_saturated_frac": float(np.mean(curves["saturated_frac"])),
        "mean_frame_change": float(np.mean(curves["frame_change"])),
        "real_frame_change": float(np.mean(real_curves["frame_change"])),
        "mean_motion_px": float(np.mean(curves["motion"])),
        "sharpness_track": [float(v) for v in curves["sharpness"]],
        "mean_track": [float(v) for v in curves["mean"]],
        "real_sharpness_mean": float(np.mean(real_curves["sharpness"])),
        "real_mean": float(np.mean(real_curves["mean"])),
    }
    if base_sharp:
        out["baseline_sharpness_mean"] = float(np.mean(base_sharp))
        out["sharpness_ratio_gen_over_baseline"] = (
            float(np.mean(curves["sharpness"])) / max(1e-9, float(np.mean(base_sharp))))
    return out


# --------------------------------------------------------------------------- #
# G2 / G3 / G4
# --------------------------------------------------------------------------- #

def run_g2(packed, model, sidecar, table, K, H, device="cuda", seeds=(0, 1, 2),
           baseline_ckpt=None, n_ctx=200, split="val"):
    """Steering with the generator in the loop + quality, diversity, 60-step track."""
    out = {"split": split}
    out["steering"] = run_g1(packed, model, sidecar, table, K, H, device, 40, seeds,
                             n_ctx, split=split)
    wins = onset_windows(packed, table, K, split=split)[:100]
    out["quality"] = quality_metrics(packed, model, K, wins, steps=1, device=device,
                                     seeds=tuple(s for s in seeds if s is not None) or (0,),
                                     baseline_ckpt=baseline_ckpt)
    out["diversity"] = diversity_reveal_vs_steady(packed, model, K, device=device,
                                                 split=split)
    out["rollout60"] = rollout_quality(packed, model, K, steps=60, n=30, seed=0,
                                       device=device, baseline_ckpt=baseline_ckpt,
                                       split=split)
    return out


def _rollout_scripted(packed, model, e, s, K, steps, script, device="cuda", seed=0):
    """Closed-loop rollout under a scripted click pattern [(frame, x), ...]."""
    from eval_harness import features_from_history
    from steer import click_features
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    lo = max(0, s - 8)
    hist = [r for r in np.nan_to_num(packed.action_vectors(
        packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)]
    frames = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K])).to(device)
    held = (0.0, 0.0)
    out = []
    with torch.no_grad():
        for t in range(steps):
            for cf, cx in script:
                if cf == t:
                    held = (float(cx), 1.0)
            hist.append(np.array(held, dtype=np.float32))
            feats = click_features(torch.from_numpy(np.stack(hist))[None].float()
                                   .to(device))[:, -K:]
            nxt = model_predict(model, frames[-K:].unsqueeze(0), None, feats, seed)
            nxt = nxt[0] if isinstance(nxt, tuple) else nxt
            frames = torch.cat([frames, nxt], dim=0)
            out.append(nxt[0])
    return torch.stack(out, dim=0)


def run_g3(packed, model, K, device="cuda", steps=300, n=30, seed=0,
           script_frames=40, split="val"):
    """300-frame closed-loop stability in three scenarios."""
    clips = _sample_windows(packed, split, K, steps, n)
    scenarios = {}
    tracks = {k: [] for k in ("sharpness", "mean", "frame_change", "motion",
                              "black_frac", "saturated_frac", "shift")}
    real_ref = {"sharpness": [], "mean": [], "frame_change": []}
    for e, s in clips:
        gen_no, _ = rollout_v2(packed, model, e, s, K, steps,
                               action_override=torch.zeros(1, K, 2, device=device),
                               seed=seed, device=device)
        gen_no = _to_uint8_seq(gen_no)
        sc = [(f, 0.8 if (f // script_frames) % 2 == 0 else -0.8)
              for f in range(0, steps, script_frames)]
        gen_sc = _to_uint8_seq(_rollout_scripted(packed, model, e, s, K, steps, sc,
                                                device=device, seed=seed))
        real = _to_uint8_seq(frames_to_tensor(_real_seq(packed, e, s, K, steps)).to(device))
        for tag, seq in (("none", gen_no), ("scripted", gen_sc), ("real", real)):
            rec = {"sharpness": [], "mean": [], "frame_change": [], "motion": [],
                   "black_frac": [], "saturated_frac": [], "shift": []}
            m = min(len(seq), len(real))
            for j in range(1, m, 10):
                st = frame_stats(seq[j])
                rec["sharpness"].append(st["sharpness"])
                rec["mean"].append(st["mean"])
                rec["black_frac"].append(st["black_frac"])
                rec["saturated_frac"].append(st["saturated_frac"])
                rec["frame_change"].append(frame_change(seq[j], seq[j - 1]))
                rec["motion"].append(abs(estimate_shift_x(seq[j - 1], seq[j])))
                rec["shift"].append(estimate_shift_x(seq[0], seq[j]))
            agg = {k: float(np.mean(v)) for k, v in rec.items()}
            agg["last10_frame_change"] = float(np.mean(rec["frame_change"][-5:])) if rec["frame_change"] else 0.0
            agg["last10_motion_px"] = float(np.mean(rec["motion"][-5:])) if rec["motion"] else 0.0
            agg["tail_sharpness"] = float(np.mean(rec["sharpness"][-5:])) if rec["sharpness"] else 0.0
            scenarios.setdefault(tag, []).append(agg)
    out = {"n_clips": len(clips), "steps": steps}
    for tag, rows in scenarios.items():
        out[tag] = _mean_stats(rows)
    real_change = float(out.get("real", {}).get("frame_change", 0.0))
    for tag in list(out):
        if tag in ("n_clips", "steps", "real"):
            continue
        d = out[tag]
        d["frozen"] = bool(real_change > 0 and d["last10_frame_change"] < 0.2 * real_change)
        d["collapsed"] = bool(d["black_frac"] > 0.5 or d["saturated_frac"] > 0.5)
        d["sharpness_ratio_vs_real"] = float(
            d["sharpness"] / max(1e-9, out["real"]["sharpness"]))
        d["motion_ratio_vs_real"] = float(
            d["motion"] / max(1e-9, out["real"]["motion"]))
    # steering check for the scripted pattern (clicks at 0, 40, 80, ...)
    hits = []
    for e, s in clips[:12]:
        sc = [(f, 0.8 if (f // script_frames) % 2 == 0 else -0.8)
              for f in range(0, steps, script_frames)]
        seq = _to_uint8_seq(_rollout_scripted(packed, model, e, s, K, steps, sc,
                                              device=device, seed=seed))
        for f, x in sc:
            hi = min(f + script_frames, len(seq) - 1)
            if hi - f < 10:
                continue
            sh = estimate_shift_x(seq[max(0, f - 1)], seq[hi])
            hits.append(np.sign(sh) == -np.sign(x))     # x>0 (right) -> content left
    out["scripted_steering_sign_acc"] = float(np.mean(hits)) if hits else None
    out["scripted_steering_n"] = len(hits)
    return out


def _to_uint8_seq(t):
    return _u8(t)


def run_g4(model, packed, K, device="cuda", steps=1000, warmup=50):
    """The full per-frame loop at batch 1: fps, latency percentiles, peak VRAM."""
    import time
    from eval_harness import features_from_history
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    e, s = int(ep[0]), int(start[0])
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    hist = np.nan_to_num(packed.action_vectors(
        packed.labels[off + s - 8: off + s + K]), nan=0.0).astype(np.float32)
    frames = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K])).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lat = []
    n = warmup + steps
    with torch.no_grad():
        for t in range(n):
            hist = np.concatenate([hist, np.zeros((1, 2), np.float32)], axis=0)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            feats = features_from_history(hist, K, device)
            nxt = model_predict(model, frames[-K:].unsqueeze(0), None, feats, t % 17)
            nxt = nxt[0] if isinstance(nxt, tuple) else nxt
            frames = torch.cat([frames, nxt], dim=0)
            _ = (_u8(nxt[0]))                     # decode to a displayable uint8 image
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if t >= warmup:
                lat.append(dt * 1000.0)
    lat = np.array(lat)
    out = {
        "device": device, "steps": len(lat),
        "mean_ms": float(lat.mean()), "p95_ms": float(np.percentile(lat, 95)),
        "worst_ms": float(lat.max()), "fps_mean": float(1000.0 / lat.mean()),
        "fps_p95": float(1000.0 / np.percentile(lat, 95)),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
    }
    out["pass_fps_15"] = bool(out["fps_mean"] >= 15.0)
    out["pass_vram_3gb"] = bool(out["peak_vram_mb"] is None or out["peak_vram_mb"] < 3000)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _resolve_KH(payload):
    mc = payload.get("model_config", {})
    return mc.get("K", 4), mc.get("H", 10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5 gate runner")
    ap.add_argument("--gates", default="g1",
                    choices=["g1", "g2", "g3", "g4", "quality"])
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-contexts", dest="n_contexts", type=int, default=300)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", default="", help="comma-separated sampling seeds")
    ap.add_argument("--baseline", default=None,
                    help="Stage 4 baseline checkpoint for the quality comparison")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed)
    sidecar = load_analysis(packed)
    model, payload = load_model(args.checkpoint, device)
    K, H = _resolve_KH(payload)
    table = load_target_table(packed, args.packed, "heading")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x != "") or (None,)

    if args.gates == "g1":
        out = run_g1(packed, model, sidecar, table, K, H, device, args.steps, seeds,
                     args.n_contexts, split=args.split)
    elif args.gates == "g2":
        out = run_g2(packed, model, sidecar, table, K, H, device, seeds,
                     baseline_ckpt=args.baseline, n_ctx=args.n_contexts,
                     split=args.split)
    elif args.gates == "g3":
        out = run_g3(packed, model, K, device=device, steps=args.steps,
                     n=min(30, args.n_contexts), split=args.split)
    elif args.gates == "g4":
        out = run_g4(model, packed, K, device=device, steps=1000)
    else:
        wins = onset_windows(packed, table, K, split=args.split)[:args.n_contexts]
        out = quality_metrics(packed, model, K, wins, steps=1, device=device,
                             seeds=tuple(s for s in seeds if s is not None) or (0, 1, 2),
                             baseline_ckpt=args.baseline)
    out.update({"checkpoint": args.checkpoint, "split": args.split, "K": K, "H": H})
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
