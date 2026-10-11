"""P4: can a model read turn *direction* from pixels + actions, at play time?

A direct heading predictor (no image generation): K context frames + K context
actions in (the click lands on the final context action a_{K-1}), one signed
scalar out — the net heading change over the next H steps, supervised from the
sidecar (ground truth, in radians; positive = turning toward positive screenX).

This isolates the P2 question from image generation: if the direction is *not*
extractable at this resolution/capacity with a direct, low-dimensional target,
no frame generator will find it either. Evaluated on val click-onset windows
(click at a_{K-1} + a turn develops in the horizon) against:

  (a) chance, and the majority-class floor;
  (b) an ORACLE = sign of the nearest reachable door's screenX to the click X
      (the engine's rule — the ceiling); and
  (c) a near/far door-distance breakdown, plus the two-opposite-doors subset.

Run: .venv/bin/python head_predict.py --name hd64 --packed data/pilot_v1_64x48 \
       --mirror --max-steps 4000
     ... --packed data/pilot_v1_96x72 ...        (resolution sweep)
     ... --change ...                              (explicit ΔactionX input)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis import load_analysis
from dataset import PackedDataset
from model import ConvBlock, stack_context_action
from train import _set_seed

_ONSET_AHEAD = np.deg2rad(0.5)   # a turn must develop in the horizon
_THETA = np.deg2rad(0.5)         # |heading| < this = "none"
_DOOR_NEAR = 2.0                 # world units
_NEAR_SPLIT = 0.5                # near/far breakdown threshold


class HeadingHead(nn.Module):
    """Small conv encoder (the baseline U-Net's encoder) + global-pool + MLP -> 1."""

    def __init__(self, in_ch: int, base: int = 32, extra_dim: int = 0):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
        self.head = nn.Sequential(
            nn.Linear(base * 4 + extra_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        e = self.enc3(self.down2(self.enc2(self.down1(self.enc1(x)))))
        v = e.mean((2, 3))                    # global average pool
        if extra is not None:
            v = torch.cat([v, extra], dim=1)
        return self.head(v).squeeze(1)


def _click_onset(packed, sidecar, ep, start, K, H):
    out = np.zeros(len(ep), dtype=bool)
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        g = s + K - 1
        off = packed.episodes[int(e)]["frame_offset"]
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        ax0, ax1 = a[K - 2, 0], a[K - 1, 0]
        if np.isnan(ax0) or np.isnan(ax1) or abs(ax1 - ax0) < 1e-4:
            continue
        ahead = max((abs(sidecar.heading_change(e, g + j)) for j in range(1, H + 1)),
                    default=0.0)
        if ahead >= _ONSET_AHEAD:
            out[i] = True
    return out


def _net_heading(sidecar, e, g, H):
    h0 = sidecar.heading_at(e, g)
    h1 = sidecar.heading_at(e, g + H)
    if np.isnan(h0) or np.isnan(h1):
        return np.nan
    return float((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)


def _targets(sidecar, ep, start, K, H):
    y = np.zeros(len(ep), dtype=np.float32)
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        y[i] = _net_heading(sidecar, int(e), int(s) + K - 1, H)
    return y


def _make_batch(packed, ep, start, K, H, device, change):
    ctx, act, extra = [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K]
        ctx.append(fr.transpose(0, 3, 1, 2))
        a = packed.action_vectors(packed.labels[off + s: off + s + K])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        act.append(a)
        if change:
            extra.append(np.array([a[K - 1, 0] - a[K - 2, 0]], dtype=np.float32))
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    x = stack_context_action(ctx, act)
    ex = torch.from_numpy(np.stack(extra)).float().to(device) if change else None
    return x, ex


def _sample_batch(ep, start, onset, B, rng):
    on = np.where(onset)[0]
    off = np.where(~onset)[0]
    n_on = min(len(on), B // 2)
    n_off = B - n_on
    pick = list(rng.choice(on, n_on, replace=False)) + list(
        rng.choice(off, n_off, replace=(n_off > len(off))))
    pick = np.array(pick, dtype=np.int64)
    return ep[pick], start[pick], pick


def train(config):
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, H = config["K"], config["horizon"]
    change = config.get("change", False)

    packed = PackedDataset(config["packed"], in_memory=True)
    ep, start = packed.enumerate_windows("train", K, H, require_safe=True)
    ep = ep.astype(np.int64); start = start.astype(np.int64)
    sidecar = load_analysis(packed)
    onset = _click_onset(packed, sidecar, ep, start, K, H)
    y_all = _targets(sidecar, ep, start, K, H)
    keep = ~np.isnan(y_all)
    ep, start, onset, y_all = ep[keep], start[keep], onset[keep], y_all[keep]
    print(f"train windows {len(ep)}, onset {int(onset.sum())} "
          f"({onset.sum() / len(ep) * 100:.1f}%), change_input={change}", flush=True)

    model = HeadingHead(K * 5, config.get("base", 32), extra_dim=(1 if change else 0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump({**config, "n_train_windows": len(ep), "n_onset": int(onset.sum()),
               "n_params": sum(p.numel() for p in model.parameters())},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    t0 = time.time()
    for step in range(1, config["max_steps"] + 1):
        eb, sb, idx = _sample_batch(ep, start, onset, B, rng)
        yb = torch.from_numpy(y_all[idx]).float().to(device)
        x, ex = _make_batch(packed, eb, sb, K, H, device, change)
        if config.get("mirror"):
            m = torch.from_numpy(rng.random(B) < 0.5).to(device)
            mi = m.nonzero(as_tuple=False).squeeze(1)
            if len(mi):
                x[mi] = torch.flip(x[mi], dims=[-1])
                if ex is not None:
                    ex[mi] = -ex[mi]
                yb[mi] = -yb[mi]
        opt.zero_grad(set_to_none=True)
        pred = model(x, ex)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == config["max_steps"]:
            print(f"step {step} loss {float(loss.item()):.6f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    torch.save({"state_dict": model.state_dict(), "config": dict(config)},
               os.path.join(out_dir, "best.pt"))
    return out_dir


def evaluate(packed, model, sidecar, K, H, change):
    device = next(model.parameters()).device
    ep, start = packed.enumerate_windows("val", K, H, require_safe=True)
    ep = ep.astype(np.int64); start = start.astype(np.int64)
    onset = _click_onset(packed, sidecar, ep, start, K, H)
    ep, start = ep[onset], start[onset]

    rows = []   # (e, s, dh_true, dh_pred, clickx, oracle_sign, near, two_door)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ep), 128):
            eb, sb = ep[i:i + 128], start[i:i + 128]
            x, ex = _make_batch(packed, eb, sb, K, H, device, change)
            pred = model(x, ex).cpu().numpy()
            for j, (e, s) in enumerate(zip(eb.tolist(), sb.tolist())):
                g = s + K - 1
                dh = _net_heading(sidecar, int(e), int(g), H)
                if np.isnan(dh):
                    continue
                off = packed.episodes[int(e)]["frame_offset"]
                a = packed.action_vectors(packed.labels[off + s: off + s + K])
                clickx = float(np.nan_to_num(a[K - 1, 0], nan=0.0))
                cands = [c for c in sidecar.candidates_at(int(e), int(g)) if c["reachable"]]
                left = [c for c in cands if c["screenX"] < -0.05]
                right = [c for c in cands if c["screenX"] > 0.05]
                two = bool(left and right)
                if cands:
                    d = min(cands, key=lambda c: abs(c["screenX"] - clickx))
                    oracle = float(np.sign(d["screenX"]))
                    near = float(d["screenDistance"])
                else:
                    oracle, near = 0.0, float("nan")
                rows.append((int(e), int(s), dh, float(pred[j]), clickx, oracle, near, two))

    R = np.array([(r[2], r[3], r[5], r[6], r[7]) for r in rows], dtype=float)
    dh, pr, oracle, near, two = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4].astype(bool)
    n_ep = len({r[0] for r in rows})
    print(f"\n=== P4 heading predictor — {packed.path} (change={change}) ===")
    print(f"val click-onset windows: {len(rows)} across {n_ep} distinct episodes")

    def cls(v):
        return np.where(v < -_THETA, -1, np.where(v > _THETA, 1, 0))

    tc, pc = cls(dh), cls(pr)
    turn = tc != 0
    print(f"\nclass balance (true): left {int((tc == -1).sum())}  none "
          f"{int((tc == 0).sum())}  right {int((tc == 1).sum())}")

    # (a) vs chance
    major = max((tc == c).mean() for c in (-1, 0, 1))
    print(f"3-way accuracy: model {np.mean(pc == tc):.3f}   majority-class floor {major:.3f}")
    print(f"binary sign acc (on |dh|>theta, n={turn.sum()}): model "
          f"{np.mean(pc[turn] == tc[turn]):.3f}   chance 0.50")

    # (b) oracle
    oturn = oracle[turn] != 0
    print(f"oracle sign acc (n={oturn.sum()}): {np.mean(oracle[turn][oturn] == tc[turn][oturn]):.3f} "
          f" [ceiling]")
    print(f"model-vs-oracle agreement: {np.mean(np.sign(pc[turn]) == np.sign(oracle[turn])):.3f}")
    print(f"corr(pred, true dh): {np.corrcoef(pr, dh)[0, 1]:.3f}   "
          f"corr(pred, oracle): {np.corrcoef(pr, oracle)[0, 1]:.3f}")

    # (c) two-opposite-doors subset + door distance
    for name, mask in (("all", np.ones(len(dh), bool)),
                       ("two-opposite-doors", two & (turn))):
        m = mask & (turn)
        if m.sum() < 5:
            print(f"[{name}] n={m.sum()} too few")
            continue
        print(f"[{name}] n={m.sum()}  model sign acc {np.mean(pc[m] == tc[m]):.3f}  "
              f"oracle {np.mean(oracle[m] == tc[m]):.3f}")
        for lab, nm in (("near", near < _NEAR_SPLIT), ("far", ~(near < _NEAR_SPLIT))):
            mm = m & nm & ~np.isnan(near)
            if mm.sum() < 5:
                continue
            print(f"    [{lab} n={mm.sum()}] model {np.mean(pc[mm] == tc[mm]):.3f}  "
                  f"oracle {np.mean(oracle[mm] == tc[mm]):.3f}")

    # magnitude calibration on turn windows
    m = turn
    ratio = np.mean(np.abs(pr[m])) / max(1e-9, np.mean(np.abs(dh[m])))
    print(f"predicted-vs-real |heading| ratio (turn windows): {ratio:.3f}")
    return {"n": len(rows), "n_episodes": n_ep, "acc3": float(np.mean(pc == tc)),
            "bin_acc": float(np.mean(pc[turn] == tc[turn])),
            "oracle_acc": float(np.mean(oracle[turn] == tc[turn])),
            "ratio": float(ratio)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hd64")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--change", action="store_true", help="explicit ΔactionX input")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--batch", dest="batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    config = dict(vars(args))
    config["weight_decay"] = 0.0
    packed = PackedDataset(args.packed, in_memory=True)
    sidecar = load_analysis(packed)
    K, H = args.K, args.horizon

    out_dir = train(config)
    model = HeadingHead(K * 5, args.base, extra_dim=(1 if args.change else 0))
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location="cuda", weights_only=False)["state_dict"])
    model.to("cuda").eval()
    res = evaluate(packed, model, sidecar, K, H, args.change)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({**config, **res}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
