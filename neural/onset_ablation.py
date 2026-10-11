"""Step 3 of the action-conditioning investigation: horizon ablation at turn onset.

The one-step ablation (eval_harness 2.4b) is blind to the action because the
action's effect on the NEXT frame is ~0.03 px. This script instead scores each
action variant (true / zeroed / shuffled / shifted ±1 frame) by the error of a
~10-step rollout from a turn-onset window — a context that shows NO turn yet, so
only the action can tell the model a turn is coming.

If a model is genuinely action-conditioned, the true actions should roll out
clearly better than zeroed/shuffled/shifted here.

Run: .venv/bin/python onset_ablation.py --packed data/pilot_v1_64x48 \
       --checkpoint runs/<name>/best.pt --steps 10 --max-windows 300
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from analysis import load_analysis
from dataset import PackedDataset, frames_to_tensor
from eval_harness import _action_at
from model import load_checkpoint

_ONSET_BEFORE = np.deg2rad(0.2)
_ONSET_AHEAD = np.deg2rad(0.5)


def _rollout_seq(packed, model, e, s, K, steps, acts, device):
    e2 = packed.episodes[e]
    off = e2["frame_offset"]
    ctx_u8 = packed.images[off + s: off + s + K]
    frames = frames_to_tensor(ctx_u8).to(device)
    preds = []
    for t in range(steps):
        c = frames[t: t + K].unsqueeze(0)
        a = torch.from_numpy(acts[t: t + K].astype(np.float32)).to(device).unsqueeze(0)
        nxt = model.predict(c, a)
        frames = torch.cat([frames, nxt], dim=0)
        preds.append(nxt[0])
    return torch.stack(preds, dim=0)  # [steps,3,H,W]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-windows", dest="max_windows", type=int, default=300)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed, in_memory=True)
    model, payload = load_checkpoint(args.checkpoint, device)
    K = payload["model_config"]["K"]
    steps = args.steps
    sidecar = load_analysis(packed)

    ep, start = packed.enumerate_windows(args.split, K, steps, require_safe=True)
    ep = ep.astype(np.int64)
    start = start.astype(np.int64)

    # onset windows: context not turning, but a turn develops in the horizon
    onset = []
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        g = s + K - 1
        b = sidecar.heading_change(e, g)
        if np.isnan(b) or abs(b) >= _ONSET_BEFORE:
            continue
        ahead = max((abs(sidecar.heading_change(e, g + j)) for j in range(1, steps + 1)),
                    default=0.0)
        if ahead >= _ONSET_AHEAD:
            onset.append((e, s))
    if len(onset) > args.max_windows:
        idx = np.sort(np.random.default_rng(0).choice(len(onset), args.max_windows,
                                                      replace=False))
        onset = [onset[i] for i in idx]
    print(f"onset windows: {len(onset)}", flush=True)

    # full action sequences (one frame of slack on each side for the ±1 shifts)
    n_a = steps + K - 1
    seqs = []
    for e, s in onset:
        full = np.stack([_action_at(packed, e, s - 1 + j) for j in range(n_a + 2)])
        seqs.append(full)
    seqs = np.stack(seqs)  # [N, n_a+2, 2]

    true = seqs[:, 1: 1 + n_a]
    earlier = seqs[:, 0: n_a]
    later = seqs[:, 2: 2 + n_a]
    zeroed = np.zeros_like(true)
    shuffled = true[np.random.default_rng(0).permutation(len(onset))]

    variants = {"true": true, "zeroed": zeroed, "shuffled": shuffled,
                "earlier": earlier, "later": later}
    acc = {name: np.zeros(steps) for name in variants}
    with torch.no_grad():
        for i, (e, s) in enumerate(onset):
            gt = frames_to_tensor(
                packed.images[packed.episodes[e]["frame_offset"] + s + K:
                              packed.episodes[e]["frame_offset"] + s + K + steps]
            ).to(device)
            for name, arr in variants.items():
                preds = _rollout_seq(packed, model, e, s, K, steps, arr[i], device)
                err = (preds - gt).abs().mean(dim=(1, 2, 3)).cpu().numpy() / 2.0
                acc[name] += err

    print(f"{'variant':9s} {'L1@step1':>8s} {'L1@step5':>8s} {'L1@step10':>8s} "
          f"{'mean(1..10)':>11s}")
    for name in variants:
        a = acc[name] / len(onset)
        print(f"{name:9s} {a[0]:8.4f} {a[4]:8.4f} {a[-1]:8.4f} {a.mean():11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
