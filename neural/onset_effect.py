"""Step 0 of the action-conditioning investigation: quantify the true effect size.

Answers, using the sidecar + phase correlation on the packed 64x48 frames:

  * how many *pixels* a turn shifts the image at horizons 1 / 5 / 10 / 15 steps,
    compared with steady (no-turn) frames — i.e. whether a single-frame prediction
    even has a signal to learn;
  * the left/right balance of heading changes in the training data and in the
    counterfactual-eval contexts (the 12% left-turn result suggested a bias).

Read-only; prints a compact summary. Run:

    .venv/bin/python onset_effect.py --packed data/pilot_v1_64x48
"""

from __future__ import annotations

import argparse

import numpy as np

from analysis import load_analysis
from dataset import PackedDataset
from metrics import estimate_shift_x

_ONSET_DEG = 0.5          # |heading change| >= this = turning
_QUIET_DEG = 0.2          # |heading change| < this = steady
_HORIZONS = (1, 5, 10, 15)


def _find_onsets(packed, sidecar, split, max_n=200, quiet_before=3):
    """(episode, frame) where a sustained turn begins after a quiet stretch."""
    onsets = []
    for e in packed.episodes_by_split[split]:
        ei = e["episode_index"]
        L = e["length"]
        quiet = 0
        for g in range(0, L - max(_HORIZONS) - 1):
            d = sidecar.heading_change(ei, g)
            if np.isnan(d):
                continue
            if abs(d) < np.deg2rad(_QUIET_DEG):
                quiet += 1
            else:
                if quiet >= quiet_before:
                    onsets.append((ei, g))
                quiet = 0
    if len(onsets) > max_n:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(onsets), max_n, replace=False))
        onsets = [onsets[i] for i in idx]
    return onsets


def _cum_shift(packed, ep, g, t):
    e = packed.episodes[ep]
    off = e["frame_offset"]
    a = packed.images[off + g]        # [H,W,3] uint8
    b = packed.images[off + g + t]
    return estimate_shift_x(a, b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed, in_memory=True)
    sidecar = load_analysis(packed)

    onsets = _find_onsets(packed, sidecar, args.split)
    print(f"turn onsets sampled: {len(onsets)}")
    for t in _HORIZONS:
        sh = np.array([_cum_shift(packed, e, g, t) for e, g in onsets])
        print(f"  onset cum |shift| over {t:2d} steps: mean |.|={np.abs(sh).mean():.3f} "
              f"px  (mean {sh.mean():+.3f}, +right/-left)")

    # steady baseline: a frame in the middle of a long quiet stretch
    steady = []
    for e in packed.episodes_by_split[args.split]:
        ei = e["episode_index"]
        L = e["length"]
        for g in range(5, L - max(_HORIZONS) - 1):
            d = sidecar.heading_change(ei, g)
            if not np.isnan(d) and abs(d) < np.deg2rad(0.05):
                steady.append((ei, g))
        if len(steady) > 200:
            break
    for t in _HORIZONS:
        sh = np.array([_cum_shift(packed, e, g, t) for e, g in steady[:200]])
        print(f"  steady cum |shift| over {t:2d} steps: mean |.|={np.abs(sh).mean():.3f} px")

    # left/right balance of heading changes
    dc = []
    for e in packed.episodes_by_split[args.split]:
        ei = e["episode_index"]
        L = e["length"]
        for g in range(L - 1):
            d = sidecar.heading_change(ei, g)
            if not np.isnan(d):
                dc.append(d)
    dc = np.array(dc)
    pos = (dc > 0).mean()
    turn = np.abs(dc) >= np.deg2rad(_ONSET_DEG)
    print(f"heading-change balance (train): {pos*100:.1f}% positive, "
          f"{100-pos*100:.1f}% negative (n={len(dc)}); "
          f"turns |d|>=0.5deg: {turn.mean()*100:.1f}% of frames")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
