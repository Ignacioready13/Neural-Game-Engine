"""Stage 4B: ONE sample grid image for the user's visual glance (brief Section 4).

Drawn at random from a packed set — random episodes from train/val/test, random
mid-episode positions, plus a few click-counterfactual moments that show two
reachable doors on opposite sides of screen centre (so the user can see the
"visually meaningful choice" situations the bulk is meant to cover). Frames are
shown at reduced size (the packed resolution, nearest-enlarged) with a small
caption of split / episode / frame.

This is a *look* aid only (variety of rooms and hallways; nothing broken) — the
selection is not judged here. Read-only through the loader; no model, no PNG
decoding.

Run: .venv/bin/python sample_grid.py --packed data/bulk_v1_96x72 \
       --out review/bulk_v1_sample.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import load_analysis
from dataset import PackedDataset
from probe_click import _select_contexts

_K = 4
_H = 1


def _qualifying(packed, sidecar, split):
    """Qualifying counterfactual contexts (reuses the Stage 4 probe selector)."""
    return _select_contexts(packed, sidecar, _K, _H, max_per_ep=10 ** 9, split=split)


def _random_safe_frames(packed, split, n, rng):
    """n random frames, mid-episode, no viewUnsafe frame, distinct episodes preferred."""
    eps = list(packed.episodes_by_split[split])
    rng.shuffle(eps)
    out = []
    for e in eps:
        if len(out) >= n:
            break
        off, L = e["frame_offset"], e["length"]
        unsafe = packed.labels["viewUnsafe"][off:off + L].astype(bool)
        lo, hi = int(0.1 * L), int(0.8 * L)
        cand = [g for g in range(lo, hi) if not unsafe[g]]
        if not cand:
            continue
        g = int(rng.choice(cand))
        out.append((e["episode_index"], g))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-split", type=int, default=8)
    ap.add_argument("--n-qual", type=int, default=3, help="qualifying-context frames per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    packed = PackedDataset(args.packed)
    sidecar = load_analysis(packed)
    W, H = packed.width, packed.height

    picks = []  # (split, episode, frame, kind)
    for split in ("train", "val", "test"):
        ctx = _qualifying(packed, sidecar, split)
        ctx = [c for c in ctx if not packed.labels[packed.episodes[c[0]]["frame_offset"]
                                                 + (c[1] + _K - 1)]["viewUnsafe"]]
        nq = min(args.n_qual, len(ctx), args.per_split)
        if nq:
            idx = rng.choice(len(ctx), nq, replace=False)
            for i in idx:
                e, s = ctx[int(i)][0], ctx[int(i)][1]
                picks.append((split, e, s + _K - 1, "two doors"))
        picks += [(split, e, g, "random")
                  for e, g in _random_safe_frames(packed, split, args.per_split - nq, rng)]

    ncols = 6
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 1.75))
    axes = np.atleast_1d(axes).ravel()
    scale = 2 if W <= 96 else 1
    for ax in axes:
        ax.axis("off")
    for ax, (split, e, g, kind) in zip(axes, picks):
        off = packed.episodes[e]["frame_offset"]
        img = np.asarray(packed.images[off + g])
        big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        ax.imshow(big)
        eid = packed.episodes[e]["episode_id"]
        ax.set_title(f"{split}  {eid}  f{g}  [{kind}]", fontsize=6)
    fig.suptitle(f"Stage 4B sample — {os.path.basename(packed.path)} "
                 f"({len(picks)} frames: train/val/test, random + two-door choices)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out} ({len(picks)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
