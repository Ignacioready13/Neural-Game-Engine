#!/usr/bin/env python
"""Build the seed-context cache the player starts every run from.

The player seeds a run with K real dataset frames (plus the real click-signal
history that precedes them) so the model starts from a plausible place. Those clips
are the *only* dataset pixels the player ever shows. Rather than memory-map the 18 GB
packed set every time you want to play, this caches a few hundred of them — from all
three splits, labelled — into one small ``models/seeds.npz``. The player then runs
without the packed set at all (pass ``--packed ...`` to override).

Usage::

    ./memcap.sh --mem 12G -- .venv/bin/python build_seeds.py \
        --packed data/bulk_v1_96x72 --per-split val=80,train=40,test=40

Frames are cached at the packed set's own resolution; a model registered at a
different resolution should be played with ``--packed`` instead.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(ROOT, "models", "seeds.npz")


def build(packed_path: str, per_split: dict[str, int], K: int, hist_extra: int,
          seed: int = 0) -> dict:
    from dataset import PackedDataset
    packed = PackedDataset(packed_path)
    W, H = packed.width, packed.height
    rng = np.random.default_rng(seed)

    frames, hists, eps, starts, splits = [], [], [], [], []
    for split, n in per_split.items():
        if n <= 0:
            continue
        cand = [e for e in packed.episodes_by_split[split] if e["length"] > K + 20]
        if not cand:
            raise SystemExit(f"split {split!r} has no episodes longer than {K + 20} frames")
        # sample distinct episodes where possible, so the bank is not one clip repeated
        order = rng.permutation(len(cand))
        picked = []
        for i in order:
            picked.append(cand[int(i)])
            if len(picked) >= n:
                break
        while len(picked) < n:                       # small splits: allow repeats
            picked.append(cand[int(rng.integers(len(cand)))])
        for e in picked:
            L = int(e["length"])
            start = int(rng.integers(0, max(1, int(0.1 * L))))
            off = int(e["frame_offset"])
            frames.append(np.asarray(packed.images[off + start: off + start + K]))
            lo = max(0, start - hist_extra)
            h = packed.action_vectors(packed.labels[off + lo: off + start + K])
            h = np.nan_to_num(h, nan=0.0).astype(np.float32)
            if h.shape[0] < hist_extra + K:          # episode begins right at the start
                pad = np.zeros((hist_extra + K - h.shape[0], 2), np.float32)
                h = np.concatenate([pad, h], axis=0)
            hists.append(h)
            eps.append(int(e["episode_index"]))
            starts.append(start)
            splits.append(split)
        print(f"  {split}: {len(picked)} seeds", flush=True)

    meta = {
        "packed": os.path.relpath(os.path.abspath(packed_path), ROOT),
        "K": int(K),
        "hist_extra": int(hist_extra),
        "width": int(W),
        "height": int(H),
        "per_split": {k: int(v) for k, v in per_split.items()},
    }
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "hist": np.stack(hists).astype(np.float32),
        "episode": np.asarray(eps, np.int64),
        "start": np.asarray(starts, np.int64),
        "split": np.asarray(splits, dtype="<U8"),
        "packed": np.asarray(meta["packed"]),
        "K": np.asarray(meta["K"], np.int64),
        "hist_extra": np.asarray(meta["hist_extra"], np.int64),
        "width": np.asarray(meta["width"], np.int64),
        "height": np.asarray(meta["height"], np.int64),
        "per_split": np.asarray(json.dumps(meta["per_split"])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the player's seed-context cache")
    ap.add_argument("--packed", default=os.path.join(ROOT, "data", "bulk_v1_96x72"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--per-split", default="val=80,train=40,test=40",
                    help="how many seed clips to cache per split")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hist-extra", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    per_split = {}
    for part in args.per_split.split(","):
        if not part.strip():
            continue
        s, n = part.split("=")
        per_split[s.strip()] = int(n)

    print(f"building seeds from {args.packed} (K={args.K}, hist_extra={args.hist_extra})",
          flush=True)
    data = build(args.packed, per_split, args.K, args.hist_extra, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    n = data["frames"].shape[0]
    mb = os.path.getsize(args.out) / 2 ** 20
    print(f"wrote {args.out}: {n} seeds, {data['frames'].shape[1]} frames each at "
          f"{int(data['width'])}x{int(data['height'])}, {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
