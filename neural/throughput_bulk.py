"""Stage 4B loader throughput on the bulk packed set (memory-mapped), bounded.

Measures windows/s for a few DataLoader configurations WITHOUT ever loading the
whole set into RAM (the 96x72 bulk set is ~18 GB). Reads a bounded number of
windows, then exits. Reports peak RSS.

Run under a cap, e.g.:
  neural/memcap.sh --mem 6G -- .venv/bin/python throughput_bulk.py --packed data/bulk_v1_96x72
"""

from __future__ import annotations

import argparse
import os
import resource
import time

from torch.utils.data import DataLoader

from dataset import PackedDataset, WindowDataset


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure(packed, split, K, H, batch_size, workers, target_windows):
    ds = WindowDataset(packed, split, K=K, H=H, normalize=True)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        shuffle=True, pin_memory=False)
    t0 = time.time()
    n = 0
    it = iter(loader)
    while n < target_windows:
        batch = next(it)
        n += batch["context"].shape[0]
    dt = time.time() - t0
    return n / dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/bulk_v1_96x72")
    ap.add_argument("--split", default="train")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=12288)
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed)  # memmap (default), never in_memory
    print(f"{args.packed}  frames={packed.num_frames}  ({peak_mb():.0f} MB RSS after open)")
    print(f"{'workers':>8} {'batch':>6} {'windows/s':>10} {'peak MB':>8}")
    for workers in (0, 2, 4):
        for batch in (64,):
            measure(packed, args.split, args.K, 1, batch, workers, args.warmup)  # warm caches
            wps = measure(packed, args.split, args.K, 1, batch, workers, args.windows)
            print(f"{workers:>8} {batch:>6} {wps:>10.0f} {peak_mb():>8.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
