"""Packer verification: pixel round trip (4), totals (5), determinism + reuse (6)."""

import hashlib
import json
import shutil

import numpy as np
import pytest

from packer import downsample, pack


def test_pixel_round_trip(packed96, src):
    from packer import downsample
    rng = np.random.default_rng(1)
    idx = rng.integers(0, packed96.num_frames, size=12)
    for i in idx:
        got = packed96.images[int(i)]  # [H, W, 3] uint8
        ep = int(packed96.labels["episode_index"][i])
        fi = int(packed96.labels["frame_index"][i])
        e = packed96.episodes[ep]
        png = src / e["source_path"] / "frames" / f"{fi:06d}.png"
        expected = downsample(str(png), (packed96.height, packed96.width))
        assert np.array_equal(got, expected), f"frame {i} (ep {ep} f{fi}) differs"


def test_totals(packed96, packed64):
    for d in (packed96, packed64):
        assert d.num_frames == 88047
        unsafe = int((d.labels["viewUnsafe"] == 1).sum())
        assert unsafe == 357, f"{d.path}: viewUnsafe {unsafe} != 357"
        # 149 complete+truncated episodes (145 complete + 4 truncated aborted)
        assert len(d.episodes) == 149


def test_determinism(src, tmp_path):
    """Pack a small subset twice; the packed bytes must be identical."""
    # pick 2 train + 1 val + 1 test episodes (deterministic)
    picks = []
    for split in ("train", "val", "test"):
        n = 2 if split == "train" else 1
        dirs = sorted(d for d in (src / split).iterdir()
                      if (d / "manifest.json").exists())
        picks.extend((split, d.name) for d in dirs[:n])

    sub = tmp_path / "sub"
    for split, name in picks:
        (sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / split / name, sub / split / name)

    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    pack(str(sub), str(out1), 96, 72, 4)
    pack(str(sub), str(out2), 96, 72, 4)

    for fn in ("images.bin", "labels.bin", "index.json", "meta.json"):
        h1 = hashlib.sha256((out1 / fn).read_bytes()).hexdigest()
        h2 = hashlib.sha256((out2 / fn).read_bytes()).hexdigest()
        assert h1 == h2, f"{fn} differs between two packs"


def test_reuse_skips_work(packed96, src):
    """Re-running the packer on an already-packed set must skip all work."""
    result = pack(str(src), packed96.path, packed96.width, packed96.height, 8)
    assert result["up_to_date"] is True
    assert result["skipped"] == 149
    assert result.get("packed_new", 0) == 0
