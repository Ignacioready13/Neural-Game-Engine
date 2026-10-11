"""
Dataset packager: engine episodes (PNG + manifest) -> compact memory-mapped
training arrays at a chosen resolution.

Reads a source dataset produced by the engine (a directory with train/val/test
splits, each containing episode directories with frames/*.png + manifest.json),
and writes a *packed set*:

    <out>/
      meta.json          provenance + statistics + format descriptor
      index.json         one record per episode (split, seed, offsets, ...)
      images.bin         uint8 RGB, row-major [n, y, x, c]
      labels.bin         structured array, one row per frame
      analysis.jsonl     sidecar of debug/analysis fields, keyed by (episode, frame)

The packed set is memory-mapped at read time (no full-RAM load), deterministic
(packing the same input twice gives identical bytes), and incremental (re-running
skips episodes already packed, appending only new ones).

This module is read-only toward the source dataset; it never writes into it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

PACKER_VERSION = "1.0.0"
SCHEMA_VERSION = 2  # the schemaVersion DATASET-SCHEMA.md describes; refuse others

# --- on-disk label dtype (must stay stable; loader rebuilds it from meta.json) ---
LABEL_FIELDS = [
    ("episode_index", "u4"),
    ("frame_index", "u4"),
    ("actionX", "f4"),            # NaN == null (episode's final frame)
    ("actionHasClicked", "i1"),   # 0/1, -1 == null (final frame)
    ("viewUnsafe", "i1"),         # 0/1
]
LABEL_DTYPE = np.dtype(LABEL_FIELDS, align=False)

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #

def downsample(png_path: str, size: tuple[int, int]) -> np.ndarray:
    """Decode a source PNG and box/area-downsample it to (height, width) as uint8 RGB.

    The masters are RGBA (opaque). We drop alpha (take the RGB channels) and
    resize with Pillow's BOX filter, which averages the source pixels covered by
    each destination pixel — the same box/area downsampling family the engine's
    own review images used. Deterministic for a given source PNG and size.
    """
    img = Image.open(png_path)
    rgb = np.asarray(img)[..., :3]           # H x W x 3 uint8, alpha dropped
    small = Image.fromarray(rgb).resize((size[1], size[0]), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.uint8)


def _decode_episode(args) -> np.ndarray:
    """Worker entry point: decode + downsample one episode's frames."""
    image_paths, size = args
    frames = [downsample(p, size) for p in image_paths]
    return np.stack(frames, axis=0)  # (L, H, W, 3) uint8


# --------------------------------------------------------------------------- #
# Episode discovery / validation
# --------------------------------------------------------------------------- #

def read_manifest(episode_dir: str) -> dict:
    path = os.path.join(episode_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"episode has no manifest.json: {episode_dir}")
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    v = m.get("schemaVersion")
    if v != SCHEMA_VERSION:
        raise ValueError(
            f"manifest {path} has schemaVersion {v!r}, expected {SCHEMA_VERSION}; "
            f"refusing to pack (see DATASET-SCHEMA.md)"
        )
    return m


def discover_episodes(dataset_root: str):
    """Return (complete episodes sorted by split then seed, problems list).

    Ignores anything that is not an episode (plan/report/aborted/review files and
    directories). An episode whose manifest is missing, unreadable, or not marked
    `complete: true` is skipped (it is a partial episode) and reported in `problems`.
    """
    episodes: list[dict] = []
    problems: list[str] = []
    for split in SPLITS:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            episode_dir = os.path.join(split_dir, name)
            if not os.path.isfile(os.path.join(episode_dir, "manifest.json")):
                continue  # not an episode (report.json, aborted.jsonl, _review, ...)
            try:
                m = read_manifest(episode_dir)
            except ValueError:
                raise  # schema mismatch is fatal
            except Exception as e:  # unreadable manifest -> treat as partial
                problems.append(f"{episode_dir}: unreadable manifest ({e})")
                continue
            if m.get("complete") is not True:
                problems.append(f"{episode_dir}: not complete (partial), skipped")
                continue
            episodes.append({"split": split, "dir": episode_dir, "manifest": m})
    episodes.sort(key=lambda e: (e["split"], e["manifest"]["seed"]))
    return episodes, problems


# --------------------------------------------------------------------------- #
# Frame record extraction (training fields + analysis sidecar fields)
# --------------------------------------------------------------------------- #

_ANALYSIS_FRAME_FIELDS = (
    ("clickX", "clickX"),
    ("clickAccepted", "clickAccepted"),
    ("playerX", "playerX"),
    ("playerZ", "playerZ"),
    ("playerHeading", "playerHeading"),
    ("currentEntityId", "currentEntityId"),
)

_ANALYSIS_DEBUG_FIELDS = (
    ("playerOmega", "playerOmega"),
    ("lastClickedX", "lastClickedX"),
    ("targetDoorwayId", "targetDoorwayId"),
    ("targetPointX", "targetPointX"),
    ("targetPointZ", "targetPointZ"),
    ("targetPhase", "targetPhase"),
    ("collisionEvents", "collisionEvents"),
    ("wallOverlap", "wallOverlap"),
    ("candidates", "candidates"),
    ("persona", "persona"),
    ("intentDoorwayId", "intentDoorwayId"),
    ("slid", "slid"),
)


def extract_frame_records(manifest: dict):
    """Return (labels rows, analysis rows) for one episode.

    labels: numpy structured array of LABEL_DTYPE, one row per frame.
    analysis: list of dicts for the sidecar (analysis fields only; not model inputs).
    """
    frames = manifest["frames"]
    debug = manifest.get("debugFrames", [])
    labels = np.zeros(len(frames), dtype=LABEL_DTYPE)
    analysis = []
    for i, fr in enumerate(frames):
        ax = fr.get("actionX")
        ahc = fr.get("actionHasClicked")
        labels[i]["frame_index"] = fr["frameIndex"]
        labels[i]["actionX"] = np.float32(np.nan) if ax is None else np.float32(ax)
        labels[i]["actionHasClicked"] = -1 if ahc is None else (1 if ahc else 0)
        labels[i]["viewUnsafe"] = 1 if fr.get("viewUnsafe") else 0
        row = {"episode": 0, "frame": fr["frameIndex"]}  # episode filled in later
        for src, dst in _ANALYSIS_FRAME_FIELDS:
            row[dst] = fr.get(src)
        if i < len(debug):
            d = debug[i]
            for src, dst in _ANALYSIS_DEBUG_FIELDS:
                row[dst] = d.get(src)
        analysis.append(row)
    return labels, analysis


# --------------------------------------------------------------------------- #
# Packed-set I/O
# --------------------------------------------------------------------------- #

def _ensure_file_bytes(path: str, nbytes: int):
    """Grow a file to at least nbytes (with zeros) without touching existing bytes."""
    if os.path.exists(path):
        cur = os.path.getsize(path)
        if cur < nbytes:
            with open(path, "r+b") as f:
                f.truncate(nbytes)
    else:
        with open(path, "wb") as f:
            f.truncate(nbytes)


def load_index(out_dir: str):
    p = os.path.join(out_dir, "index.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["episodes"]


def write_index(out_dir: str, episodes: list[dict]):
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=1)
        f.write("\n")


def write_meta(out_dir: str, meta: dict):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, allow_nan=True)
        f.write("\n")


def compute_stats(out_dir: str, n: int, height: int, width: int, episodes: list[dict]) -> dict:
    """Per-split counts, per-channel mean/std, viewUnsafe + hasClicked shares."""
    per_split = {s: {"frames": 0, "episodes": 0} for s in SPLITS}
    for e in episodes:
        per_split[e["split"]]["episodes"] += 1
        per_split[e["split"]]["frames"] += e["length"]

    labels = np.memmap(
        os.path.join(out_dir, "labels.bin"), mode="r", dtype=LABEL_DTYPE, shape=(n,)
    )
    unsafe_count = int((labels["viewUnsafe"] == 1).sum())
    hasclicked_false = int((labels["actionHasClicked"] == 0).sum())
    null_rows = int((labels["actionHasClicked"] == -1).sum())

    images = np.memmap(
        os.path.join(out_dir, "images.bin"), mode="r", dtype=np.uint8, shape=(n, height, width, 3)
    )
    # streaming per-channel mean/std, memory-bounded
    chunk = 2048
    total = 0
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        flat = images[start:end].reshape(-1, 3).astype(np.float64)
        total += flat.shape[0]
        d = flat - mean
        mean += d.sum(axis=0) / total
        d2 = flat - mean
        m2 += (d * d2).sum(axis=0)
    std = np.sqrt(m2 / total)

    return {
        "num_frames": n,
        "num_episodes": len(episodes),
        "per_split": per_split,
        "channel_mean": [float(m) for m in mean],
        "channel_std": [float(s) for s in std],
        "viewUnsafe_frames": unsafe_count,
        "viewUnsafe_share": unsafe_count / n if n else 0.0,
        "actionHasClicked_false_frames": hasclicked_false,
        "actionHasClicked_false_share":
            hasclicked_false / (n - null_rows) if n > null_rows else 0.0,
    }


# --------------------------------------------------------------------------- #
# Main pack routine
# --------------------------------------------------------------------------- #

def pack(dataset_root: str, out_dir: str, width: int, height: int, workers: int) -> dict:
    dataset_root = os.path.abspath(dataset_root)
    out_dir = os.path.abspath(out_dir)
    size = (height, width)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Discover source episodes (deterministic order) and read their manifests.
    episodes, problems = discover_episodes(dataset_root)
    src_meta = []
    for e in episodes:
        m = e["manifest"]
        image_paths = [os.path.join(e["dir"], fr["imagePath"]) for fr in m["frames"]]
        labels, analysis = extract_frame_records(m)
        src_meta.append({
            "split": e["split"],
            "dir": e["dir"],
            "seed": m["seed"],
            "episode_id": m.get("episodeId", f"{m.get('seed')}"),
            "length": len(m["frames"]),
            "truncated": bool(m.get("truncated")),
            "source_path": os.path.relpath(e["dir"], dataset_root),
            "image_paths": image_paths,
            "labels": labels,
            "analysis": analysis,
            "config_hash": m.get("configHash"),
            "engine_source_hash": m.get("engineSourceHash"),
        })

    # 2. Determine what is already packed (incremental / idempotent).
    existing = load_index(out_dir)
    existing_by_id = {e["episode_id"]: e for e in existing} if existing else {}
    done = list(existing) if existing else []

    stale = [eid for eid, e in existing_by_id.items()
             if any(sm["episode_id"] == eid and sm["length"] != e["length"] for sm in src_meta)]
    if stale:
        raise RuntimeError(
            f"packed episodes changed length ({stale}); delete {out_dir} and re-pack"
        )

    to_pack = [sm for sm in src_meta if sm["episode_id"] not in existing_by_id]

    if not to_pack:
        return {"out_dir": out_dir, "up_to_date": True, "skipped": len(done),
                "problems": problems}

    # 3. Assign global episode indices and frame offsets. Already-packed episodes
    #    keep their offsets; new episodes are appended after them.
    episode_index = len(done)
    frame_offset = sum(e["length"] for e in done)
    for sm in to_pack:
        sm["episode_index"] = episode_index
        sm["frame_offset"] = frame_offset
        episode_index += 1
        frame_offset += sm["length"]

    total_frames = frame_offset  # existing + new
    num_episodes = len(done) + len(to_pack)

    # 4. Extend (or create) the memory-mapped arrays without disturbing old bytes.
    img_path = os.path.join(out_dir, "images.bin")
    lbl_path = os.path.join(out_dir, "labels.bin")
    _ensure_file_bytes(img_path, total_frames * height * width * 3)
    _ensure_file_bytes(lbl_path, total_frames * LABEL_DTYPE.itemsize)

    images = np.memmap(img_path, mode="r+", dtype=np.uint8,
                       shape=(total_frames, height, width, 3))
    labels = np.memmap(lbl_path, mode="r+", dtype=LABEL_DTYPE, shape=(total_frames,))

    # 5. Decode PNGs in parallel; write each new episode to its fixed offset.
    decode_jobs = [(sm["image_paths"], size) for sm in to_pack]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sm, frames in zip(to_pack, pool.map(_decode_episode, decode_jobs)):
            off = sm["frame_offset"]
            images[off: off + sm["length"]] = frames
            sm["labels"]["episode_index"] = sm["episode_index"]
            labels[off: off + sm["length"]] = sm["labels"]

    images.flush()
    labels.flush()

    # 6. Write index.json (existing + appended, in order).
    index_rows = [dict(e) for e in done]
    for sm in to_pack:
        index_rows.append({
            "episode_index": sm["episode_index"],
            "episode_id": sm["episode_id"],
            "split": sm["split"],
            "seed": sm["seed"],
            "source_path": sm["source_path"],
            "length": sm["length"],
            "frame_offset": sm["frame_offset"],
            "truncated": sm["truncated"],
        })
    write_index(out_dir, index_rows)

    # 7. Append analysis sidecar rows for the new episodes (deterministic order).
    with open(os.path.join(out_dir, "analysis.jsonl"), "a", encoding="utf-8") as f:
        for sm in to_pack:
            for row in sm["analysis"]:
                row["episode"] = sm["episode_index"]
                f.write(json.dumps(row, allow_nan=True) + "\n")

    # 8. Compute stats and write meta.json.
    stats = compute_stats(out_dir, total_frames, height, width, index_rows)
    config_hashes = sorted({sm["config_hash"] for sm in src_meta})
    source_hashes = sorted({sm["engine_source_hash"] for sm in src_meta})
    meta = {
        "format_version": 1,
        "packer_version": PACKER_VERSION,
        "source_dataset": dataset_root,
        "plan_name": os.path.basename(dataset_root.rstrip("/")),
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hashes,
        "engine_source_hash": source_hashes,
        "resolution": [width, height],
        "downsample_method":
            "Pillow Image.Resampling.BOX (box/area filter) applied to RGB (alpha "
            "channel dropped); source PNGs are 320x240 RGBA",
        "image_dtype": "uint8",
        "image_shape": [total_frames, height, width, 3],
        "label_fields": [list(f) for f in LABEL_FIELDS],
        "actionX_null_representation": "NaN (float32)",
        "actionHasClicked_null_representation": -1,
        "stats": stats,
        "problems": problems,
    }
    write_meta(out_dir, meta)

    return {"out_dir": out_dir, "up_to_date": False, "packed_new": len(to_pack),
            "skipped": len(done), "total_frames": total_frames,
            "total_episodes": num_episodes, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack engine episodes into compact training arrays")
    ap.add_argument("--src", required=True, help="source dataset root (e.g. engine/data/pilot_v1)")
    ap.add_argument("--out", required=True, help="output packed-set directory")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args(argv)
    result = pack(args.src, args.out, args.width, args.height, args.workers)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
