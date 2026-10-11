# Stage 3 Report — Python environment, dataset packager, training-data loader

Brief: `stage3-prompt-python-packager-loader.txt`. Date: 2026-10-11.

This stage built the first piece of the **neural half** of NeuralGameEngine: a
working PyTorch environment, a packager that turns the engine's episodes into
compact training arrays, and a loader that serves training windows and clips. No
model is trained (beyond the mandated smoke test). The engine and pilot data were
**not** modified; git was **not** touched.

The definitive data contract is
[`DATASET-SCHEMA.md`](../../engine/docs/DATASET-SCHEMA.md). The definitive
consumer documentation for this stage is [`../README.md`](../README.md).

---

## 1. Environment (verification item 1)

| | |
|---|---|
| Python | 3.14.4 (system default) |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| cuDNN | 9.2.4 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, capability 12.0 (`sm_120`) |
| `sm_120` in build's supported arch list | **yes** (`sm_75/80/86/90/100/120`) |
| Real GPU computation | matmul finite ✓, conv + backward grads finite ✓ |

The default Python 3.14.4 is already supported by PyTorch 2.14.1, so **no
user-space Python manager was needed** — a plain `venv` at `neural/.venv` (git-
ignored) is enough. The RTX 5070 (Blackwell) needs a CUDA ≥ 12.8 build; the
driver is CUDA 13.0, so `cu130` is the correct wheel. Pinned in
`requirements.txt` / `requirements.lock`.

## 2. Packed formats and sizes

The packer converts `engine/data/pilot_v1` into two packed sets:

| Packed set | `images.bin` | total (incl. ~60 MB `analysis.jsonl`) |
|---|---|---|
| `data/pilot_v1_96x72` | 1,825,741,056 B (1.83 GB) | ~1.9 GB |
| `data/pilot_v1_64x48` | 811,440,192 B (811 MB) | ~0.84 GB |

Format (per packed set; see README §2):

- `images.bin` — uint8 RGB, row-major `[n, y, x, c]`, memory-mapped.
- `labels.bin` — structured, one 14-byte row per frame: `episode_index u4`,
  `frame_index u4`, `actionX f4` (NaN = null), `actionHasClicked i1` (0/1, −1 =
  null), `viewUnsafe i1`.
- `index.json` — one record per episode (id, split, seed, source path, length,
  `frame_offset`, truncated).
- `meta.json` — provenance (schemaVersion, plan name, configHash,
  engineSourceHash, packer version, resolution, downsample method) + stats.
- `analysis.jsonl` — sidecar of debug/analysis fields (persona, raw clicks,
  player pose, target/candidates, …), keyed by `(episode, frame)`, not loaded by
  default.

Downsampling is a **box/area filter**: Pillow `Image.Resampling.BOX` applied to
the RGB channels (320×240 RGBA masters; alpha dropped). Decode runs in parallel
over 24 workers; packing the pilot took ~31 s (96×72) and ~18 s (64×48).

## 3. Loader interface (brief)

- `PackedDataset` — memory-mapped view; `get_clip(episode, start, L)` for
  contiguous clips (rollout), `window(episode, start, K, H)` for training
  windows, `get_clip` for random access by `(split, episode, frame)`.
- `WindowDataset` — `torch.utils.data.Dataset` over a split's valid windows
  (random access, reproducible; works with DataLoader workers/pinning/batching).
- `iter_windows(...)` — deterministic ordered iteration (evaluation).
- `count_valid_windows(split, K, H, require_safe=…)`.

**Action representation** (documented in code + README): each action is a float32
vector `[actionX, actionHasClicked]`; `actionX[i]` drives the transition
`i → i+1`, `actionHasClicked ∈ {0,1}` (NaN on the final frame). Before the first
click `actionHasClicked == 0` and `actionX == 0`.

## 4. Dataset statistics

149 episodes packed (145 complete + 4 aborted-but-truncated; the 5th aborted
episode was deleted by the engine's failsafe). 88,047 frames.

| split | episodes | frames |
|---|---|---|
| train | 119 | 70,580 |
| val | 15 | 9,000 |
| test | 15 | 8,467 |

Per-channel pixel mean / std (RGB, computed over all frames):

| res | mean | std |
|---|---|---|
| 96×72 | 127.365 / 122.811 / 97.401 | 63.913 / 60.477 / 42.137 |
| 64×48 | 127.313 / 122.760 / 97.351 | 63.574 / 60.154 / 41.907 |

- `viewUnsafe` frames: **357** (0.41%), across **90 / 149** episodes (60%) — matches
  the Stage 2 report (357 / 88,047).
- `actionHasClicked == false`: **8,603** frames = **9.8%** of non-final frames —
  matches the Stage 2 report.

Valid windows and how many `viewUnsafe` removes (all splits):

| K | H | total | safe (default) | removed by `viewUnsafe` |
|---|---|---|---|---|
| 4 | 1 | 87,451 | 86,609 | 842 |
| 4 | 4 | 87,004 | 85,807 | 1,197 |
| 8 | 1 | 86,855 | 85,541 | 1,314 |
| 8 | 4 | 86,408 | 84,746 | 1,662 |
| 16 | 1 | 85,663 | 83,431 | 2,232 |
| 16 | 4 | 85,216 | 82,642 | 2,574 |

## 5. Loader throughput (verification item 7)

At 96×72, K=4:

- **~10,000 windows/s** single process with the split kept in RAM
  (`PackedDataset(in_memory=True)`), the documented fast path (the pilot fits in
  RAM).
- ~3,500 windows/s single process from the memory-mapped file (default).

Windows are tiny, so `DataLoader` worker IPC overhead dominates and
`num_workers=0` is actually the fastest config; workers/pinning are exercised by
the smoke test and work correctly. Either way the loader comfortably out-runs a
small model, so it is not the bottleneck.

## 6. Verification results

| # | check | result |
|---|---|---|
| 1 | environment / real GPU | ✓ (see §1) |
| 2 | action alignment | ✓ sampled windows: `actionX[j] == manifest actionX[j] == manifest controlSignalX[j+1]`; `actionHasClicked` matches |
| 3 | rules | ✓ no unsafe frame in returned windows; no window spans episodes; no episode in two splits; valid-window count == independent manifest-derived count |
| 4 | pixel round trip | ✓ 12 random frames: packed == independent BOX downsample of the source PNG |
| 5 | totals | ✓ 88,047 frames; 357 `viewUnsafe` — matches Stage 2 |
| 6 | determinism + reuse | ✓ subset packed twice → identical sha256 of `images.bin`/`labels.bin`/`index.json`/`meta.json`; re-run reports `up_to_date`, skips all 149 |
| 7 | throughput + smoke | ✓ ~10k windows/s; 40 steps of a trivial conv net on the GPU: loss 0.282 → 0.023 |

All 9 pytest tests pass (`python -m pytest -q`).

## 7. Shortcuts taken

- Determinism is verified by packing a **4-episode subset** twice (byte-identical)
  rather than the full pilot twice, since byte-determinism is per-episode and
  order-independent; reuse/idempotency is verified on the full pilot by re-running
  the packer (skips all 149). This avoids writing a second 1.8 GB copy.
- `in_memory=True` loads the **whole** packed set into RAM (not a per-split
  slice). Fine for the pilot (1.8 GB); for the future bulk dataset, keep the
  memory-mapped default.
- `pad_first` (repeat-frame start padding) is implemented but not exercised by a
  test (the default no-padding path is the tested one).
- The review image is generated with matplotlib; I did **not** judge it visually
  (that is yours — see §9).

## 8. `.gitignore` edits

Added to the repo-root `.gitignore`:

```
/neural/.venv/
/neural/data/
/neural/**/__pycache__/
/neural/.pytest_cache/
*.pyc
```

## 9. Needs user decision / please look at

1. **Review image** — `neural/data/pilot_v1_96x72/_review/windows.png`. Look for:
   the frames should read as one smooth continuous sequence; the `actionX` value
   should change at (or just before) the frame where the turn visibly begins (the
   turn starts one frame *after* the action appears); the two excluded windows
   should visibly be the camera pressed against or through a wall (red-bordered
   frame = `viewUnsafe`).
2. Nothing else is blocking. The next stage (a baseline next-frame model) can read
   this report + `neural/README.md` to build on `WindowDataset`.

## 10. Files added / changed (Stage 3)

Added (all under `/neural`): `packer.py`, `dataset.py`, `review.py`,
`gpu_check.py`, `requirements.txt`, `requirements.lock`, `conftest.py`,
`tests/test_environment.py`, `tests/test_loader.py`, `tests/test_packer.py`,
`docs/STAGE3-REPORT.md`, `README.md` (replaced the placeholder stub).

Changed outside `/neural`: `.gitignore` only (see §8). No change to git history,
the engine, or the pilot data.
