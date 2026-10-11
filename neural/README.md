# /neural — Python environment, dataset packager, and training-data loader

This is the **neural half** of NeuralGameEngine: the Python/PyTorch side that
turns the classical engine's episodes (PNG frames + per-frame labels) into
compact, fast-to-read training arrays and serves training windows to a model. It
is fully independent of `/engine`; it only *reads* the engine's dataset and never
calls back into it.

No model is trained here yet (that is the next stage). What exists:

- `packer.py` — packs a source dataset into memory-mapped arrays at a chosen
  resolution.
- `dataset.py` — the loader: `PackedDataset` (random access / clips) and
  `WindowDataset` (training windows), plus the action-alignment contract.
- `review.py` — renders `windows.png`, the one review image for human judgment.
- `gpu_check.py` — records versions and proves PyTorch runs real GPU work.
- `tests/` — the Stage 3 verification suite (alignment, rules, round trip,
  totals, determinism/reuse, throughput + smoke).

The packed data lives in `data/` (gitignored) and is **derived and rebuildable**:
the PNG masters under `engine/data/` remain the source of truth.

---

## 1. Environment

The machine has a laptop RTX 5070 (NVIDIA Blackwell, compute capability 12.0 =
`sm_120`) and driver CUDA 13.0. PyTorch must therefore be a **CUDA 12.8+** build;
we pin the `cu130` wheel.

```bash
cd neural
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt      # torch==2.14.1+cu130, numpy, pillow, matplotlib, pytest
```

`requirements.txt` pins the top-level packages (and points pip at the PyTorch
wheel index for the GPU build); `requirements.lock` is the full `pip freeze`
resolved set for exact rebuilds.

Verify it actually computes on the GPU (not just `is_available()`):

```bash
.venv/bin/python gpu_check.py
# -> gpu_name "NVIDIA GeForce RTX 5070 Laptop GPU", gpu_arch sm_120,
#    arch_in_supported_list true, matmul_finite true, conv_backward_grad_ok true
```

## 2. Pack a dataset

```bash
.venv/bin/python packer.py \
    --src ../engine/data/pilot_v1 \
    --out data/pilot_v1_96x72 --width 96 --height 72
.venv/bin/python packer.py \
    --src ../engine/data/pilot_v1 \
    --out data/pilot_v1_64x48 --width 64 --height 48
```

- `--src` is a dataset root whose `train/val/test` subdirectories contain episode
  directories (`frames/*.png` + `manifest.json`). Non-episode files (report,
  `aborted.jsonl`, `_review`) are ignored; only episodes marked `complete: true`
  are included (episodes truncated by the clipping failsafe are valid, just
  shorter). Any manifest with a `schemaVersion` other than `2` is **refused**.
- Resolution is a parameter (`--width`/`--height`); any size works without code
  changes.
- Downsampling is a **box/area filter**: Pillow `Image.Resampling.BOX` applied to
  the RGB channels (the 320×240 masters are RGBA; alpha is dropped). Deterministic
  and round-trip verified.
- Decoding runs in parallel (`--workers`, default 24). The packer is deterministic
  (same input → identical bytes) and **incremental**: re-running skips already
  packed episodes and only appends new ones.

### Packed format

```
<out>/
  meta.json          provenance (schemaVersion, configHash, engineSourceHash,
                     packer version, resolution, downsample method) + statistics
  index.json         one record per episode: episode_id, split, seed, source_path,
                     length, frame_offset, truncated
  images.bin         uint8 RGB, row-major [n, y, x, c]  (memory-mapped)
  labels.bin         structured array, one 14-byte row per frame:
                     episode_index u4, frame_index u4, actionX f4 (NaN = null),
                     actionHasClicked i1 (0/1, -1 = null), viewUnsafe i1
  analysis.jsonl     sidecar of debug/analysis fields (persona, raw clicks, player
                     pose, target/candidates, ...) keyed by (episode, frame) —
                     NOT loaded by default
```

The pilot packs to **1.83 GB** at 96×72 (`images.bin`) and **811 MB** at 64×48.
`analysis.jsonl` is ~60 MB. Everything is read via `np.memmap`, so the loader
never loads the whole dataset into RAM (unless you ask it to — see §4).

## 3. The loader

```python
from dataset import PackedDataset, WindowDataset

d = PackedDataset("data/pilot_v1_96x72")          # memory-mapped
d.num_frames, d.width, d.height                     # 88047, 96, 72
d.episodes_by_split["train"][0]                     # first train episode record

# A contiguous clip (L frames + labels) from one episode:
frames, labels = d.get_clip(0, 100, 32)            # episode 0, frames 100..131

# A training window (K context + H target frames):
w = d.window(0, 100, K=4, H=1)
w.context        # [K, 72, 96, 3] uint8
w.target         # [H, 72, 96, 3] uint8
w.context_action # [K, 2] float32 [actionX, actionHasClicked]
w.target_action  # [H, 2] float32  (action driving into each target)

# Random-access clips for rollout: d.get_clip(split, episode, start, length)
```

`WindowDataset` is a standard `torch.utils.data.Dataset` (works with `DataLoader`
workers, pinning, batching):

```python
ds = WindowDataset(d, "train", K=4, H=1, normalize=True)
batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                              num_workers=2, pin_memory=True)))
# batch["context"] [32, 4, 3, 72, 96] float in [-1,1]; batch["target"] [32, 1, 3, 72, 96]
# batch["context_action"] [32, 4, 2]; batch["target_action"] [32, 1, 2]
```

Deterministic ordered iteration over valid windows (evaluation):

```python
for w in iter_windows(d, "val", K=4, H=1):
    ...
```

### The action-alignment rule (read this first)

`actionX[i]` is the action that drives the transition from frame `i` to `i+1`
(it equals `controlSignalX[i+1]`). So **the action paired with frame `i` is
`actionX[i]`**, and the action that drives into target frame `t` is
`actionX[t-1]` (the previous frame's action). Training must use `actionX`, never
`controlSignalX`.

Each action is returned as a float32 vector `[actionX, actionHasClicked]`:

- `actionX` ∈ [-1, 1]; `NaN` only on an episode's final frame (never a context
  frame).
- `actionHasClicked` ∈ {0.0, 1.0}; `NaN` only on the final frame.
- Before the first accepted click the engine is in its "never clicked" default
  mode: `actionHasClicked == 0` and `actionX == 0`.

Images are `uint8` in `[0, 255]` at the array level; `normalize=True` (default in
`WindowDataset`) returns `[-1, 1]` floats — the diffusion convention.

### Window validity

A window never crosses an episode boundary or a split. By default any window
containing a `viewUnsafe` frame (context or target) is dropped; pass
`require_safe=False` to keep them for analysis. By default a window needs a full
`K` frames of context (`pad_first=True` pads the start of an episode by repeating
frame 0). The final frame can only ever be a target, never a context frame.

`count_valid_windows(split, K, H)` reports how many windows exist at a setting
(and `require_safe=False` vs `True` gives the number removed by `viewUnsafe`).

## 4. Throughput

Measured at 96×72, K=4 (see `tests/test_loader.py::test_throughput`):

- **~12,000 windows/s** single process with the split kept in RAM
  (`PackedDataset(..., in_memory=True)` — the documented fast path; the pilot's
  1.8 GB fits in RAM).
- ~3,500 windows/s single process from the memory-mapped file (default).
- Windows are tiny, so `DataLoader` worker IPC overhead dominates and
  `num_workers=0` is actually the fastest here; workers/pinning are still exercised
  by the smoke test and work correctly.

A small model consumes far less than this, so the loader is not the bottleneck.

## 5. The review image

```bash
.venv/bin/python review.py --packed data/pilot_v1_96x72 \
    --out data/pilot_v1_96x72/_review/windows.png
```

`windows.png` (drawn **through the loader**, never by reading PNGs directly) shows
6 training windows chosen where the action changes (just after a click), each as
K=4 context frames followed by the target, with `actionX`/`actionHasClicked`
printed under each frame, plus 2 windows that were **excluded** because they
contain a `viewUnsafe` frame (marked with a red border). It is for your visual
judgment; see the Stage 3 report for what to look for.

## 6. Run the checks

```bash
.venv/bin/python -m pytest -q          # 9 tests: alignment, rules, round trip,
                                       # totals, determinism/reuse, throughput+smoke, environment
```
