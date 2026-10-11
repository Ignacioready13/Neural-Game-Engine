# /neural — Python environment, dataset packager, and training-data loader

This is the **neural half** of NeuralGameEngine: the Python/PyTorch side that
turns the classical engine's episodes (PNG frames + per-frame labels) into
compact, fast-to-read training arrays and serves training windows to a model. It
is fully independent of `/engine`; it only *reads* the engine's dataset and never
calls back into it.

Stage 4 adds the first model and the reusable evaluation harness. What exists:

- `packer.py` — packs a source dataset into memory-mapped arrays at a chosen
  resolution.
- `dataset.py` — the loader: `PackedDataset` (random access / clips) and
  `WindowDataset` (training windows), plus the action-alignment contract.
- `model.py` — the baseline action-conditioned next-frame predictor (a small
  residual U-Net) and the `predict(context, action) -> next_frame` interface
  every future model must implement.
- `train.py` — config-driven training (seeded, checkpoints + loss curves).
- `train_multistep.py` — autoregressive multi-step training with turn-onset
  oversampling (used in the "why action conditioning is hard" investigation).
- `onset_effect.py` — quantifies the action's true effect on the next frame in
  pixels (investigation Step 0).
- `pathway_sanity.py` — synthetic shift task proving the action-conditioning
  pathway can learn (investigation Step 1).
- `onset_ablation.py` — horizon action-use ablation at turn onsets (investigation
  Step 3).
- `metrics.py` — L1 / MSE / PSNR / SSIM and the phase-correlation image-shift
  estimator.
- `analysis.py` — read-only access to the packed analysis sidecar (heading, pose,
  doorway candidates), keyed onto global frame indices for evaluation.
- `eval_harness.py` — the long-lived evaluation harness (see §9).
- `review_images.py` — renders the three Stage 4 review images (see §10).
- `bench.py` — model size / FLOPs / memory / latency measurements (see §11).
- `review.py` — renders `windows.png`, the one Stage 3 review image.
- `gpu_check.py` — records versions and proves PyTorch runs real GPU work.
- `tests/` — the verification suite (Stage 3 loader + Stage 4 metrics/harness).

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
.venv/bin/python -m pytest -q          # loader suite (alignment, rules, round trip,
                                       # totals, determinism/reuse, throughput+smoke,
                                       # environment) + Stage 4 metrics/harness tests
```

## 7. The baseline model

`model.py` defines the Stage 4 baseline and the **interface every future model
must implement** (the evaluation harness only knows this):

```python
predict(context, action) -> next_frame
#   context : [B, K, 3, H, W] float in [-1, 1]   (K context frames)
#   action  : [B, K, 2] float  [actionX, actionHasClicked]
#   next    : [B, 3, H, W] float in [-1, 1]
```

The baseline is a small **residual U-Net**: K=4 context frames + their 4 actions
in, one predicted next frame out. It predicts the *change* from the last context
frame (a residual) rather than the whole frame, and adds it back in float32 so
mixed precision never loses the base frame. The action is injected the simplest
way: each frame's 2-dim action is broadcast as two constant image channels
concatenated to its RGB channels, so the network sees `K * (3 + 2) = 20` input
channels. Two-level encoder-decoder (widths 32/64/128), GroupNorm, nearest-upsample
+ conv in the decoder. ~708k parameters.

`PersistenceModel` (repeat the last context frame, ignore the action) is the
reference every model must beat; the harness evaluates it on the same windows.

**Known result (Stage 4):** the baseline beats persistence ~3× on one-step L1 but
does **not** use the action — the true/zeroed/shuffled/shifted ablation ties, and
the counterfactual direction test fails. The cause is the action's tiny per-step
effect at 20 Hz (~0.03 px, accumulating only over ~10 frames), documented in
`docs/STAGE4-REPORT.md` §7 along with what was tried. Any future model should be
judged with the horizon ablation and counterfactual metrics, not one-step pixel
error alone.

## 8. Training

```bash
# iterate at 64x48 (fast), then run once at 96x72
.venv/bin/python train.py --name baseline_64 --packed data/pilot_v1_64x48 \
    --max-steps 6000 --batch 64 --val-every 250 --seed 0
.venv/bin/python train.py --name baseline_96 --packed data/pilot_v1_96x72 \
    --max-steps 6000 --batch 64 --val-every 250 --seed 0
```

`train.py` is config-driven (every knob is a key; CLI flags override). Everything
lands in `runs/<name>/` (gitignored): `config.json` (full config + window/param
counts), `best.pt` / `last.pt`, `train_log.json` (step/train/val losses — the
loss curve), `result.json` (one-line summary incl. throughput). Key knobs:

- `--limit N` — train on only the first N windows (the **sanity overfit** check).
- `--train-fraction F` — train on a seeded random F fraction of windows
  (the **learning curve**).
- `--mixed-preprocessing bf16` (default) uses bf16 autocast on the GPU.

Training uses `require_safe=True` windows (drops `viewUnsafe`), L1 loss, Adam
lr 1e-3, seed 0.

## 9. The evaluation harness

`eval_harness.py` evaluates **any** model with the `predict(context, action)`
interface (plus a built-in autoregressive rollout driver). It runs the five parts
of Stage 4 §2.4 and writes one JSON:

```bash
.venv/bin/python eval_harness.py --packed data/pilot_v1_64x48 \
    --checkpoint runs/baseline_64/best.pt --split val --out eval_out/baseline_64_val.json
```

- **(a) one-step metrics** — L1 / MSE / PSNR / SSIM, overall **and** on an
  "action-relevant" subset (windows whose target-step heading change is ≥ 0.5°,
  i.e. a turn is happening).
- **(b) action-use ablation, no retraining** — the same windows scored with the
  true actions, zeroed actions, shuffled actions, and actions shifted one frame
  earlier / later. True alignment should score best if the off-by-one rule is
  right and the model uses the action.
- **(c) rollouts** — autoregressive ~40-step rollouts from real context with the
  true actions; error vs horizon for the model and persistence.
- **(d) action-following** — phase-correlation horizontal image shift between
  predicted consecutive frames vs the real frames' shift and the sidecar heading.
- **(e) counterfactual direction** — contexts with reachable doors on both sides,
  rolled under "click left / right / none"; measures whether the view turns the
  expected way.

Evaluate the test split **once**, at the end, with the final model.

## 10. The review images

```bash
.venv/bin/python review_images.py --packed data/pilot_v1_96x72 \
    --checkpoint runs/baseline_96/best.pt --out-dir review/
```

Produces three images (drawn **through the loader + model**, never by reading
PNGs): `predictions.png` (~8 val windows, context + GT + model + persistence, action
values printed), `rollouts.png` (~3 val clips, GT above the model's autoregressive
frames every 4th step for ~40 steps), `counterfactual.png` (~3 contexts, left/right/
no-click at a few time steps). Visual judgment is yours; see the Stage 4 report.

## 11. Speed and resource

```bash
.venv/bin/python bench.py --packed data/pilot_v1_96x72 \
    --checkpoint runs/baseline_96/best.pt --out eval_out/baseline_96_profile.json
```

Reports parameter count, forward FLOPs (batch 1), precision, peak GPU memory for a
training step and for batch-1 inference, batch-1 inference latency/fps, and
autoregressive rollout-loop fps vs the 15 fps target.
