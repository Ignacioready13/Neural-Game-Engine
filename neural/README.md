# /neural — Python environment, dataset packager, and training-data loader

This is the **neural half** of NeuralGameEngine: the Python/PyTorch side that
turns the classical engine's episodes (PNG frames + per-frame labels) into
compact, fast-to-read training arrays and serves training windows to a model. It
is fully independent of `/engine`; it only *reads* the engine's dataset and never
calls back into it.

---

## How to play and compare models (the player)

`play.py` is the real-time thin client: it runs **one registered model** at the
training rate (20 Hz) in a window, applies the engine's click semantics, and draws the
engine's click pulse. It never touches the engine, the sidecar or door information —
at play time the model sees only its own recent generated frames plus the click
signals, after a real K-frame seed context (N1/N2). Each run starts from a real
early-episode clip taken from the cached seed contexts (`models/seeds.npz`), so the
18 GB packed set does not have to be mounted to play.

```bash
cd neural

./play.sh                            # newest selected model (currently flow96c), windowed
./play.sh --model flow96             # pick a model by its internal run name
./play.sh --model flow96d            # a candidate that was NOT selected (see below)
./play.sh --list-models              # what is registered, which are selected
./play.sh --model flow96c --fullscreen
./play.sh --help                     # every flag with its meaning and default
```

**Models are named after their internal training run** — `flow96`, `flow96c`,
`flow96e`, … — which is the name the player shows. The Stage 5-era letters survive only
as aliases (`5a` = `flow96`, `5b` = `flow96c`), so older commands still work.

`play.sh` runs the player under `memcap.sh` (a hard memory cap, so a runaway job can
only kill itself, never the editor). Add `--no-memcap` as the first argument to skip
that wrapper.

### Keys

| key | what it does |
|---|---|
| left click | steer: the click's X sets the held signal (clamped to the image, not the window). Every click draws the engine's pulse; a click within 150 ms (3 frames) of the last *accepted* click is ignored for the signal but still shows its pulse. |
| `F11` / `alt+enter` | fullscreen on/off |
| `Esc` | leave fullscreen (then quit if already windowed) |
| `M` / `shift+M` | next / previous **selected** model — restarts from the **same** seed context and noise seed (and the same recorded clicks if a replay is running), so the only thing that changed is the model. (If you started on a non-selected model, `M` joins the selected cycle at its first entry.) |
| `R` | restart with a new random seed context and noise seed |
| `shift+R` | restart with the *identical* seed context and noise seed (an exact rerun) |
| `space` | pause / resume |
| `.` or `→` | single step (pauses first) |
| `Tab` | compact HUD on/off (`H`/`F1` shows the full help overlay) |
| `T` | nearest (integer) / smooth scaling |
| `S` / `F2` | screenshot to `review/stage6/` |
| `C` | start/stop recording the session's clicks for replay |
| `V` | start/stop recording the display to a GIF in `review/stage6/` |
| `P` | print a one-line status (model, fps, held X, seed, dropped steps) |
| `Q` | quit |

### Registered models

`models/registry.json` is the registry; each name freezes a copy of its generator
checkpoint under `models/<name>/generator.pt`, so a later training run that reuses
`runs/<name>/` cannot change what the name means.

| name (internal) | alias | generator | selected | notes |
|---|---|---|---|---|
| `flow96` | `5a` | `runs/flow96/best.pt` | yes | the Stage 5 model (`v0` in the Stage 5b scorecard) |
| `flow96c` | `5b` | `runs/flow96c/best.pt` | yes | **default**: the Stage 5b overnight winner |
| `flow96d` | — | `runs/flow96d/best.pt` | no | rejected by the frozen scorecard; excluded from the `M` cycle, still playable with `--model flow96d` |

**"Selected"** means the model passed the project's selection step: the frozen
scorecard's `keep` verdict in `eval_out/decision_*.json` (and `winner` in
`docs/ENERGY-LEDGER.json`). Only those are in the `M` cycle and eligible as the default.
Registered-but-unselected models, and even a `runs/<name>/best.pt` that was never
registered, are still playable by naming them — so you can inspect any candidate by hand
without promoting it:

```bash
./play.sh --model flow96d      # registered, not selected
./play.sh --model flow96e      # not registered: loads runs/flow96e/best.pt directly
```

Add the next revision (reads the training run, copies the checkpoint, prints the play
command):

```bash
.venv/bin/python register_model.py --run flow96e \
    --core runs/steer96/best.pt --description "flow96e: longer self-rollout" \
    --parent flow96c
./play.sh --model flow96e
```

### Recording and comparing two models

Press `C` while playing (or pass `--record FILE`) to record the model, seed context,
noise seed and every click with its frame index, then replay those exact clicks on any
model:

```bash
# replay the recording in a window on another model (live input is ignored; M switches
# the model mid-replay to see both on the same clicks)
.venv/bin/python play.py --model flow96 --replay models/recordings/flow96c_session_<stamp>.json

# or headless: write a GIF for the review folder
.venv/bin/python play.py --model flow96 --headless \
    --replay models/recordings/flow96c_session_<stamp>.json --out review/stage6/replay_flow96.gif
```

Replaying the same recording on the same model reproduces the frames bit for bit
(`tests/test_stage6.py` checks this); the same recording on the other model is a fair
comparison of the two models' responses to the *same* clicks.

### Scripted / headless runs and frame times

```bash
./memcap.sh --mem 12G -- .venv/bin/python play.py --model flow96c --headless \
    --frames 400 --pace --clicks "15:0.8,60:-0.6,140:0.9" \
    --out review/stage6/play_flow96c.gif --report-frames review/stage6/frames_flow96c.json
```

`--pace` holds the real 20 Hz rate; `--report-frames` writes mean/p95/worst frame time
and whether any step was dropped. Run a checkpoint that is not registered with
`--checkpoint runs/<run>/best.pt --packed data/bulk_v1_96x72`.

Where things live: `models/registry.json` (registry), `models/<name>/generator.pt`
(frozen checkpoints), `models/seeds.npz` (cached seed contexts, built by
`build_seeds.py`), `models/recordings/` (recorded sessions), `review/stage6/`
(screenshots, display recordings, headless GIFs). A live session also appends one row
to `docs/ENERGY-LEDGER.json/md` on exit and prints GPU W, J per frame and Wh per hour
of play.

A review set already exists in `review/stage6/`: `demo_flow96c.gif` is a recorded
240-frame session, and `demo_flow96.gif` / `demo_flow96d.gif` are the *same recording*
(`models/recordings/demo_flow96c.json`) replayed on the other two models — the
flow96-versus-flow96c comparison on identical clicks.

---

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
- `train_future.py` — H-step teacher-forced lookahead training + horizon ablation +
  counterfactual (the objective that made the model action-responsive, §7.7).
- `train_hold.py` — deployment-realistic hold-last conditioning (future = `a_{t-1}`
  repeated) + click-onset / quiet-onset ablation (§7.8).
- `probe_click.py` — the redesigned click-direction counterfactual: distinct
  mid-episode contexts, close doors on both sides, "click vs hold" shifts (§7.7).
- `head_predict.py` — P4: direct heading predictor (pixels + actions → signed heading
  change) showing the turn direction is extractable at the oracle ceiling (§7.10).
- `train_hold_aux.py` — P5: hold-last frame model with a training-only auxiliary heading
  head, showing the direction stays in the bottleneck and does not reach the frames
  (§7.11).
- `train_intent.py` — P6: hold-last frame model with an explicit oracle-intent input and
  (optionally) a structured warp output, showing a ~7 px view shift needs both an
  explicit direction signal and a differentiable warp, not a free pixel residual (§7.12).
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
`docs/STAGE4-REPORT.md` §7 along with what was tried. **What works, and its limit**:
an H≈10-step teacher-forced lookahead objective (`train_future.py`) makes the same
model decisively action-responsive (shuffled lookahead actions 38% worse) — but the
direction comes from the *true future* action sequence's click "change events"; the
deployment-realistic hold-last form (`train_hold.py`) keeps only "turn presence",
not direction (§7.8). The direction *is* extractable from pixels + actions — a direct
heading readout (`head_predict.py`) hits the oracle ceiling (0.85 vs 0.87) — but it
does **not** flow through a pixel-loss-trained frame decoder: an auxiliary heading head
on the hold-last model reads direction at 0.89 yet the frames stay direction-blind
(§7.11). P6 closes the loop: the generator *can* render a ~7 px view shift, but only as
a structured warp of the last frame driven by an explicit direction signal (both-correct
0.87, ratio 0.65) — a free pixel residual stays direction-blind even when told the
direction (§7.12). So the diffusion stage should condition the generator on an explicit
direction signal *and* render motion through a differentiable warp/flow — see §9. Any
future model should be judged with the horizon ablation, the redesigned click
counterfactual (`probe_click.py`), and the heading-readout accuracy (`head_predict.py`),
not one-step pixel error alone.

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

---

## 12. Stage 5 — the real generator (steering core + stochastic generator + gates + player)

Stage 4 ended with a precise diagnosis: the direction of a click *is* readable from
the frames and the click history, but a pixel-residual decoder cannot render the
resulting few-pixel global view shift, while a differentiable **warp** can. Stage 5
turns that into a generator.

### What the pieces are

- `shift_table.py` — precomputes the per-step horizontal image shift for a packed set
  (phase correlation, vectorized per episode) and a **denoised** target table derived
  from the sidecar `playerHeading` (`dshift = scale · dheading`, `scale ≈ −35 px/rad`
  at 64×48). Both are training-only. Caches live in `data/_cache/` so the packed
  datasets stay untouched.
- `steer.py` — the steering core: click features (`actionX`, `actionHasClicked`,
  `deltaActionX`, frames-since-click), the differentiable `warp_translate`, the
  `DirectionReadout` (per-step shift increments for the next H steps, with Gaussian
  *click-column* channels so the encoder can compare the click position against the
  door layout), and `SteeringModel` = readout → warp → small gated low-resolution
  residual.
- `train_steer.py` — trains the core with three ingredients that matter: a denoised
  heading-derived target, **DAgger roll-ins** (roll the model forward k steps and
  supervise on its own frames — without it the closed loop is bistable), and
  **counterfactual clicks** (swap in the other door's click with a target from the
  fitted engine-rule geometry — without it the readout reads the direction off the
  frames and ignores the click).
- `flow.py` — the stochastic generator: conditional rectified flow on the **residual
  relative to the motion-warped last frame** (`next = warped + r̂`), 4 sampling steps.
  Keeping the warp outside the learned residual makes steering structural.
- `train_flow.py` — trains the generator on top of a frozen steering checkpoint, with
  turn-onset oversampling, mirror augmentation and noise-augmented context frames.
- `gates.py` — the one-command gate runner (G1–G4) and the Stage 5 metrics.
- `play.py` — the minimal interactive player (pygame) plus a headless scripted mode.
- `review_stage5.py` — the review images for §6 (rollouts, counterfactual, reveals,
  long runs).

### Run it

```bash
# 1. shift tables (once per packed set; a couple of minutes, CPU only)
./memcap.sh --mem 12G -- .venv/bin/python shift_table.py --packed data/bulk_v1_64x48

# 2. steering core (Milestone 1)
./memcap.sh --mem 14G -- .venv/bin/python train_steer.py --name steer64 \
    --packed data/bulk_v1_64x48 --max-steps 14000 --batch 64

# 3. generator (Milestone 2), on top of the core
./memcap.sh --mem 14G -- .venv/bin/python train_flow.py --name flow64 \
    --packed data/bulk_v1_64x48 --core runs/steer64/best.pt --max-steps 8000

# 4. gates (val; the test split is touched once, in Milestone 4)
./memcap.sh --mem 14G -- .venv/bin/python gates.py --gates g1 \
    --packed data/bulk_v1_64x48 --checkpoint runs/steer64/best.pt \
    --out eval_out/g1_steer64.json
./memcap.sh --mem 14G -- .venv/bin/python gates.py --gates g2 \
    --packed data/bulk_v1_64x48 --checkpoint runs/flow64/best.pt \
    --baseline runs/baseline_64/best.pt --seeds 0,1,2 --out eval_out/g2_flow64.json
./memcap.sh --mem 14G -- .venv/bin/python gates.py --gates g3 \
    --packed data/bulk_v1_96x72 --checkpoint runs/flow96/best.pt --steps 300
./memcap.sh --mem 12G -- .venv/bin/python gates.py --gates g4 \
    --packed data/bulk_v1_96x72 --checkpoint runs/flow96/best.pt

# 5. review images and the player
./memcap.sh --mem 12G -- .venv/bin/python review_stage5.py \
    --packed data/bulk_v1_96x72 --checkpoint runs/flow96/best.pt \
    --out-dir review/stage5 --which all
./memcap.sh --mem 12G -- .venv/bin/python play.py \
    --checkpoint runs/flow96/best.pt --packed data/bulk_v1_96x72
```

### The model interface

Every Stage 5 model keeps the Stage 4 call and adds two optional arguments:

```python
predict(context, action=None, features=None, seed=None, return_shift=False)
#   context  [B,K,3,H,W] float in [-1,1]
#   action   [B,K,2]     [actionX, actionHasClicked]         (click signals)
#   features [B,K,4]     click features (derived from `action` if omitted)
#   seed     int|None    stochastic sampling seed (generators)
#   returns  next frame [B,3,H,W] in [-1,1], or (frame, info) if return_shift
```

`eval_harness.model_predict` dispatches on the signature, so Stage 4 checkpoints stay
evaluable, and `eval_harness.rollout_v2` is the feature-aware, seed-aware closed-loop
driver used by the gates and the player.

### No-oracle (N1/N2)

At inference the models accept only frames, click-derived signals and a seed.
`tests/test_stage5.py::test_rollout_driver_never_reads_the_sidecar` asserts the
closed-loop driver works with the analysis sidecar unavailable, and
`test_inference_models_accept_only_frames_and_clicks` pins the signatures. The sidecar
(heading, doorway candidates) is used **only** while building training targets.

---

## 13. Stage 5b — the frozen scorecard, the judges, and energy accounting

Stage 5's gates passed but did not capture the user's playtest complaint (in long
closed-loop play no doorway ever appears and the walls dissolve into smears), so Stage 5b
freezes a scorecard **before** changing the model, scores every candidate against it, and
accounts for the energy of every run.

### What the pieces are

- `powerlog.py` — the energy logger. Samples GPU power (NVML) and CPU package power
  (Intel RAPL) at ~1 Hz in a daemon thread, integrates with the trapezoid rule to Wh, and
  appends one record per run to `docs/ENERGY-LEDGER.json` / `.md` with the command line,
  purpose, mean/peak W, temperatures, throttle events and `parent_run` so a model's
  lineage is traceable. Wraps any command:
  `./memcap.sh --mem 15G -- .venv/bin/python powerlog.py --run-id X --purpose train -- <cmd>`,
  and `docs/ENERGY-LEDGER.md` carries the idle baseline, the measured totals, the
  lineage and (separately, labelled) the estimate for pre-logger runs.
- `judges.py` — evaluation-only judges; **nothing in the inference path imports it**
  (asserted by a test). The opening detector is trained on real frames only, with labels
  derived from the sidecar's doorway candidates ("a *reachable* candidate with
  `screenDistance ≤ 0.35` and `|screenX| ≤ 0.35`"), per-frame instance normalisation so
  darkness is not a shortcut, oversampled no-door controls, and a threshold calibrated on
  real frames. Also the real-vs-generated classifier, the radial power spectrum,
  multi-scale gradients and the long-run health statistics.
- `scorecard.py` — one frozen evaluation set (`eval_sets/scorecard_v1.json`, never
  rewritten in place) and one command per candidate: 40 val clips × {no click, the clip's
  real clicks, a scripted left/right pattern} × 3 seeds × 300 closed-loop steps, plus the
  frozen Gate-G1 steering contexts, the reveal one-step test, and the confirmation/test
  runs. Rollouts execute batched, which is 4x faster and bit-identical to the
  single-rollout path at batch 1.
- `score_table.py` — the pre-declared keep/revert rule, applied against the incumbent's
  **measured seed-to-seed noise** (the record keeps both the main reading and the stricter
  "every horizon individually beyond noise" one).
- `diagnose_stage5b.py`, `review_stage5b.py`, `night_queue.sh` — the diagnosis, the
  review images (large, tightly cropped) and the ordered night plan.

### Where the two stages disagree, and why it matters

Stage 5b's diagnosis measured that real frame-to-frame motion is a **pure 0.89 px/step
translation** (adding a zoom term reduces the one-step residual by 0.08 %), so the
translation-only warp was right and the generator's job is to invent the thin revealed
strip. The frozen detector then showed the incumbent displaying a walkable opening in
12.5 % of its 300-step frames against 52.5 % of real. A 3.19 M-parameter denoiser with
global attention and a longer self-rollout curriculum fixed most of that (1.015x real on a
disjoint confirmation set); an edge-weighted loss that improved the training loss and
every one-step metric made the 300-step number **0.000**, and was rejected. Freeze the
long-horizon scorecard first.
