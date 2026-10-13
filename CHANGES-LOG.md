# Changes Log

A running log of changes requested and what happened. **Newest first.** Each
entry records *what was asked → root cause → fix → verification*. For permanent
architecture and subsystem contracts, see [`README.md`](README.md) (definitive)
and `engine/docs/CONTRACTS.md`. This file replaces the old `handoff.md`.

---

## 2026-10-12 (evening) — Stage 6: the player, a model registry, and head-to-head replay

**Asked.** Stage 6 (`stage6-prompt-player.txt`): turn the Stage 5 script into the
project's real-time thin client — pick a model by name, a window you can actually see
(resizable, fullscreen), the engine's click pulse, the engine's click semantics at the
training rate, seed/restart controls, a HUD and help, session recording and replay so two
models can be compared on the same clicks, a launcher and README instructions, and keep
the live-session power accounting. The user waived the side-by-side view.

**Built.** `neural/model_registry.py` + `models/registry.json` + `register_model.py`
(**models are named after their internal training runs** — `flow96`, `flow96c`, `flow96d`
— per the user's preference, with the Stage 5 letters `5a`/`5b` kept only as aliases;
each name freezes a copy of its generator so a later run cannot move the name, and each
entry carries a `selected` flag mirroring the frozen scorecard's `keep` verdicts in
`eval_out/decision_*.json`); `neural/build_seeds.py` + `models/seeds.npz` (160 labelled
seed clips, so the 18 GB packed set is not needed to play); `neural/play.sh` (the one
command to start it); `neural/tests/test_stage6.py` (41 tests).
`neural/play.py` rewritten as the real client: 20 Hz fixed timestep with
drop accounting, resizable/fullscreen aspect-preserving letterbox, image-relative click
mapping with clamping, the engine's fading click-pulse disc on every click, registry
selection by flag and hotkey — `M`/`shift+M` cycle only the **selected** models, while
any model (including rejected candidates and unregistered `runs/<name>/best.pt`) is
playable by naming it — restarting from the same seed context and noise seed, seed/restart
keys, HUD + help overlay (showing the internal model name, and `[not selected]` when
relevant), screenshots, session record/replay (byte-identical on the same model, any model
on the same clicks), display GIF recording, and power-ledger rows with J/frame and
Wh/hour.
`neural/powerlog.py`: added a non-blocking
`PowerLogger.latest()` for the live HUD readout. `neural/README.md`: a "How to play and
compare models" section at the top. `.gitignore`: the derived `models/**/*.pt`,
`models/seeds.npz` and `models/recordings/` are not committed (the registry is).

**Bugs found in the Stage 5 player and fixed.** The action tensor handed to the model was
the current action repeated K times instead of the real click history; the click history
grew unboundedly, so `steer.click_features` walked it in a Python loop every frame and a
long session slowed down without bound; the seed's real click history was not used to
restore the debounce state (so the first click was accepted even 1 frame after a real
accepted click); the "pulse" was a full-height white bar rather than the engine's disc;
clicks were mapped against the window instead of the image; `--headless --frames N`
emitted the seed frame as frame 0. No engine, dataset, packer, loader or checkpoint was
touched; git untouched.

**Verification.** 41 new tests pass (97 passed overall on the final idle run; an earlier
run on the busy machine had the pre-existing load-dependent
`test_loader.py::test_throughput` failure — it imports only the untouched `dataset.py` and
is recorded as load-sensitive in the Stage 4B/5b reports). Both registered models generate 500 paced frames at 20 Hz with **0
dropped steps**: mean 4.29 / 6.18 ms, p95 4.65 / 7.51 ms, worst 5.25 / 8.37 ms against
the 50 ms budget. Replaying a recording on the model that recorded it reproduces the
frames bit for bit; the window was created, resized and toggled to fullscreen on the real
3840x2400 desktop (windowed 1536x1152 default; fullscreen letterboxed 3168x2376) and
under SDL's dummy driver. Feel, how the pulse looks in motion, and fullscreen behaviour
on the user's compositor are for the user to judge.

---

## 2026-10-12 (overnight) — Stage 5b: structure, a frozen scorecard, and energy accounting

**Asked.** Continue Stage 5 overnight (brief `stage5b-prompt-overnight.txt`): make the
generator produce real structure (the user's playtest found no doorway ever appears and
the hallway dissolves), tune it against a **scorecard frozen before any training**, and
account for the energy of every run.

**Built.** `neural/powerlog.py` (1 Hz NVML + Intel RAPL sampling, trapezoid Wh
integration, one ledger record per run with `parent_run` lineage, `--baseline`,
`--winner`, `--mark-aborted`, `--reconcile`); `neural/judges.py` (frozen opening
detector, real-vs-generated classifier, radial power spectrum, multi-scale gradients,
long-run health, opening-event counting); `neural/scorecard.py` (frozen sets, the
detector, judge validation, per-candidate rows, the real-reveal one-step test);
`neural/score_table.py` (candidate table + the pre-declared noise-aware keep/revert
rule); `neural/diagnose_stage5b.py`; `neural/review_stage5b.py`;
`neural/night_queue.sh`; `neural/zoom_table.py`; `neural/tests/test_stage5b.py`;
`neural/eval_sets/scorecard_v1.json`; `neural/docs/{ENERGY-LEDGER.json,ENERGY-LEDGER.md,STAGE5B-REPORT.md}`.
Additive edits to `flow.py` (low-resolution self-attention, `net_base`, edge-weighted
reveal loss), `train_flow.py` (multi-seed checkpoint selection, `log_every`, a guard
against confusing `--steps` with `--max-steps`), `gates.py` (frozen context/window
lists, power in `run_g4`) and `play.py` (per-play-session power accounting, uint8-aware
frame decoding). No engine, dataset, packer or loader change; git untouched.

**Result.** The frozen detector showed the incumbent showed a walkable opening in
**12.5 %** of its 300-step frames against 52.5 % of real. Winning candidate
`runs/flow96c/best.pt` (denoiser 0.77 M → 3.19 M + global attention + a 4-step
self-rollout curriculum, 19.6 min of training) reaches **1.015x real on a disjoint
40-clip confirmation set — 8.6x the incumbent's share on the same clips** — and 0.758x
on the test split, at 37.1 fps / 417 MB VRAM. A second candidate with an edge-weighted
reveal loss improved the training loss and *every* one-step metric yet collapsed the
300-step opening share to 0.000, and was rejected by the frozen rule: the clearest
demonstration of why the scorecard was frozen first. Energy: 374.28 Wh GPU + 187.71 Wh
CPU measured over 5.03 h, 32.13 Wh for the winner's training, plus ~359 Wh as a labelled
estimate for the pre-logger Stage 5 runs.

---

## 2026-10-11 — Stage 5: the real generator (steering core, flow generator, gates, player)

**Asked.** Build the actual neural game engine: from recent frames plus the click
signal alone, generate the next frame over and over, steerably and stochastically,
in real time (brief `stage5-prompt-generator.txt`).

**Built.** `neural/steer.py` (click features, differentiable warp, direction
readout, steering model), `neural/train_steer.py`, `neural/flow.py` (4-step
rectified-flow generator on the warped-frame residual), `neural/train_flow.py`,
`neural/shift_table.py` (shift + heading-derived targets), `neural/gates.py`
(one-command G1-G4 runner), `neural/play.py` (interactive + scripted headless
player), `neural/review_stage5.py`, `neural/tests/test_stage5.py`,
`neural/docs/STAGE5-{PROGRESS,REPORT}.md`. Additive extensions to
`eval_harness.py` (seed/feature-aware `rollout_v2`, `model_predict`) and
`metrics.py` (sharpness, colour, diversity, collapse, frame-change metrics);
`pygame-ce` added to `requirements.txt`/`.lock`. No engine, pilot, bulk_v1,
packer or loader change; git untouched.

**Result.** Steering core (64x48): readout direction sign accuracy 0.923 vs the
truth and 0.898 vs the engine-rule ceiling of 0.954; counterfactual both-directions
0.880 (near-door 0.878); 10-step shift ratio 0.780; 40-step sustained median ratio
0.680 with no runaway; 60-step generated sharpness within 25% of real and
per-channel colour within 0.003. Full numbers, the failure modes found (a
scene-consistent click can be ignored; the denoiser cancels the warp given
un-warped context; an L1 velocity loss collapses the sampler to the median; flow
noise must match the residual scale; long-rollout phase-correlation aliasing), the
speed/hardware measurements and the file list are in
`neural/docs/STAGE5-REPORT.md`.

**Verified.** `pytest` Stage 3/4/5 suites pass (27 Stage 5 tests incl. the N1/N2
no-oracle checks); gates run with one documented command each.

---

## 2026-10-11 — Stage 4B: bulk dataset, `mixture_v2`, packing, coverage

**Asked.** Build the bulk training set the real model will use — tuned bot mixture,
large val/test splits, packing at two resolutions, and a coverage report (brief
`stage4b-prompt-bulk-dataset.txt`).

**Built.** `engine/data/bulk_v1/` — 1,500 planned episodes (1,100/200/200) × 600
frames under a **new** `mixture_v2` profile; plan `engine/data/bulk_v1.plan.json`;
`report.json`/`report.md`; `aborted.jsonl`. Packed at 96×72 (18.17 GB) and 64×48
(8.08 GB) with the Stage-3 packer under `neural/data/` (gitignored).
`neural/docs/STAGE4B-REPORT.md` + sample image `neural/review/bulk_v1_sample.png`.
`mixture`/`uniformOnly`/`doorAwareOnly` are **unchanged**, so the pilot is
reproducible; the engine simulation/rendering/schema were not touched; git was not
touched.

**Result.** 1,445 complete + 55 aborted (3.67 %), **876,261 frames**. Abort share
3.67 %, turn-window direction balance 50.1/49.9 (target 47–53 %), `viewUnsafe`
0.36 %. Qualifying click-counterfactual yield **126.5/1,000** (near-door 100.1) on
train — **1.44× / 1.50×** the pilot's 87.7/66.8. Val/test each hold **>11,500
near-door contexts from ~190 episodes** (target ≥400 from ≥100). All persona
frame shares ≥ 4.7 %; straightest-choice 48.6 % and label ambiguity 11.8 %
(unchanged from the pilot). Packed totals equal the manifests at both
resolutions; the loader tests pass on the bulk set.

**Tuning.** One round. Per-persona yield measurement showed `spammer`
(173.9/1,000) and `dither` (138.3/1,000) dominate, so `mixture_v2` raises them
(5→26 and 10→30) and lowers `uniform`/`idle`/`late`/`prober`, keeping every persona
≥ ~5 % of frames. No bot bug: the left/right imbalance is not an engine
world-generation problem (the world is left/right balanced, ~48.5/51.5), so no bot
or engine fix was needed.

**Fixed (memory safety).** Post-pack verification OOM-killed the editor session:
`torch`'s DataLoader uses the `forkserver` start method here, so the dataset is
pickled to each worker — and `np.memmap` pickles by copying its whole file, so
each worker tried to materialise the 18 GB `images.bin`. Added
`PackedDataset.__reduce__` (re-open the memmap from the path in the worker);
verified 642 MB at `num_workers=2` (was OOM), bulk loader throughput 3,808 w/s (2
workers) / 7,672 w/s (4 workers) at 2.8 GB. Added `neural/memcap.sh` to run heavy
jobs in a memory-capped systemd user scope so a runaway can never take down the
editor. **Swap was not touched** (and is not needed — the runaway was one bug, now
fixed).

**Also changed (small, additive).** `workerPool.ts`/`plan.ts`: the abort-share
early stop now needs ≥ `clipStopMinEpisodes` (default 100) finished episodes (was
hard-coded 20), at the user's request; thresholds and the failsafe are unchanged.
`cli.ts` help lists profiles dynamically. `analysis.py` exposes `clickX`,
`clickAccepted`, `persona`; `probe_click.py` `_select_contexts` gained an optional
`split`; `conftest.py`/`test_loader.py` gained env knobs to point the tests at the
bulk set. New read-only helpers: `count_contexts.py`, `persona_stats.py`,
`sample_grid.py`, `throughput_bulk.py`.

**Open (see report §8).** Whether to keep the `dataset.py` loader change; whether
the new early-stop default should be permanent; two world seeds (`10000017`,
`10000022`) abort under both profiles (engine L2 edge cases, no fix made).

---

## 2026-10-11 — Stage 4: baseline action-conditioned model and evaluation harness

**Asked.** Train the first neural model on the pilot data and — more importantly —
build the reusable evaluation harness every later model (including the diffusion
model) will be judged with (brief `stage4-prompt-baseline-model-and-evaluation.txt`).

**Built (all under `/neural`).** `model.py` (small residual U-Net, `predict(context,
action) -> next_frame` interface, channel-concat and FiLM action injection);
`train.py` (config-driven, seeded, checkpoints + loss curves) and
`train_multistep.py` (autoregressive multi-step + turn-onset oversampling);
`train_future.py` (H-step teacher-forced lookahead + horizon ablation/counterfactual);
`train_hold.py` (deployment-realistic hold-last conditioning + click/quiet-onset
ablation); `probe_click.py` (redesigned click-direction counterfactual);
`head_predict.py` (P4 direct heading readout), `train_hold_aux.py` (P5 auxiliary
heading head) and `train_intent.py` (P6 oracle-intent + structured warp output);
`metrics.py` (L1/MSE/PSNR/SSIM + phase-correlation shift) and `eval_harness.py`
(the long-lived harness: one-step metrics + action-relevant subset, true/zeroed/
shuffled/shifted action ablation, rollouts vs persistence, action-following,
counterfactual direction); `review_images.py` (the three review images);
`bench.py` (params/FLOPs/precision/memory/latency/fps). Report in
`neural/docs/STAGE4-REPORT.md`. `.gitignore` gained `/neural/runs/` and
`/neural/review/`. Git, the engine and the pilot data were **not** touched; no
Stage 3 code changed.

**Key findings.**
- The baseline beats persistence ~3× on one-step L1 (0.0064 vs 0.0197 at 64×48,
  0.0066 vs 0.0204 at 96×72), PSNR 35 vs 25.5, and is far faster than 15 fps
  (batch-1 ~1699 fps, rollout loop ~1143 fps; 2.13 GFLOPs/frame, 708k params).
- **The model ignores the action.** The true/zeroed/shuffled/shifted ablation ties,
  action-following is the same with and without the action, and the counterfactual
  direction test is below chance. A four-step investigation showed this is a signal
  problem, not a bug: at 20 Hz a click moves the next frame by ~0.03 px (measured by
  phase correlation), accumulating only over ~10 frames, so a one-step pixel loss
  gives no gradient to read the action. The conditioning *pathway* works (a synthetic
  shift task learns to 0.04 L1); multi-step objectives, onset oversampling and
  full backprop-through-time all failed to make the model action-responsive.
- **The fix that works — and its limit: an H-step lookahead objective.**
  `train_future.py` predicts the frame H=10 steps ahead in one forward pass from K
  context frames + the K+H action sequence, with turn onsets oversampled. The *same*
  FiLM U-Net then becomes decisively action-responsive: on onset val windows, true
  lookahead actions score 0.101 vs zeroed 0.120 vs shuffled 0.139 (shuffled 38%
  worse; +44% with mirror augmentation), and the redesigned click counterfactual
  (`probe_click.py`) turns the correct way for a *change event* (left +0.72 px /
  right −0.78 px; 0.30 both-correct) but not for a held constant position. The
  catch: that direction comes from the **true-future** action sequence's click change
  events. The deployment-realistic **hold-last** form (`train_hold.py`, future =
  `a_{K-1}` repeated) collapses the ablation to *turn presence* only — shuffled 0.202
  vs zeroed 0.212 (click onset) / 0.112 vs 0.117 (quiet onset), click counterfactual
  both-correct 0.00, turn-sign agreement below chance (0.26). So directional
  action-following at play time needs an explicit signal (auxiliary heading head or an
  explicit `ΔactionX` input) — report §7.7–§7.9, §9. The one-step baseline is unchanged.
- **Where the direction lives (P4/P5).** A direct heading readout from pixels + actions
  (`head_predict.py`) reaches the oracle ceiling at 64×48 (0.85 vs 0.87, chance 0.50; the
  explicit `ΔactionX` input helps, resolution does not) — so the direction *is*
  extractable and no engine/detector change is needed. But an auxiliary heading head on
  the hold-last frame model (`train_hold_aux.py`) reads direction at 0.89 yet leaves the
  frames direction-blind (counterfactual both-correct 0.00, ablation unchanged): the
  frame decoder, trained only by H-step pixel L1, cannot map the bottleneck's direction
  to pixel motion. So the diffusion stage must condition the generator on an explicit
  direction signal directly — report §7.10–§7.11, §9.
- **P6 — the failure is in *expressing* the shift; a structured warp + explicit
  direction signal fixes it.** Feeding the hold-last model the ground-truth intent (the
  door nearest to the click's screenX, as a FiLM scalar + a Gaussian column map) is used
  strongly — zeroing it raises onset H-step L1 44% — but a free pixel residual still
  renders no shift (counterfactual both-correct 0.17). Reparameterizing the output as a
  differentiable warp of the last frame by a predicted scalar shift (warp-only, so the
  residual cannot shadow it) reaches **both-correct 0.87, predicted-vs-real shift ratio
  0.65** under hold-last, and the turn follows the intent alone (0.83). It *needs* the
  signal: `actionX` alone gives 0.43 with a left/right asymmetry. So the diffusion stage
  should condition on the P4 readout's direction *and* render motion through a structured
  warp/flow, not a free residual — report §7.12, §9. (The sidecar's literal
  `targetDoorwayId` is a laggy intent — 0.55 turn agreement at click onset — vs 0.87 for
  the door nearest to the click.)
- The counterfactual shows a rightward turn bias traced to the pilot's heading
  imbalance (42.5% positive / 57.5% negative heading changes).
- The learning curve is flat: 10% of the data already matches 100% (val L1 0.0122 vs
  0.0121) — the one-step task is not data-limited.

---

## 2026-10-11 — Stage 3: Python environment, dataset packager, training-data loader

**Asked.** Build the first piece of the /neural half (brief
`stage3-prompt-python-packager-loader.txt`): a reproducible PyTorch environment
running real GPU work, a packager that turns the engine's episodes into compact
memory-mapped training arrays at a chosen resolution, and a loader that serves
training windows and rollout clips, plus one review image. No model training.

**Built (all under `/neural`).** `packer.py` (parallel PNG→box-downsampled
uint8-RGB arrays + structured labels + episode index + analysis sidecar;
deterministic and incremental/idempotent); `dataset.py` (`PackedDataset` with
`get_clip`/`window`, `WindowDataset` for training, ordered `iter_windows`,
action-vector representation `[actionX, actionHasClicked]`); `review.py`
(`windows.png` drawn through the loader); `gpu_check.py`; `requirements.txt` +
`requirements.lock`; a 9-test pytest suite. `.gitignore` gained entries for the
venv, packed data, and caches. Git, the engine, and the pilot data were **not**
touched.

**Environment.** Default Python 3.14.4 is already supported by PyTorch 2.14.1, so
no user-space Python manager was needed. RTX 5070 (Blackwell `sm_120`) needs
CUDA ≥ 12.8; driver is CUDA 13.0, so `torch==2.14.1+cu130` is the build.
Verified with a real matmul + conv + backward (not just `is_available()`).

**Key findings.**
- Pilot packs to **149 episodes / 88,047 frames**: 145 complete + 4
  aborted-but-truncated (the 5th aborted episode was deleted by the engine's
  failsafe). `viewUnsafe` **357 frames (0.41%)**, `actionHasClicked=false`
  **9.8%** — both match the Stage 2 report.
- Loader throughput at 96×72, K=4: **~10,000 windows/s** single-process with the
  split in RAM (`in_memory=True`), ~3,500/s memory-mapped. Windows are tiny, so
  DataLoader worker IPC overhead dominates and `num_workers=0` is fastest; the
  loader is not the bottleneck.

**Verification.** `python -m pytest -q` — **9/9 pass**: action alignment
(`actionX[j] == controlSignalX[j+1]`), validity rules + independent window count,
pixel round trip vs an independent box downsample, totals (88,047 / 357),
determinism (byte-identical double pack) + reuse (re-run skips all 149), and
throughput + a 40-step GPU smoke train (loss 0.282 → 0.023). Full report at
`neural/docs/STAGE3-REPORT.md`; consumer docs at `neural/README.md`.

---

## 2026-10-11 — Stage 2 follow-up: `viewUnsafe` label (label-only) + pilot regeneration

**Asked.** From Check-in 3: a read-only diagnosis had found frames where the
camera (a floor point with no collision radius) sits close enough to a rendered
wall that either the camera near plane clips it (leaving a hole) or the camera
crosses a single-sided wall plane and sees through it — and the clipping failsafe
cannot catch either, because it only watches the simulation. Implement the
smallest fix: a per-frame **label-only** `viewUnsafe` boolean, with **no
simulation or rendering change** (near plane stays 0.02).

**Built.** `src/sim/viewSafety.ts` — `ViewSafetyProbe` measures the closest
rendered triangle of the active entity + maintained neighbours (Ericson §5.1.5
point-triangle distance over cached world-space triangles) and tests the
back-facing/in-front see-through case; wired into `Engine.getState()` (cache
cleared in `syncWorld()`). `Renderer.cameraNear` exposes the live near plane.
The field flows through `BackroomsState`, `FrameRecord`, the manifest config
subset (`viewUnsafeMargin`) and the driver. `report.ts` gained a `viewUnsafe`
section (§11 of `report.md`). Threshold = camera near (0.02) +
`VIEW_UNSAFE_MARGIN` (0.10) ⇒ 0.12.

**Guarantee.** Label only — it never aborts, trims or otherwise changes an
episode (the clipping failsafe is untouched). Loaders must drop any training
window that touches a flagged frame; documented in DATASET-SCHEMA.md §5b.

**Measured (regenerated pilot).** **357 / 88,047 frames (0.41%)** — under the
~1% budget, so the margin was not tightened — across **90 / 150 episodes
(60.0%)**; 72 of those 90 episodes have only 1–4 flagged frames. Margin 0.10 was
chosen so `seed 0`'s whole doorway graze flags (f129–f135, spanning the named
f132/f133/f134; f128/f136 clear).

**Verification.** `tsc --noEmit` clean; 56 vitest tests pass; `seed 0` smoke
episode flags exactly f129–f135. Pilot regenerated in place (behaviour-neutral:
identical 145 complete / 5 aborted, 88,047 frames, 6.70 GB, 130 s).

---

## 2026-10-10 — Stage 2: action labels, click-bot personas, resumable runs, pilot

**Asked.** Implement Stage 2 of the training-data generator (brief
`stage2-prompt-bots-labels-pilot.txt`): unambiguous action-aligned labels
(4.1–4.4), a click-bot persona framework whose choices spread uniformly at
*choice moments* rather than always picking the straightest door (4.6), run
plans with resumable split-aware batch runs (4.7), a diagnostics report (4.8),
doc drift fixes (4.9), the pilot dataset + review images (4.10), and the
clipping failsafe (4.11). Simulation behavior (world gen, steering, reachability,
debounce, collision, timing, RNG) was **not** changed.

**Built.** `src/headless/bot.ts` (seven personas + three mixture profiles, bot
RNG separate from world RNG), `src/headless/driver.ts` (v2 manifest, action
alignment, clipping failsafe, scripted `botOverride`), `src/headless/plan.ts`
+ `src/headless/workerPool.ts` (`runPlan`: resume, disjoint seed ranges,
deterministic episode dirs, retry-once, early-stop), `src/headless/cli.ts`
(plan CLI + legacy flags), `src/headless/provenance.ts` (configHash /
engineCommit / dirty / engineSourceHash), `src/headless/report.ts` (the ten
4.8 items), `docs/DATASET-SCHEMA.md` (the data contract). **No render change was
needed for the click pulse (4.5a):** it is drawn after the blit, and
`captureFrame` reads the pre-pulse `outputTarget`, so it never reached captured
frames (confirmed by a 400-event paired pixel probe).

**Key findings.**
- *Section 3.* The earlier "three signal bins (0 / +1 / −1)" observation does
  **not** reproduce: the real placeholder bot's held-signal histogram is spread
  across all 0.1 bins, exactly as the held-signal model predicts. The signal
  path is correct; the old number was an earlier-investigation artifact. No
  engine change.
- *4.6 straightest-target.* The literal per-frame criterion reads **88.7%**
  (above the ~65% threshold) — but this is the per-frame-vs-moments confound
  §3(b) already flags: the `uniform` persona (uniform random X) also reads
  90.2%. Measured at real choice moments (accepted clicks with 2+ reachable),
  the straightest-choice rate is **48.8%** — i.e. choices are spread ~uniformly
  as intended. No tuning applied (see STAGE2-REPORT for the reasoning).

**Verification.** `tsc --noEmit` clean; 56 vitest tests pass (action alignment,
run plans, provenance, failsafe trim + delete paths). Pilot `engine/data/pilot_v1/`:
150 episodes (145 complete + 5 aborted = 3.3%), 88,047 frames, 6.70 GB, 134 s
on the RTX 5070. Diagnostics report and review images written to the pilot dir.

---

## 2026-10-08 — Config-drift world-generation fix + single-sided walls + wall thickness

**Asked (three requests):** (1) make the world generator resilient to the
enlarged room sizes — the 4 failing rule tests must pass without weakening the
tests; (2) walls *and* their blackout doorways should be visible only from the
inside (fixing back-to-back wall z-fighting); (3) give walls a bit of thickness
(visible only in doorways), without ever clipping.

**Root cause (request 1).** Rooms were enlarged to 10–20 while `HALLWAY_LENGTH`
is 8–30 and doorway spacing was `doorwaySpacingFactor * doorwayWidth` = 8.1. A
side-doorway room's "along" dimension (width for E/W travel, depth for N/S)
must fit inside its corridor *cell*, but cells could now be smaller than
`ROOM_WIDTH_MIN`/`ROOM_DEPTH_MIN` (10). `generateRoom` then (a) shrank rooms
below the configured minimum, and (b) produced a degenerate offset region so the
room poked out of its cell into an adjacent room or a crossing corridor. The
seed-13 case was a 9.83-long hallway whose two side rooms (min 10) overflowed
into the parent room at its start and the bend child at its end.

**Fix (request 1).** `placeSideDoorways` (`engine/src/world/doorway.ts`) now
derives `minAlong` from the travel axis and enforces every side room's cell ≥
`minAlong`: consecutive doorways are spaced `max(minSpacing, minAlong)` apart,
each side's cell accounts for the bend turn side (`W/2` at the end) and a bend
child's parent side (`W/2` at the start), and a side whose cell is below
`minAlong` gets **no** doorways (a too-short hallway now yields no side rooms
instead of overlapping ones). `generateHallway` picks the bend turn side up
front and threads `turnSide`/`startCorridorSide` down so placement knows the
back-offs; `generateNeighbor` passes `startCorridorSide` in via `HallwayOpts`.
`generateRoom`'s shrink branch is defensively clamped to never go below the
configured minimum.

**Request 2 (single-sided).** `materials.ts` floor/wall/ceiling changed
`DoubleSide → FrontSide`; `geometry.ts` `addQuad` normalizes winding against the
passed normal so every wall/floor/ceiling front-faces its interior (previously
half the walls faced outward and `DoubleSide` hid it); `blackout.ts` dropped
`DoubleSide` so the fill is culled from the far side. Back-to-back coplanar
walls now each face their own interior and no longer z-fight.

**Request 3 (thickness).** Added `WALL_THICKNESS = 0.5` (`config.ts`);
`geometry.ts` `buildWall` now extrudes jamb/lintel faces **outward** at each
opening so doorways read as having a reveal. Extrusion is outward-only and
collision stays floor-based, so walkable space is unchanged and it can never
clip.

**Decisions (user-confirmed):** `ROOM_HEIGHT_MIN` raised 2.5 → **2.8** (equal to
`DOORWAY_HEIGHT`, so a doorway always fits its wall); wall thickness **0.5**.

**Verification.** `vitest run` 52/52 pass; `tsc --noEmit` clean; a 400-seed ×
80-step sweep (46,882 room instances) found 0 under-min rooms, 0 room–room
overlaps, and 0 room–hallway overlaps. Rendering changes (2 & 3) are
typecheck-clean but **not yet visually verified** — still need a human
playthrough / recorded run.

---

## 2026-10-08 — Wall-crash in a bend hallway (recorded playthrough, seed `3447179621`)

**Asked:** "I'm still able to make the player crash into a wall… around frame 80
[actual: frame 283]… Does it have to do with my anti-sudden-movements,
jerk-restriction code? … Find the root cause whatever it is, I might be wrong."

**Root cause (NOT the jerk restriction).** The hallway `h7` terminates in a
~90° north bend. Its doorway's `entryPoint` is a waypoint
`TURN_WAYPOINT_DEPTH` units **past** the corner — and `TURN_WAYPOINT_DEPTH` had
been hand-tuned from `3` to `30` (the `//3` comment was left as a hint). The
candidate collection in `Engine.collectCandidates` projected and
reachability-tested that doorway against `entryPoint`. From the player
approaching the corner, a waypoint 30 units past the corner sits ~75–86° off
heading, landing **outside** the 75° front cone — so the bend was marked
unreachable. With no reachable candidate, the sticky target resolved to `null`,
and `Steering.step(null)` froze the player's diagonal heading (they had entered
`h7` at ~0.393 rad), drifting them straight into the south wall.

**Fix (two parts, both required for robustness):**
1. Reverted `TURN_WAYPOINT_DEPTH` `30 → 3` (`engine/config/config.ts`).
2. `Engine.collectCandidates` now evaluates candidates against the doorway's
   **approach point** (new helper `doorwayApproachPoint` in
   `engine/src/sim/navigation.ts`) — the corner for a bend, the opening center
   otherwise — for *both* screen-X projection and reachability. This is the
   point the player actually steers toward on the approach leg, so it can never
   falsely fail the front cone regardless of how `TURN_WAYPOINT_DEPTH` is later
   tuned.

**Verification:** new regression test `tests/regression.test.ts` replays the
recorded click sequence through a faithful renderer-free replica of
`Engine.step` and asserts the player navigates the bend with **0** collision
events and never leaves the floor. Plus a unit test in `tests/navigation.test.ts`
("measures a bend doorway reachability against its approach point, not its
waypoint"). `tsc --noEmit` clean.

---

## 2026-10-05 — Four playability bugs (carried over from the old `handoff.md`)

The following four fixes predate this log's introduction and are preserved here
for continuity. Each is guarded by a regression test in
`engine/tests/world.rules.test.ts` or `engine/tests/navigation.test.ts`.

### Dead-end rooms
Rooms could be generated with no reachable exit from their entry (extra doorways
landed on the entry wall or near the entry corner, behind the ~75° cone). Fix:
`placeRoomDoorways` always places the first extra doorway on the wall **opposite**
the entry. Regression test: "every room has a reachable exit doorway from its
entry."

### Extra wall / room overlap
A room on a bend corner poked into the continuation corridor (and its mirror at
the bend child's start), rendering an "extra wall" that vanished on entry. Fix:
`backOffCorner` backs side rooms off by `W/2` from the relevant corridor end;
added `EntityDescriptor.startCorridorSide`. Regression tests: room-vs-room and
room-vs-hallway overlap checks.

### Phantom doorways
A room could get a second doorway on its **entry wall**, occluded by the parent
hallway's solid wall but crossable (floor-based collision). Fix:
`placeRoomDoorways` never places an extra doorway on the entry wall. Regression
test: "never places an extra doorway on a room's entry wall."

### Bend-corner wall-hit + shallow-diagonal doorway penetration
A hallway bend and tight side-doorway/junction turns all sit on a wall the player
drives parallel to, so they must "drive straight, then turn ~90°." Fix: two-phase
turn model — `Doorway.approachPoint` (the corner / corridor centerline abreast of
the opening / junction node) for the approach leg, `entryPoint` (waypoint past
the corner) for the through leg, and per-doorway gate radii
(`TURN_GATE_RADIUS` = 2.5 for bends, `DOORWAY_TURN_GATE_RADIUS` = 1.5 for tight
turns). Regression tests steer the real pipeline through every bend / side
doorway / junction turn and assert perpendicular entry with no pinning.

**Correction (2026-10-10, Stage 2).** The "per-doorway gate radii" description
above is wrong. `DOORWAY_TURN_GATE_RADIUS` does not exist in `config.ts`, and
`Doorway.gateRadius` is never assigned. Only **bend** doorways set `turn: true`
and an `approachPoint`; side doorways and junction turns use the ordinary entry
waypoint. The two-phase model therefore applies to bends, and tight side /
junction turns are not gated. The README config table was corrected accordingly.

---

## Open items / known issues

- **Visual verification of single-sided walls + wall thickness (not yet done).**
  Backface culling and the 0.5-unit outward wall extrusion are implemented and
  typecheck-clean, but need a human playthrough / recorded run to confirm: walls
  render from inside only, no z-fighting at doorways, doorways show a visible
  jamb, and walking through a doorway shows no clipping or see-through.

- **Discard-on-close for recordings.** `engine/src/headless/record.ts` writes
  the manifest even when the player closes the window without pressing Q. Fix:
  write to a temp dir and only keep/rename on Q.

- **Corner-hugging steering (aesthetic polish).** Turns are functionally correct
  (perpendicular entry, no clipping); the remaining ask is cosmetic "don't hug
  the corner as close as possible," tunable via `TURN_RATE_GAIN` /
  `MAX_TURN_RATE` / `JERK_THRESHOLD` in `engine/src/control/steering.ts`.

---

## Operational notes (recurring gotchas)

- `npx` is broken in this environment — invoke binaries directly (see README
  "Testing"): tests `node node_modules/vitest/vitest.mjs run`, typecheck
  `node node_modules/typescript/bin/tsc --noEmit`, build
  `node node_modules/vite/bin/vite.js build`, scripts
  `node node_modules/tsx/dist/cli.mjs <file>`. Run these from `engine/`.
- The `Read` tool cannot view the PNG frame captures; inspect a frame by
  delegating to a vision-capable subagent, or ask the user (they describe frames
  precisely by frame number).
- `World` is lazy: to replay a recorded path you must *walk* it —
  `world.init()`, then `world.maintain(active.id)` and follow the doorway whose
  `targetEntityId` is the next id. Never `maintain` a non-active id directly.
- Recording manifests record `seed` + per-frame click sequence but not room
  coordinates; regenerate `new World(seed, CONFIG, { buildGeometry: false })`
  to reconstruct geometry.
- Headful recorder (`npm run record`) uses different Chrome launch args than the
  headless pipeline (`HEADFUL_LAUNCH_ARGS` vs `GPU_LAUNCH_ARGS` in
  `engine/src/headless/driver.ts`) — Vulkan/ANGLE has no surface in a windowed
  X11 session.
