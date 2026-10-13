# Stage 5 progress — the real generator (action-conditioned, steerable, stochastic)

Running notes file (brief §0 rule 7). Read this first if the session restarts.

================================================================================
## MORNING SUMMARY  (Stage 5b overnight — read this first)

**Phase** 2026-10-12 00:12 – 04:30 MDT (deadline 07:42; finished early). Machine left
idle: no background jobs, no stray processes.

### What won

**`neural/runs/flow96c/best.pt`** — a 3.19 M-parameter denoiser (width 64 + a
self-attention block on a 9x12 grid) with a 4-step self-rollout curriculum, trained for
19.6 min on top of the unchanged `steer96` core. It replaced `flow96` (`v0`).

The user's complaint, in numbers (frozen "a walkable opening is prominently in view?"
detector, 300-step closed-loop frames):

| | opening share @300 | ratio to real |
|---|---|---|
| real | 0.525 | 1.00 |
| **v0** (start) | 0.125 | 0.238 |
| **flow96c** (winner) | 0.581 | 1.106 |
| flow96c on the **confirmation set** (40 disjoint clips) | **0.575** | **1.015** |
| v0 on the same confirmation set | 0.067 | 0.118 |
| flow96c on the **test split** | 0.442 | 0.758 |
| flow96d (rejected) | 0.000 | 0.000 |

So: **8.6x the incumbent's opening share on clips neither model was selected on**, and
0.76–1.02x real on the two held-out sets. Still 37.1 fps at 96x72 (1.5x the 25 fps
floor). Training energy for the winner: **32.1 Wh**.

### The night's most useful negative result

`flow96d` (edge-weighted reveal loss, 25k steps) looked better on *every* short-horizon
signal — lower flow loss (0.0021 vs 0.0023), lower 3-seed sampled one-step L1
(0.0674 ± 0.0008 vs 0.0744) and a one-step doorway-reveal ratio of 0.914 against
0.770 — and its 300-step opening share was **0.000**. The frozen long-horizon scorecard
rejected it (`eval_out/decision_flow96c_vs_flow96d.json`). This is why the scorecard was
frozen first. Its checkpoint and row are kept.

### What to look at first

1. `neural/review/stage5b/longrun_scripted_alt.png` — v0 (top) vs the winner (bottom),
   one frame every 20 steps to 300. This is the image the user's complaint is about.
2. `neural/review/stage5b/rollouts_clip*.png` — real above generated, 4 clips, every
   4th step to 60.
3. `neural/review/stage5b/counterfactual_*.png` — left / right / no click, 4 episodes.
4. `neural/review/stage5b/reveals.png` — real, Stage 4 baseline, 4 seeds.
5. `neural/review/stage5b/play_flow96c.gif`, `candidate_{v0,flow96d}_strip.png` — to
   overrule the pick.
6. `neural/docs/STAGE5B-REPORT.md` — the technical report.
7. `neural/docs/ENERGY-LEDGER.md` — the energy totals.

### Energy, headline

* Measured since the phase began: **374.28 Wh GPU** (353.83 Wh idle-subtracted) +
  **187.71 Wh CPU package** over **5.03 h**; 18.68 Wh of that was wasted on my own
  aborted attempts.
* The winner's lineage: **32.13 Wh** measured (`flow96c`), plus the pre-logger core it
  sits on, estimated at ~170 Wh.
* Pre-logger Stage 5 (1.7 h core + 2.7 h generator/gates): **~359 Wh as a labelled
  estimate**, never mixed into the measured totals.
* Live play: 44.9 W mean, **1.21 J/frame, ~45 Wh per hour of play**.

### Anything alarming?

* **Nothing is broken, and nothing is fully solved.** The generator still does not
  translate enough content in closed loop (0.8 px per 20 steps against 10.2 real — part
  of which is correct for the no-click scenario), frame-to-frame change is 4–9x too
  small, and the faint dark-region tint persists (now slightly *blue*-shifted on the
  winner instead of red). See report §8.
* **Two hours were lost to my own flag error** (`--steps` is the ODE sampler count, not
  the training budget) and another ~20 min to orphaned DataLoader workers from the killed
  runs thrashing this machine's tight RAM. Both are now guarded: `train_flow.py` refuses
  `--steps > 64`, and `night_queue.sh` waits for both an empty process list *and* ≥2.5 GB
  free VRAM before starting (a job once raced the previous job's teardown and died with a
  CUDA OOM).
* **GPU thermal**: every long run reported `sw_power_cap` and `sw_thermal_slowdown`
  throttle events at ~98–102 W mean against a 115 W cap; the measured frame rates already
  include that.
* `I3` (stack-level readout fine-tune) was **not** run — the user said clicks clearly turn
  the view and asked for no steering change, so the budget went to validating the
  structure fix. Reason recorded in report §6.

## DECISIONS MADE WITHOUT USER  (brief §0: no interactive questions after 01:00)

================================================================================

D1. **Opening-detector label.** A plain "is any doorway on screen?" label is true in
    **97.4 %** of val frames (measured on 34,744), so it cannot be discriminative.
    Frozen instead as: positive = a *reachable* doorway candidate with
    `screenDistance <= 0.35` and `|screenX| <= 0.35` — "a walkable opening is
    prominently in view". Base rate 57.6 %. Negative controls = the 4.1 % of frames
    with no candidate inside `|screenX| <= 0.98` (fog / black far wall).
    Undo: retrain with a different `OPEN_T` in `judges.py`; the scorecard rows are
    then not comparable, so this must be a new scorecard version.

D2. **Detector has a dark bias; fixed by input instance-normalisation.** At v1 the
    detector fired on 94 % of the darkest no-door frames (fog) and 19 % overall.
    v4 (frozen) adds per-frame instance normalisation, 3x oversampling of no-door
    controls, and a threshold calibrated on real frames so the real positive share
    equals the sidecar base rate. Result: precision 0.61, recall 0.56, no-door
    control fire rate 10 %, dark-control fire rate 17 %. **This judge is mediocre at
    the frame level and that limit is reported, not hidden.** It is still
    informative because its real-vs-generated separation is enormous (67 % vs 0 %).

D3. **Scorecard cost.** The full frozen set is 40 clips x 3 scenarios x 3 seeds x
    300 steps = 360 rollouts. At batch 1 that took ~12 s/rollout. I added
    `collect_batched` (identical semantics, verified by a test) so the same frozen
    set runs ~4x faster. The *set and metrics* are unchanged; only the execution is
    batched. Undo: call `collect` instead of `collect_batched`.

D4. **Candidates are scored on the same frozen sets**, seed-averaged, exactly as
    the brief specifies; nothing in the set definition was tuned after seeing a
    result.

## Candidate table (frozen scorecard v1, 40 clips x 3 scenarios x 3 seeds)

| candidate | D ratio @150 | D ratio @300 | R spec_d @300 | R lap @300 | S cf / shift / sust-sign | fps | decision |
|---|---|---|---|---|---|---|---|
| **v0** = flow96 on steer96 | 0.290 | 0.238 | 1.215 | 1.472 | 0.962 / 0.810 / 0.506 | 40.8 | incumbent (start) |
| **flow96c** = 3.19 M denoiser + attention, 12k steps | 1.593 | **1.106** | 1.188 | 1.499 | 0.962 / 0.883 / 0.514 | 37.1 | **KEEP → v1** |
| flow96d = flow96c + edge-weighted reveal loss, 25k steps | 0.179 | **0.000** | 1.949 | 2.039 | 0.962 / 0.819 / 0.478 | 37.1 | **REJECTED** (kept) |
| I3 (core DAgger) | — | — | — | — | — | — | not run: user said steering is fine; budget went to validation |

`D ratio` = frozen opening-detector positive share on generated frames divided by the
same on real frames; **1.0 is the target**. v0 shows a doorway in 12.5 % of its
300-step frames against 52.5 % of real ones; flow96c shows 58.1 %.

**flow96d is the night's most useful negative result.** Lower `val_flow` (0.0021 vs
0.0023), lower 3-seed sampled one-step L1 (0.0674±0.0008 vs 0.0744) and a better one-step
doorway-reveal ratio (0.914 vs 0.770) — and it is still much worse in closed loop: the
frozen detector fires on **0 %** of its 300-step frames (flow96c 58 %, real 52 %). Neither
the training loss nor any one-step metric can see this; the frozen long-horizon scorecard
can, which is the whole point of freezing it first.

Verdict for flow96c (noise-aware, `eval_out/decision_v0_vs_flow96c.json`): D improves
beyond the incumbent's measured seed-to-seed noise at h=300 (gain 0.656 vs noise
0.164) with the same direction at h=150 (gain 0.117 vs noise 0.174 — direction only,
not beyond noise, and the mean 0.117 vs 0.174 is the thing to check if the user wants
more evidence), H (motion/drift) improves, R is unchanged **within** noise, steering
counterfactual and shift ratio hold, 37.1 fps ≥ 25, no collapse/freeze. **KEEP.**
The strict reading (both rule horizons individually beyond noise) is recorded as false
in the same JSON, because the h=150 real reference is only 40 clips x 5 frames.

## Run log (Stage 5b, newest first)

- **01:47** `flow96c` (I2 candidate) done: denoiser 3.19 M params (base 64 +
  low-res self-attention), self-rollout curriculum self_k 4 / p 0.25, 12000 steps in
  1177 s (10.2 steps/s), 32.13 Wh GPU, 13.25 Wh CPU, mean 98.1 W, **throttle events:
  `sw_power_cap`, `sw_thermal_slowdown`**. `val_flow` 0.006 → **0.0023** (incumbent
  stopped at 0.0027), i.e. the 4x larger denoiser has a genuinely lower flow loss;
  `val_sample_l1` is noisy as always (0.0744 best at step 7000, 0.0794 at 12000).
- **01:00-01:35** Two false starts on the same run, both my own CLI error: `--steps` is
  the **ODE sampling-step count**, not the training budget, so `--steps 8000/12000`
  meant 8000-12000 U-Net evaluations *per generated frame*. Symptom: 100 % CPU, GPU at
  98 %/104 W, and no step logged in 16 minutes. Fixed by (a) using `--max-steps` and
  (b) making `train_flow.py` refuse `--steps > 64` with a message naming `--max-steps`.
  Also found and killed a pile of orphaned DataLoader workers from the killed runs,
  which were competing for the machine's tight RAM (30 GB, VS Code resident).
- **00:45** Frozen scorecard written (`eval_sets/scorecard_v1.json`) and the opening
  detector frozen (`eval_out/opening_detector.pt`). Judges validated against
  known-different cases. Diagnosis run (CPU): zoom explains 0.08 % of the one-step
  residual → **I1 skipped**.
- **00:14** Idle baseline: 4.07 W GPU. RAPL became readable at ~00:20 (it was root-only
  at 00:13), so CPU package energy is measured from then on.

## Status

- Milestone 0 (orientation): done — env verified (torch 2.14.1+cu130, RTX 5070
  sm_120, `gpu_check.py` passes real matmul+conv-backward through `memcap.sh`;
  systemd-run user scope available so the RAM cap is real).
- Milestone 1 (steering core): in progress.
- Milestone 2 (generator): not started.
- Milestones 3-5: not started.

## Plan (and the reasons)

Order: M1 steering core → M2 generator → M3 stability → M4 final/speed → M5 player.
Budget estimate (brief rule 7, ~10-12 GPU h total, ≤ ~90 min per run):
M1 ~1.0 h, M2 ~4 h (64x48 dev + one 96x72 final), M3 ~1 h (eval only),
M4 ~0.5 h (eval + speed), M5 ~0.5 h. Development at 64x48, final recipe 96x72.

Design (from the Stage 4 report §9 + §7.12):

1. **Steering is carried by a structured motion output, not by the denoiser.**
   A learned readout (context frames + click history incl. an explicit
   `deltaActionX`) predicts the ego-motion (per-step horizontal image shift,
   supervised *during training only* from the sidecar heading + a phase-correlation
   shift table). That shift drives a differentiable warp of the last frame
   (`grid_sample` translation). The warp *is* the steering; the generator only
   invents newly-revealed content.
2. **Why this shape**: at 20 Hz a click moves the next frame ~0.03 px, so no
   one-step pixel loss can reward reading it (§7.1); supervising where the effect
   is pixel-scale (H≈10) makes the direction learnable (§7.7, P4 0.85 vs oracle
   0.87); a plain pixel-residual decoder cannot render the ~7 px shift even when
   told the direction (§7.11/§7.12), while a warp + (constrained) residual reaches
   both-correct 0.87 (§7.12). So: explicit direction signal + structured warp.
3. **The residual must not shadow the warp** (§7.12). M1 keeps it low-resolution and
   gated; M2 hands new content to the stochastic generator, which is conditioned on
   the *already-warped* frame.
4. **Stochastic generator**: conditional flow matching (rectified-flow), 4 sampling
   steps, trained from scratch, conditioned on the recent frames, the click
   features, the readout, and the motion-warped last frame.
5. **Data**: bulk_v1 (876k frames, 1086/198/199 train/val/test). Mirror augmentation
   (flip image, negate actionX and deltaActionX, sign-flip readout targets).
   Turn-onset / door-reveal oversampling; keep steady walking, never-clicked and
   stale-held-click regimes in the mix; drop `viewUnsafe` windows.

## Key implementation files (Stage 5)

- `neural/steer.py` — click-history features, differentiable warp, direction
  readout, `SteeringModel` (readout + warp + gated low-res residual).
- `neural/train_steer.py` — trains the steering core (M1).
- `neural/flow.py` — conditional flow-matching generator + `FlowSteerModel`.
- `neural/train_flow.py` — trains the generator (M2).
- `neural/shift_table.py` — precomputes the per-frame per-step image-shift table.
- `neural/eval_harness.py` — extended additively (seed-aware predict, metrics,
  gate runner).

## Run log

(filled after every run: id, config, result, checkpoint path)

- `steer_smoke` (pilot 64x48, 60 steps) — pipeline smoke test, OK.
- `steer64` — 6000 steps, phase-correlation target, no DAgger. val sign 0.912,
  shift ratio 0.95 (teacher-forced) but **closed-loop commanded ratio 0.32,
  sign 0.67** → bistable feedback collapse. Weights: `runs/steer64/best.pt`.
- `steer64b` — same + DAgger roll-in (p=0.5, kmax 6). closed-loop ratio 0.69,
  sign 0.73. `runs/steer64b/best.pt`.
- `steer64c` — + heading-derived (denoised) supervision target. val sign 0.944,
  ratio 0.86; closed sign 0.80, ratio 0.58–0.9. `runs/steer64c/best.pt`.
- `steer64d` — + fixed `frames-since-click` feature (the schema's
  `actionHasClicked` is a latch, so the feature must be derived from actionX
  changes). See below.

- `steer64e` — + counterfactual "click the other door" augmentation (target from a
  fitted engine-rule geometry, `cum_H ~ coef_m · screenX_target_door`,
  coef ≈ −9.5 px per screenX unit, sign agreement 0.80 on 35k train windows).
  First version had a **sign bug** in the counterfactual target (double negation) —
  inverted the model's response; fixed.
- `steer64f` — the fixed counterfactual augmentation. G1: counterfactual both-
  directions 0.77 (near 0.79), shift ratio 0.76, sustained median ratio 0.92 with
  overshoot ≈ 0 but sign agreement 0.66; readout vs truth 0.84, vs engine-rule
  oracle 0.807 (oracle vs truth ceiling 0.954). `runs/steer64f/best.pt`.
- `steer64g` — + **spatial click channels** (a Gaussian click column plus the
  deltaActionX map, so the encoder can compare the click position against the door
  layout locally instead of multiplying a scalar by a global feature).
  `runs/steer64g/best.pt`.

- `steer64h` — + `SINCE_CLIP` 24 and DAgger `kmax` 12. G1: counterfactual
  both-directions 0.880 (near 0.878), shift ratio 0.780, sustained median ratio
  0.680, sign 0.764 (0.833 on turns >= 10 px), overshoot -0.22; readout vs truth
  0.923, vs engine-rule oracle 0.898 (ceiling 0.954). **The development steering
  core.** `runs/steer64h/best.pt`.
- `flow64`, `flow64b`, `flow64c`, `flow64d`, `flow64e` — generator development, in
  order: (a) the un-warped context frames let the denoiser reconstruct the pre-warp
  content and cancel the steering (closed-loop ratio 0.02 -> 0.36 after removing
  them); (b) unit-scale flow noise leaks a high-frequency residue into every frame
  (60-step sharpness 3.7x real) — matching the noise scale to the residual (0.08)
  fixed it; (c) the big one: **an L1 velocity loss collapses the sampler to the
  median residual ~ 0**, degenerating the generator to "motion-warped last frame"
  (|residual| 0.02 vs a real per-step change of 0.2) — MSE restored real content and
  doubled seed diversity (0.015 -> 0.030). `runs/flow64e/best.pt`.

- `steer96` — the final 96x72 core (14000 steps, 35 min). G1 val: counterfactual
  both-directions 0.967 (near 0.962, far 1.000, per-side 1.000/0.967), shift ratio
  0.669, sustained median ratio 0.566 (commanded 0.485), sign 0.663 (commanded
  0.80), readout vs truth 0.857, vs oracle 0.822 (ceiling 0.948). Test split:
  both-directions 0.950, ratio 0.577, readout 0.837/0.823. `runs/steer96/best.pt`.
- `flow96` — the final 96x72 generator (8000 steps, 17 min), on top of `steer96`.
  G2 val: counterfactual both-directions 0.947 (near 0.938), shift ratio 0.853,
  one-step sharpness 0.998x real, colour gap 0.004, seed diversity 0.034 (reveals
  0.053 vs steady 0.049), 60-step sharpness 1.53x real. G4: 67.9 fps mean, 210 MB
  peak VRAM. G3 (300 frames): sharpness 1.8-1.9x real, motion 0.23 vs real
  0.83 px/step. `runs/flow96/best.pt`.

## Final state

- **Final model**: `runs/flow96/best.pt` (96x72, 1.28 M params, 4 sampling steps,
  67.9 fps, 210 MB VRAM). Development model: `runs/flow64e/best.pt` (64x48) on
  `runs/steer64h/best.pt`.
- **Gates** (all in `neural/eval_out/`): G1 val+test, G2 val+test, G3 val+test,
  G4 = PASS; hardware/sparsity in `hw_flow96.json`.
- **Artifacts** for the user: `neural/review/stage5/{rollouts,counterfactual,reveals,longrun}.png`
  plus `play_scripted.gif` (a scripted headless player run) and its strip.
- **Tests**: `pytest` 42 passed.
- **Open**: the sustained-steering sub-gate (40-step sign 0.66-0.76 vs 0.8) and the
  long-horizon drift (sharpness 1.8-1.9x, motion fading). Both analysed in
  `STAGE5-REPORT.md` §3, §5, §11 with the next steps to try.

## Measurement notes

- A single phase correlation over a 40-step rollout **aliases** once the accumulated
  shift approaches half the frame width (measured: a commanded +37 px summed shift
  read as +3 px). Stage 5 therefore measures the accumulated shift in 8-step blocks
  and also reports the model's *commanded* shift (the warp it applies), noise-free.
- The M1 residual collapsed to exactly zero (its gate is driven to 0 and it has no
  pixel target of its own), so the 64x48 steering core is **warp-only** — the
  cleanest version of the Stage 4 P6 result, and the reason M1 frames are blurry:
  M2's generator is what restores detail.
- Stage 5 metrics take either the diffusion convention (float in [-1, 1]) or raw
  uint8 frames; `metrics._as_nchw` decides by dtype. (Mixing them silently made
  every "in [0, 1] units" number meaningless before this was fixed.)

## Decisions and reasons

- **Supervision target = heading-derived shift, not the raw phase-correlation
  per-step estimate.** At 20 Hz one step of a turn is ~0.5 px while the
  phase-correlation noise floor is ~0.4 px, so regressing on it attenuates the
  prediction (regression to the mean) and blurs the turn onset. The sidecar's
  `playerHeading` is exact; `dshift = scale * dheading` with `scale ≈ −35.3 px/rad`
  at 64×48 (corr −0.87) is a clean target. Training-only; gates still measure the
  generated frames with the phase-correlation estimator.
- **DAgger roll-in is required for the closed loop.** Without it the loop is
  bistable (model does not start turning → its frames show no turn → it never
  starts). Rolling the model forward k ≤ 6 steps and supervising on those frames
  took the closed-loop ratio 0.32 → 0.69.
- **Counterfactual clicks are the only thing that forces the readout to use the
  click.** With consistent data (the click always matches the scene) a frame-driven
  readout reaches 0.9+ sign accuracy while ignoring the click entirely: injecting a
  different door's click into an otherwise identical context left the prediction
  unchanged (L = R = hold). Training with "click the *other* door" (target from the
  fitted geometry) made the per-side counterfactual direction accuracy 0.94/0.82.
- **`actionHasClicked` is a latch**, not a per-click event (schema §3): "≥1 click
  accepted as of this step". So frames-since-click must come from *changes in
  `actionX`* (the held signal only changes on a new accepted click; the
  never-clicked default holds `actionX == 0`).
- **Scratch caches live in `neural/data/_cache/`**, not inside the packed sets
  (the packed datasets are read-only per rule 3).
- **G1 counterfactual is measured two ways**: a one-forward-pass probe (the exact
  analogue of `probe_click.py` for a model with an explicit motion output, free of
  feedback confounds) and the closed-loop image shift. "Both-correct" is reported
  as *directions* (left turns left and right turns right, each beyond 0.5 px — the
  `counterfactual_eval` convention) and as *ordered* (`L > hold > R`), since the
  strict ordering fails wherever the player is already mid-turn.

D8. **I1 (zoom/forward-motion in the structured warp) skipped.** The diagnosis measured
    that a zoom term reduces the one-step residual by 0.08 % — the brief explicitly
    allows skipping or reordering an iteration when the diagnosis says to.
    Undo: implement the zoom path in `steer.py` (the supervision table exists:
    `zoom_table.py`, whose sidecar-derived bearings scale with forward motion and which
    agrees in sign with image-based estimates).

D9. **I3 (stack-level readout fine-tune) not run.** The user's playtest said clicks
    clearly turn the view and asked for no steering change, so the remaining budget went
    to validating the structure fix on the confirmation set and the test split. The S
    family still shows a gap (sustained sign 0.514 with the generator in the loop).
    Undo / next time: `train_steer.py --frame-model <generator> --dagger-kmax 3
    --dagger-prob 0.3` (a bigger roll-in is hours: the generator's sampling steps
    dominate), then re-fit the generator with `train_flow.py --core <new core>`.

D10. **The keep/revert rule was implemented against the recorded seed noise**, as §6 of
    the brief requires ("each comparison judged against the incumbent's measured
    seed-to-seed noise"), and family H is scored on the two things §6 names for H
    (drift and motion). `frame_change` and the dark-region tint are reported as
    non-deciding detail because the long-run sequence statistics are collected at one
    sampling seed and therefore have no measured seed noise. Both readings — including
    the strict "every rule horizon individually beyond noise" one — are in
    `eval_out/decision_*.json`; the strict reading is `false` for flow96c because the
    real reference at a single horizon is only 40 clips x 5 frames.

D11. **The confirmation-set run does not measure family H** (its long-run statistics are
    collected only for the frozen `long_clips`, which are main-set clips), and the
    test-split run does not measure family S (the frozen steering contexts are val-only).
    Both are stated in the report rather than silently omitted.
