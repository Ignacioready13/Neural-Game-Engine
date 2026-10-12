# Stage 5 progress — the real generator (action-conditioned, steerable, stochastic)

Running notes file (brief §0 rule 9). Read this first if the session restarts.
Newest entries at the top of "Run log"; the plan below is stable.

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


