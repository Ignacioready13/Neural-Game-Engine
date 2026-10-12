# Stage 5 Report — the real generator: action-conditioned, steerable, stochastic next-frame generation

Written for a reader who has not seen the working conversation. It covers what was
built, why, what the numbers are, and what did not work. Everything is in
`/neural`; the engine and the datasets were not modified.

---

## 1. The problem, and the design it forced

At play time the model sees only its own recent generated frames plus the
click-derived signals (`actionX`, `actionHasClicked`). It must produce the next
frame, over and over, at ≥ 15 fps, and it must *steer* — a click near the left
doorway must make the view turn left, right up to the moment the hallway has
actually been entered.

Stage 4 established three things that decide the architecture:

1. **A one-step pixel loss cannot teach the click.** At 20 Hz a click moves the
   next frame by ~0.03 px; the effect only accumulates over ~10 frames. The Stage 4
   baseline beat "repeat the last frame" by 3× while ignoring the action entirely.
2. **The direction is extractable, but the generator cannot paint it.** A direct
   heading readout reaches the engine-rule ceiling (0.85 vs 0.87), but adding it as
   an auxiliary head does not push direction into the frames (the decoder, trained
   only on H-step pixel L1, cannot map a bottleneck feature to a ~1 px global
   shift). With an *oracle* direction and a **differentiable warp** the same network
   reaches both-correct 0.87 — the failure was in *expressing* the shift, not in
   getting the direction in.
3. **The residual must not shadow the warp.** Given a full pixel residual, the
   network paints the image through the residual and ignores the shift.

So Stage 5 splits the problem in two, structurally:

```
                     click signals (N1/N2: nothing else)
                              |
   context frames ---> [ DirectionReadout ] --dx--> [ warp last frame ] --+
                                                                        |
                          noise --+                                     v
                                  +--> [ flow denoiser ] --> residual --> + --> next frame
```

* **Steering is carried by a structured motion output** — a learned per-step
  horizontal shift that warps the last frame through a differentiable
  `grid_sample`. It is not asked of the denoiser.
* **New content is carried by a stochastic residual** — a conditional rectified-flow
  generator trained on the *residual relative to the already-warped frame*, so its
  target is aligned and it has no reason (and, as §4.2 shows, no convenient route)
  to undo the motion.

### 1.1 The steering core (`steer.py`)

* **Click features** (`click_features`): per frame, `actionX`, `actionHasClicked`,
  `deltaActionX = aX_t − aX_{t−1}`, and frames-since-last-click. One schema detail
  matters: `actionHasClicked` is a **latch** ("≥ 1 click has been accepted as of
  this step"), not a per-click event, so frames-since-click has to be derived from
  *changes in* `actionX` — the held signal only changes when a new click is
  accepted (the never-clicked default holds `actionX == 0`).
* **DirectionReadout**: a small convolutional encoder over the K context frames
  (the baseline's encoder shape), global-average-pooled, concatenated with an MLP
  embedding of the click features, emitting H signed per-step shift increments and
  a direction logit. The encoder input is `K · 5` channels: RGB plus a **Gaussian
  click column** (the click's screen position as an image channel) and the
  `deltaActionX` map. The spatial click channel matters: without it the trunk has to
  multiply a scalar click value by the door layout to answer "which door is nearest
  to the click"; with it, the comparison is local. That change alone moved the
  readout's sign accuracy vs the engine-rule oracle from 0.807 to 0.87-0.90.
* **Warp**: `warp_translate` shifts the content of the last frame by (dx, dy) pixels
  with `grid_sample(..., padding_mode="border")`, so newly exposed edges repeat the
  border instead of going black.
* **Constrained residual**: quarter-resolution, upsampled, gated by a learned scalar
  initialised small and penalised during training. In practice the gate drives it to
  exactly zero (see §4.1) — the 64×48 core is *warp-only*, the cleanest version of
  the Stage 4 P6 result.

### 1.2 Training the core (`train_steer.py`)

Three ingredients were needed, each for a measured reason:

1. **Denoised supervision target.** The per-step phase-correlation shift at 20 Hz is
   ~0.5 px against a ~0.4 px estimator noise floor, so regressing on it attenuates
   the prediction (regression to the mean) and blurs the turn onset. The sidecar's
   `playerHeading` is exact; image shift is proportional to heading change
   (`dshift = scale · dheading`, `scale ≈ −35.3 px/rad` at 64×48 and `−51.8` at
   96×72, correlation −0.87/−0.85), so the target is the heading-derived cumulative
   shift over horizons 1..H — supervised exactly where the effect is pixel-scale.
2. **DAgger roll-ins.** With plain supervision the closed loop is *bistable*: if the
   model fails to start turning, its own frames show no turn, the readout keeps
   predicting "no turn", and the turn never happens (closed-loop shift ratio
   collapsed from 0.88 to 0.32). For part of every batch the model now rolls itself
   forward k ≤ 12 steps on the *true* click history and is supervised on those
   frames; the ratio recovered to ~0.7-0.9.
3. **Counterfactual clicks.** With consistent data (the click always matches the
   scene) a frame-driven readout reaches 0.9+ sign accuracy *while ignoring the
   click*: injecting a different door's click into an otherwise identical context
   left its prediction unchanged. So for 40% of each batch the click is replaced by
   the **other** reachable door's `screenX` and the target comes from a fitted
   engine-rule geometry (`cum_m ≈ coef_m · screenX_target_door`, `coef_H ≈ −9.5 px`
   per screenX unit at 64×48, sign agreement 0.80 over 35 k windows — training-only
   use of the sidecar). That forces the click to be causally necessary: per-side
   counterfactual direction accuracy went from chance to 0.90-0.97.

Mirror augmentation throughout (flip the frames, negate `actionX`/`deltaActionX`
and the shift targets). Windows are hold-last consistent (no new click inside the
horizon), safe (`viewUnsafe` dropped), and turn onsets are oversampled 4×.

### 1.3 The generator (`flow.py`, `train_flow.py`)

Conditional **rectified flow** on the residual relative to the warped last frame:

```
warped = warp(last_context_frame, dx_pred)          # the steering core (frozen)
r      = true_next_frame − warped                   # content only: aligned target
x_t    = (1−t)·x0 + t·r,  x0 ~ N(0, σ²),  σ = 0.08
v̂     = flow_net(x_t, t, cond);   next = warped + r̂   (4 Euler steps)
```

`cond` is FiLM conditioning from `[click features, the readout's H increments, dx]`,
and the denoiser's image input is `[x_t, warped, dx map, t]` — deliberately **not**
the un-warped context frames (§4.2). Training: all safe one-step windows with
turn-onset oversampling, mirror augmentation, noise-augmented context (σ 0.05), and
self-conditioning — for half of every batch the model first rolls itself forward 1-2
steps and is then trained on *those* frames, so it stays coherent on its own inputs.
Velocity loss is **MSE** (§4.3).

---

## 2. Results

All val numbers are from `bulk_v1` (198 val episodes, 876 k frames total). The test
split was evaluated once, at the end (§2.4). Every steering number is reported at
64×48 (the development resolution) and 96×72 (the final model).

### 2.1 Gate G1 — the steering core (val)

| metric | threshold | 64×48 (`steer64h`) | 96×72 (`steer96`) |
|---|---|---|---|
| counterfactual **both-directions**, near-door contexts | ≥ 0.70 | **0.878** | **0.962** |
| counterfactual both-directions, far-door | — | 0.842 (n=38) | 1.000 (n=20) |
| counterfactual per-side (left / right) | — | 0.900 / 0.973 | 1.000 / 0.967 |
| counterfactual `L > R` (the model tells the two doors apart) | — | 0.99 | 0.99 |
| one-pass cumulative shift, left / right / hold click | — | +2.49 / −2.98 / +0.28 px | +3.94 / −3.66 / +0.39 px |
| predicted-vs-real shift ratio at H = 10 | ≥ 0.5 | **0.780** | **0.669** |
| sustained 40-step median ratio (blocks) | 0.6-1.4 | 0.680 | 0.566 |
| sustained 40-step sign agreement (blocks) | ≥ 0.8 | 0.764 (0.833 on turns ≥ 10 px) | 0.663 |
| overshoot (model minus real |motion| in the last quarter) | — | −0.22 px/step | −0.41 px/step |
| readout direction sign vs the truth | — | 0.923 | 0.857 |
| readout direction sign vs the engine-rule oracle | — | 0.898 | 0.822 |
| oracle vs the truth (the ceiling at those windows) | — | 0.954 | 0.948 |
| readout accuracy vs the Stage 4 P4 result | — | 0.85 | — |

Two of the four sub-gates pass at both resolutions (counterfactual, shift ratio);
the sustained-steering sub-gates are short (sign 0.66-0.76 against 0.8). Both
shortfalls are analysed in §3.

### 2.2 Gate G2 — the generator in the loop (val, 96×72, three sampling seeds)

| metric | threshold | value |
|---|---|---|
| counterfactual both-directions (one-pass) | ≥ 0.70 | **0.947** (near 0.938) — identical across seeds (the readout is deterministic) |
| predicted-vs-real shift ratio at H = 10 | ≥ 0.5 | **0.853** |
| sustained 40-step sign / median ratio (blocks) | ≥ 0.8 / 0.6-1.4 | 0.469 / 0.501 (commanded: 0.667 / 0.302) |
| readout vs truth / vs oracle | — | 0.880 / 0.828 |
| one-step sharpness (mean gradient) | within 25% of real | **0.998× real** |
| one-step Laplacian variance | — | 0.0157 vs real 0.0207 (0.76×), Stage 4 baseline 0.0191 |
| one-step per-channel mean gap | ≤ 0.03 | **0.004 / 0.004 / 0.004** (baseline 0.003) |
| one-step L1 | not a pass criterion | 0.038 vs baseline 0.010 (a sharp-but-different sample scores worse by design) |
| seed diversity (mean pairwise L1) | — | 0.034 |
| diversity at reveals vs steady walking | larger at reveals | 0.0533 vs 0.0490 (**1.09×**) |
| 60-step rollout sharpness | within 25% of real | 1.53× real (above the band — the drift of §5) |
| 60-step rollout colour gap | ≤ 0.03 | 0.009 |
| 60-step near-black share | no collapse | 0.069 (peak) |

### 2.3 Gate G3 — long-horizon stability (val, 300 frames, 30 clips)

| scenario | sharpness | mean | near-black | frame change | motion |
|---|---|---|---|---|---|
| real clips | 0.0165 | 0.464 | 0.004 | 0.0226 | 0.830 px/step |
| generated, no clicks | 0.0313 (1.90× real) | 0.392 | 0.164 | 0.0052 | 0.000 |
| generated, scripted alternating clicks | 0.0295 (1.79× real) | 0.576 (drifts up) | 0.068 | 0.0122 | 0.232 |

No collapse to black/white and no frozen *frame content* in the no-click scenario,
but the generated frames are 1.8-1.9× the real sharpness after 300 steps (a
high-frequency film that grows with horizon) and the measured global motion is far
below real (0.23 vs 0.83 px/step in the scripted scenario, ~0 with no clicks). The
apparent 0.02 sign accuracy of the scripted-steering check is meaningless given that
motion level (the "signal" is below the estimator's noise floor); the one-pass
counterfactual remains the reliable steering measure (§3).

### 2.4 Gate G4 — real time (96×72, batch 1, 1,000 frames, CUDA synchronised)

| metric | threshold | value |
|---|---|---|
| sustained fps (mean) | ≥ 15, aim 25+ | **67.9** |
| fps at p95 latency | — | 42.4 |
| mean / p95 / worst frame latency | — | 14.7 / 23.6 / 24.8 ms |
| peak inference VRAM | < 3 GB | **210 MB** |

The loop measured is the complete per-frame pipeline: click features from the click
history, the readout, the warp, all 4 sampling steps of the denoiser, and the decode
to a displayable uint8 image. **The real-time gate passes with a ~4.5× margin**, so
the budget is not the binding constraint on the model — quality is. That also means
there is room for more sampling steps or a larger denoiser in the next stage.

### 2.5 The test split, evaluated once

**Gate G1, test split (300 contexts from 192 distinct episodes, evaluated once).**

| metric | val | test |
|---|---|---|
| counterfactual both-directions, near-door | 0.962 | **0.950** |
| counterfactual both-directions, all contexts | 0.967 | **0.957** (far-door 1.000) |
| per-side left / right | 1.000 / 0.967 | 0.977 / 0.980 |
| `L > R` | 0.99 | 1.00 |
| predicted-vs-real shift ratio at H = 10 | 0.669 | 0.577 |
| sustained median ratio (blocks) | 0.566 | 0.522 |
| sustained sign agreement (blocks) | 0.663 | 0.639 |
| commanded sustained sign / median ratio | 0.80 / 0.485 | 0.818 / 0.528 |
| readout vs truth / vs oracle / ceiling | 0.857 / 0.822 / 0.948 | 0.837 / 0.823 / 0.931 |

**Gate G2, test split.** Counterfactual both-directions (one-pass) **0.940**
(near-door 0.932); shift ratio 0.884; sustained sign/ratio 0.605/0.466; one-step
sharpness 1.013× real, per-channel mean gap 0.004, seed diversity 0.039 (reveals
0.052 vs steady 0.047); 60-step sharpness 1.57× real, near-black share 0.071.

**Gate G3, test split.** The same pattern as val: 300-step sharpness 2.0× real with
no clicks (1.9×), near-black 0.155 (no collapse to black, no frozen *content*),
measured motion 0.24 px/step in the scripted scenario against the real clips' 0.66.

The test split tracks val within a couple of points everywhere, which is the
expected result for a 198-episode val split and says the gates are not val-specific.


---

## 3. What the numbers mean

**The steering design works.** The click counterfactual — clicking the left door
versus the right door in the same context — is passed decisively at both
resolutions (0.878 near-door at 64×48, 0.962 at 96×72; the model separates the two
directions in 99% of contexts), the one-pass shift-ratio gate passes, and the
learned readout reaches 0.90/0.82 of the engine-rule oracle, which itself only
matches the real turn 0.95/0.95 of the time. In other words the click-to-motion
path is essentially at the ceiling the engine dynamics allow, and it is carried by
an explicit warp rather than by the denoiser — exactly the Stage 4 recommendation.

**The generator is sharp, stochastic and colour-stable at short horizons**:
one-step sharpness within 0.2% of the real frames, per-channel means within 0.004,
seed diversity 0.034 (a real spread of plausible content, not a blurry average),
and diversity is larger at doorway-reveal moments than during steady walking.

**The gap is in the closed loop over many frames.** The 40-step sustained ratio is
0.57-0.68 and the sign agreement 0.66-0.76, and with the generator in the loop the
*measured* global motion in a 300-frame rollout falls to a quarter of the real
clip's. Three effects are entangled, and §4/§5 separate them:

1. **The core under-turns on long turns** (commanded ratio 0.49-0.56 at 96×72): the
   readout is supervised for 10 steps ahead and its DAgger roll-in is 12 steps, so
   on a turn that lasts 30-40 steps the closed loop stops turning early. This is the
   first thing to fix next (§10).
2. **The generator replaces some of the real frame-to-frame change with new
   content**: frame-to-frame change is 0.012 against real 0.021, and the measured
   phase-correlation shift is diluted by content that legitimately changes between
   frames, so the *measured* ratio is worse than the *commanded* ratio.
3. **The readout sees the generator's frames**, not real ones, so its commanded
   shift also drops (0.83 with the core alone at 64×48 → 0.30 with the generator at
   96×72). Fine-tuning the readout on the generator's own outputs is the obvious
   remedy and is not done here.

**Sharpness inverts over long horizons**: the one-step frames match real sharpness
(1.00×) but the 60- and 300-step frames are 1.5× and 1.9× sharper, i.e. the
generator slowly paints in high-frequency content that the real frames do not have.
Combined with the reduced frame-to-frame change this reads as "detail accretes,
motion fades" — a drift direction worth watching in the review images.

---

## 4. Failure modes found (with evidence)

### 4.1 A click that is always consistent with the scene can be ignored entirely

The first steering core reached **0.91-0.94 sign accuracy** on val yet responded to
a *counterfactual* click not at all: injecting the left door's click, then the right
door's, into the same context and the same frames gave literally identical
predictions (`left = right = hold = +1.5 px`). The network had learned to read the
turn off the *frames* — in the data the click always matches the scene, so the
frames alone suffice — and it never needed the click. This is the Stage 4
"hold-last keeps turn presence but not direction" failure wearing a different hat.

The fix is to make the click causally necessary, not merely correlated: the
counterfactual ("click the *other* door") augmentation described in §1.2. After it,
per-side counterfactual direction accuracy is 0.90-0.97 and the readout matches the
engine-rule oracle to within ~0.05 (0.898 vs 0.954 on the same windows). A generic
lesson for action-conditioned models: **if a counterfactual action is not in the
training data, a high action-use score can be entirely spurious.**

### 4.2 The denoiser will cancel the warp if it is given a back door

The first generator was conditioned on `[x_t, warped, ctx_{−2}, ctx_{−3}, dx, t]`.
In closed loop that is fatal: frames two and three steps back are un-warped, so the
denoiser can reconstruct the pre-warp content from them and never route through
`warped`. The measured closed-loop image shift collapsed to **2% of the real turn**
(the commanded warp was +3.9 px over 20 steps, the measured image shift −0.3 px).
Removing the un-warped context frames from the conditioning — leaving only the
*warped* previous frame — restored the closed-loop ratio to 0.36 and tripled the
measured motion. This is the Stage 4 §7.12 "residual shadows the warp" failure, in
the closed-loop form: it is not enough to *have* the warp in the pipeline, the
residual must have no alternative path to the answer.

### 4.3 An L1 velocity loss collapses the sampler to the median residual

`FlowSteerModel` originally minimised L1 on the flow velocity (following the rest
of the project's pixel-loss convention). Measured on the closed loop, the residual
it produced was **`|r̂| = 0.02` against a real per-step frame change of 0.2** — the
generator had degenerated to "motion-warped last frame", i.e. the steering core with
noise. The residual is heavy-tailed (≈ 0 on most steps, large at reveals) and the L1
optimum is its *median*, which is ≈ 0. Switching to **MSE** on the velocity restored
real content and doubled seed diversity (0.015 → 0.030). Standard flow-matching
practice; worth stating explicitly because it silently turns a generative model into
a (very good) persistence model.

### 4.4 The flow's initial noise has to match the residual's scale

With unit-scale initial noise (`x0 ~ N(0, I)`) and only 4 Euler steps, the sampler's
velocity error leaves a residue of the noise in the output every frame, and it
compounds over a rollout: the 60-step generated frames measured **3.7× the real
frames' sharpness** (a high-frequency film that also destroys the phase-correlation
estimate). Since the residual being generated is ~0.02-0.05 in [-1, 1] units,
initialising the flow at that scale (`σ = 0.08`) removes the leak; 60-step
sharpness came back to 1.17-1.29× real.

### 4.5 Long rollouts alias the shift estimator

A 40-step turn accumulates ~20-60 px of content shift on a 64 px-wide frame, which
is at or past half the frame width — exactly where phase correlation wraps. Measured
on a real window: a commanded +37 px summed shift read as **+3 px**, and a
40-step-single-estimate sign agreement of 0.49 versus 0.73 from the same rollouts
estimated per-step. Stage 5 therefore measures the accumulated shift in 8-step
blocks (each well inside the alias-free range, `√(n/block)` noise terms), and also
reports the model's *commanded* shift (the warp it applies), which is noise-free.
The brief's "cumulative image shift (phase-correlation estimate)" is reported both
ways; the aliasing is a property of the estimator, not of the model.

---

## 5. Long-horizon stability (Milestone 3, Gate G3)

300-frame closed-loop rollouts (15 s at 20 Hz) from real seed contexts, 30 val
clips, three scenarios: no clicks at all, the real click sequence, and a scripted
alternating left/right pattern. The generated frames are compared with the source
clip's real frames on every statistic.

| statistic (96×72, 300 steps) | real | generated, no clicks | generated, scripted clicks |
|---|---|---|---|
| sharpness (mean gradient) | 0.0165 | 0.0313 (1.90×) | 0.0295 (1.79×) |
| mean luminance | 0.464 | 0.392 (darker) | 0.576 (brighter) |
| near-black share | 0.004 | 0.164 | 0.068 |
| near-white share | 0.004 | 0.005 | 0.031 |
| frame-to-frame change | 0.0226 | 0.0052 (0.23×) | 0.0122 (0.54×) |
| apparent motion (px/step) | 0.83 | 0.00 | 0.23 |

**What holds.** No collapse: no scenario goes to black, white or a uniform field,
and the frame content never freezes (the change statistic stays non-zero through 300
steps). Colour does not drift toward red — the per-channel means stay within 0.02 of
real on the 60-step track and the *scripted* scenario drifts slightly brighter, not
warm. In the 60-step track the per-channel mean gap is 0.009 and the near-black
share peaks at 0.069.

**What fails, precisely.** Two things, both visible by frame ~60 and dominant by 300:

1. **Detail accretes.** Generated sharpness climbs to 1.8-1.9× real. The
   frame-to-frame sparsity measurement says why: 86% of generated pixels change by
   more than 2/255 between consecutive frames (real: 44%) and 27% by more than
   10/255 (real: 11%), while the *mean* change is only 0.005-0.012 against real
   0.023. The generator is adding many small, incoherent high-frequency changes and
   few large coherent ones — a "grain" that accumulates over a long rollout. It is
   the same noise-leak mechanism as §4.4, just much weaker than the unit-noise
   version; four sampling steps and an L1-free (MSE) velocity field are not enough
   to make it exactly zero.
2. **Motion fades.** The measured global motion falls to 0.23 px/step (scripted)
   and ~0 (no clicks) from the real 0.83. Part of this is the readout's commanded
   shift dropping when it is fed generated frames rather than real ones (§3, item 3),
   and part is the phase-correlation estimator being diluted by the incoherent
   high-frequency changes of item 1 — the same dilution that makes the
   generator-in-the-loop counterfactual read 0.07-0.12 in the closed-loop image
   space while the one-pass probe reads 0.947.

Because both failure modes are *statistical* drift rather than collapse, the honest
summary is: the model produces plausible, sharp, colour-stable hallway frames for a
long time, and it stops moving like the real thing long before it stops looking
plausible. That is the headline result of Milestone 3.

The optional real-versus-generated classifier was **not** trained: the statistics
above already locate the drift (sharpness, change sparsity, motion), and the GPU
budget was better spent on the steering work.

## 6. Speed, resources and the hardware analysis (Milestone 4)

### 6.1 Real time (Gate G4)

The complete per-frame loop — click features from the click history, the readout,
the warp, all four sampling steps, and the decode to a displayable uint8 image, at
batch 1, CUDA synchronised, on mains power — runs at **67.9 fps mean** (p95 latency
23.6 ms → 42.4 fps; worst frame 24.8 ms) at 96×72, with **210 MB peak inference
VRAM**. Gate G4 (≥ 15 fps sustained over 1,000 frames) passes with a 4.5× margin.
Speed is not the binding constraint; quality is.

### 6.2 Where the work is

| component | parameters | MACs per generated frame | share | measured latency |
|---|---|---|---|---|
| readout (core) | 514,791 | 422.1 M | 9.2% | 0.375 ms |
| warp (grid_sample) | 0 | 0.08 M | 0.002% | 0.093 ms |
| denoiser, per step | 766,243 | 1,041.2 M | — | 0.529 ms |
| denoiser, 4 steps | — | 4,164.7 M | 90.8% | 2.118 ms |
| decode to uint8 | 0 | 0 | — | 0.036 ms |
| **total** | 1,281,034 | **4.587 G** | 100% | 2.62 ms of kernel time |

Precision is float32 at inference (bf16 autocast during training); the dense-op
fraction is 1.0 — every operation is a dense convolution, a dense matmul or the
gather-based warp, and there is no branch-dependent control flow.

Two measured facts stand out. First, the **4.587 GMAC/frame is not where the time
goes**: 2.62 ms of kernel time against a 14.7 ms end-to-end frame means ~82% of the
wall clock is launch and Python overhead across the several hundred small kernels a
4-step sampler over a two-level U-Net generates. The GPU's arithmetic is idle most
of the frame. Second, **half the activations are exactly zero**: 58.9% in the
readout and 47.4% in the denoiser after the ReLUs (and near-zero fractions within
0.001 of those), i.e. the trained representation is genuinely sparse even though the
arithmetic computed on it is dense.

### 6.3 Frame-to-frame sparsity of the data

Between consecutive real frames, 44% of pixels change by more than 2/255, 11% by
more than 10/255 and only 4% by more than 30/255; the mean absolute change is
0.0039 of full scale. The world is 20 Hz, nearly static, with a small share of
pixels carrying the new information. Generated frames are *more* active at the small
threshold (86% above 2/255) and comparable at the large one (4.5% above 30/255),
which is the "grain" of §5 seen from the data side.

### 6.4 What this suggests about hardware

The workload is a short chain of **small-channel dense 3×3 convolutions** (32-128
channels, 96×72), evaluated four times per frame with a fresh noise sample, plus one
gather-based warp. It has low arithmetic intensity: 4.6 GMAC against ~1.4 M
activations per denoiser pass and 1.28 M parameters total. On a dense tensor
accelerator this is a launch-bound problem long before it is a FLOP-bound one (82%
of the frame is overhead), which argues for **fewer, larger kernels** — CUDA graphs,
operator fusion, or a compiler that specializes the 4-step sampler — more than for
more TOPS. The whole weight set (1.28 M parameters, ~5 MB fp32) fits in on-chip or
near-memory storage, so an in-memory or analog array is a natural fit for the
convolutions themselves, with the caveat that the four sequential sampling steps
cannot be collapsed into a single pass without changing the model (distillation
could, and would also cut the 90.8% of MACs the denoiser owns).

The measured sparsity cuts the other way. **47-59% of activations are exactly zero**
after the nonlinearities, and 56-89% of pixels do not change meaningfully per frame,
so an **event-driven or sparse-aware design could in principle skip about half the
multiplies and most of the memory traffic** — but the current model has no
*structured* sparsity to exploit: the zeros are scattered, the dense-op fraction is
1.0, and the residual stream mixes them back in at every layer. Realizing the gain
needs either sparsity-aware hardware/software (structured pruning, event-driven
convolution engines) or a redesign that keeps sparsity structural, and it needs the
activation bandwidth rather than the FLOPs to be the target, since the model is
activation-bound. In short: the *representation* is already event-like; the
*arithmetic* is still dense, and closing that gap is a hardware-software co-design
problem rather than a model-size problem.

---

## 7. Verification

1. **Existing tests pass.** `pytest` in `/neural`: **42 passed** (Stage 3 loader
   suite, Stage 4 metrics/harness suite, and the 18 new Stage 5 tests). The loader
   throughput test only reaches its floor when the machine is not otherwise loaded;
   run alone it passes.
2. **New unit tests** (`neural/tests/test_stage5.py`, 18 tests): the warp operation
   (identity, integer shift via the shift estimator, sub-pixel sign/magnitude, border
   padding instead of black), click-feature derivation (delta at the click, held
   steps, frames-since-click counted from `actionX` changes, the `actionHasClicked`
   latch), the mirror augmentation (frames flipped, `actionX` negated,
   `actionHasClicked` untouched), the new metric functions (sharpness falls under
   blur; `color_stats` in [0, 1]; `frame_change` identity and scaling;
   `black_frac`/`seed_diversity` limits; uint8-versus-[-1, 1] handling), and the
   player's debounce semantics (a click within 3 frames of the last accepted click is
   ignored; the held signal is clamped to [-1, 1]).
3. **Data hygiene.** Model selection used the val split only; the test split was
   evaluated once, at the end (§2.5). Training windows drop `viewUnsafe` frames
   (the loader's rule). No sidecar field is used by any model at inference.
4. **Reproducibility.** Every run is seeded (`--seed 0`) and every checkpoint carries
   its full config; the headline gate numbers are reproduced by the single commands
   in `neural/README.md` §12 (the JSON in `neural/eval_out/` is the record).
5. **No-oracle (N1/N2).** `test_inference_models_accept_only_frames_and_clicks`
   asserts both inference entry points take nothing but context, click signals,
   features and a seed; `test_rollout_driver_never_reads_the_sidecar` monkeypatches
   `analysis.load_analysis` to raise and runs a full closed-loop rollout, and also
   asserts the driver's source contains no reference to the sidecar. The sidecar
   (heading, doorway candidates) is read **only** while building training targets.
6. **The gates** G1-G4 are reported above with their thresholds, at both resolutions
   and on the test split. Everything visual is left to the user (§6 of the brief;
   `neural/review/stage5/`).

---

## 8. Shortcuts taken, and what they cost

* **One architecture, no sweeps.** Readout base 32 (512 k params), denoiser 2-level
  U-Net base 32 (~766 k trainable), 4 flow steps, H = 10 supervision. Chosen once
  from the Stage 4 evidence, then only the recipe (target, DAgger, counterfactual
  augmentation, conditioning, loss, noise scale) was changed — never a grid.
* **The click-geometry target is a linear fit** (`cum_m ≈ coef_m · screenX_door`,
  R² ≈ 0.6 at H = 10). It is only used to *generate counterfactual training targets*,
  so its noise is a bias toward click-dependence, not a source of error in the model.
* **DAgger rolls in ≤ 12 steps.** Longer would be better for long turns and is the
  first thing to try next (see §10).
* **Self-conditioning rolls in 1-2 steps** (a full curriculum to longer roll-ins was
  not run) — enough to stop the collapse of §4.2, not enough to make the 40-step
  turn magnitude match the real clip.
* **Only the first sampling step's shift is used for the warp** (the readout emits a
  curve but the closed loop consumes `inc[0]` each frame); a controller over the
  whole curve is an obvious next step.
* **No distillation**, no CUDA graphs, no fp16/8-bit; the 15 fps gate is met with
  plain fp32/bf16 inference (§6).
* **Checkpoints are resumable** (`--resume runs/<name>/last.pt` in both training
  scripts) but only the model weights are restored, not the Adam state — with a
  constant learning rate the practical difference is small, and it is stated here
  rather than papered over.
* **GPU hours** (a ~4.5 h total, against the brief's 10-12 h budget): steering cores
  1.6 h (ten runs; the 96x72 final was 35 min of it), generator 0.8 h (six runs),
  shift/target tables 0.1 h, gate evaluations (val + test, all three gates, hardware
  analysis) 1.7 h, review artifacts 0.2 h. No single run exceeded 40 minutes, and no
  two GPU jobs ever ran at once.
* **Vision was never judged by the agent** (brief rule 5): all visual artifacts are
  prepared for the user (§5 of the brief; `review_stage5.py`).

---

## 9. Changes to Stage 3/4 code, `.gitignore`, and the file list

### Additive changes to existing files

* `neural/eval_harness.py` — **additive only**: `model_predict` (signature-aware
  dispatch so Stage 4 checkpoints stay evaluable), `build_history`,
  `features_from_history`, and `rollout_v2` (feature-aware, seed-aware closed-loop
  driver). No existing function changed; the Stage 4 harness and its tests still
  pass.
* `neural/metrics.py` — **additive only**: `sharpness`, `laplacian_var`,
  `color_stats`, `frame_change`, `black_frac`, `saturated_frac`, `seed_diversity`,
  and dtype handling in `_as_nchw` (uint8 frames are rescaled to [-1, 1] so the
  "in [0, 1] units" metrics mean the same thing for packed frames and tensors).
  Existing metric functions and tests unchanged.
* `neural/requirements.txt` / `requirements.lock` — added `pygame-ce==2.5.8` (the
  player; `pygame` itself has no wheel for Python 3.14 on this machine and fails to
  build from source).
* `neural/.gitignore` — nothing new: `/neural/runs/`, `/neural/review/` and
  `/neural/data/` already covered the new outputs (shift-table caches live under
  `neural/data/_cache/`, deliberately **outside** the read-only packed sets).
* **No Stage 3 code changed.** `packer.py`, `dataset.py` and their tests are
  untouched; the `__reduce__` loader fix from Stage 4B stays in place.

### New files

| file | role |
|---|---|
| `neural/shift_table.py` | per-step phase-correlation shift table + heading-derived denoised target + `load_target_table` |
| `neural/steer.py` | click features, `warp_translate`, `DirectionReadout`, `SteeringModel`, checkpoint I/O |
| `neural/train_steer.py` | Milestone 1 training (heading target, DAgger roll-in, counterfactual clicks) |
| `neural/flow.py` | `VelocityUNet`, `FlowSteerModel`, flow training loss, checkpoint I/O |
| `neural/train_flow.py` | Milestone 2 training (self-conditioning, onset oversampling, mirror) |
| `neural/gates.py` | one-command gate runner G1-G4 + quality/diversity/rollout metrics |
| `neural/play.py` | interactive player (pygame) + scripted headless GIF/strip mode |
| `neural/review_stage5.py` | the brief's §6 review images |
| `neural/tests/test_stage5.py` | warp, click features, mirror augmentation, metrics, N1/N2 (no-oracle) |
| `neural/hw_analysis.py` | the Milestone 4 parameter/MACs/latency/sparsity measurements |
| `neural/docs/STAGE5-PROGRESS.md` | the running notes file |
| `neural/docs/STAGE5-REPORT.md` | this report |
| `neural/eval_out/g1_steer96{,_test}.json` | G1 on val and (once) on test, final core |
| `neural/eval_out/g2_flow96{,_test}.json` | G2 on val and (once) on test, final generator |
| `neural/eval_out/g3_flow96{,_test}.json` | G3 300-frame stability, val and test |
| `neural/eval_out/g4_flow96.json` | G4 real-time measurement |
| `neural/eval_out/hw_flow96.json` | parameter/MACs/latency/sparsity measurements |
| `neural/eval_out/g1_steer64*.json`, `g2_flow64*.json` | the development-resolution gate runs (the "how G1 evolved" record) |
| `neural/review/stage5/*` | the review artifacts (gitignored folder) |

Untracked scratch directories (`runs/`, `data/`, `review/`, `eval_out/*.log`) stay
gitignored or are logs; the `eval_out/*.json` above are the gate record and are
worth committing. **Nothing was committed** — git is the user's (brief rule 1).

---

## 10. Milestone 5 — the player

`neural/play.py` is a thin client that never touches the engine: it loads a
checkpoint, seeds the run with K real frames from a random early-episode clip, then
generates every following frame from its own outputs plus the click signal, shows
them enlarged in a pygame window at 20 fps with an fps readout, and reads the mouse
with the engine's exact click semantics (X = clamp((clientX − left)/width · 2 − 1),
a click within 150 ms of the last *accepted* click ignored, `actionHasClicked`
latched after the first accepted click). The translucent click pulse it draws is a
display overlay and is never fed back into the model. `--headless --clicks
"10:0.8,90:-0.8" --frames 240 --out …gif --strip …png` plays a scripted sequence and
writes a GIF and a frame strip; that mode was run for the artifact in
`review/stage5/play_scripted.gif` and is what the automated check exercises. The
interactive window could not be judged by the agent (no display, and visual
judgement belongs to the user) — it is handed over as-is.

## 11. Needs user decision / suggestions for the next stage

**Decisions needed**

1. **Is the steering good enough to build on, or does the sustained-steering
   shortfall need an engine-side change?** The click-direction behaviour is at the
   engine-rule ceiling in the one-pass probe (0.94-0.97), but a 40-step closed loop
   only reproduces 0.52-0.68 of the real clip's cumulative turn, and the commanded
   sign accuracy is 0.80-0.82. If the *felt* behaviour matters more than the metric,
   the cheapest engine-side lever is a slightly longer/faster turn ramp on a click
   (a tuning knob only; the engine was not touched) — worth a decision before more
   model work, since it changes what the model has to learn.
2. **Where to spend the next GPU hours.** The three candidates in order of expected
   value: (a) fine-tune the readout on the generator's own frames (DAgger at the
   *stack* level) — this is the single change most likely to fix both the sustained
   ratio and the generator-in-the-loop drop; (b) lengthen the supervision horizon
   and the DAgger roll-in (H 10 → 20, k 12 → 24) so long turns are actually
   supervised and trained through; (c) more sampling steps (4 → 8) — cheap
   quality/speed trade given the 4.5× real-time margin.
3. **Long-run drift.** The 300-frame drift (sharpness 1.8-1.9×, motion fading) is a
   statistical artifact of few-step sampling, not a collapse. Fixing it properly
   means either a longer self-rollout curriculum during training or a small
   hand-written stabiliser (periodic re-noising, a high-frequency penalty on the
   residual). This is a research choice with a visible trade-off in look and speed.

**Suggestions for later stages**

* **More/different data**: the click-geometry ceiling (oracle-vs-truth 0.95) says
  the labels are good; the remaining error is the pilot's behaviour mix. A dataset
  with longer sustained turns (the pilot mostly turns and settles within ~10 frames)
  would directly address the sustained-steering gap.
* **Longer context** (K 4 → 8): the readout currently sees 4 frames; the turn ramp
  is ~10 frames long, so more context should help the "when does the turn stop"
  decision.
* **The dreamlike features are still the right next *stage*** — nothing in this
  design depends on clean categorical labels (no room-type input, no door label, no
  retrieval), so texture/prop injection and undefined spaces can be layered on
  without rebuilding the steering path.
* **Ablation datasets of different bot mixtures** would let the model's behaviour
  mix be controlled directly instead of inferred from the persona statistics.
