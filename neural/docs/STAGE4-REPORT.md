# Stage 4 — Baseline Action-Conditioned Next-Frame Model and Evaluation Harness

This stage trains the first neural model on the pilot data and, more importantly,
builds the evaluation harness every later model (including the eventual diffusion
model) will be judged with. It is written so a student new to the project can follow
it and reuse it in a class write-up.

---

## 1. What we built and why

The project's end goal is a neural network that *acts as a game engine*: given a few
recent frames and the player's click signal, it generates the next frame, over and
over, in real time (15+ fps), producing an endless backrooms-style maze. The likely
final model is an action-conditioned diffusion model. Before committing to that
machinery we want, cheaply, to prove the data/labels/loader/training loop are correct
end to end, to find out whether the action signal is actually usable, and to
characterize what a simple deterministic predictor can and cannot do.

So the baseline is deliberately small and deterministic. It is **not** intended to
produce good-looking generations — its expected failure (blurry averages where new
content is revealed) is exactly the evidence that motivates a generative model.

### The model

A 2-level U-Net with `GroupNorm`, encoder widths `32 / 64 / 128`, and a
nearest-upsample + convolution decoder. It predicts a **residual** (the change from
the last context frame), added back to the last frame in float32, so consecutive
nearly-identical frames are easy. Input is `K = 4` context frames plus their
`K = 4` actions; output is one predicted next frame.

Two action-injection schemes were built:

* **channels** (the default / adopted baseline): the `K` actions
  (`[actionX, actionHasClicked]` float32 each) are broadcast as `K*2 = 8` extra
  image-shaped channels alongside the `K*3 = 12` color channels, giving 20 input
  channels to the U-Net. Simplest possible scheme.
* **film**: an MLP maps the flattened `K*2` actions to FiLM scale/shift parameters at
  the bottleneck and two decoder levels.

`model.py` exposes a single interface used by the harness and by every future model:

```python
def predict(context, action) -> next_frame   # context [B,K,3,H,W], action [B,K,2]
```

**Parameter count:** 708,323 (channels), 735,715 (film).

### Why a residual, and the action alignment rule

The loader represents each action as `[actionX, actionHasClicked]`, where `actionX in
[-1, 1]` is the held horizontal click position and `actionHasClicked` is 0/1. The
Stage-2 alignment rule — `action[i]` drives frame `i → i+1`, i.e. the control signal
that produces frame `i+1` is `controlSignalX[i+1]` — is what the ablation in §3.3
tests empirically.

---

## 2. Training setup

* **Data**: pilot dataset, 149 episodes / 88,047 frames, packed at 64×48 (fast
  iteration) and 96×72. Train 119 episodes (~70,600 frames), val 15, test 15, at
  20 Hz. `viewUnsafe` frames (357, 0.41%) are dropped via `require_safe=True` for
  training and evaluation. `actionHasClicked == false` on 9.8% of frames.
* **Recipe** (`train.py`, config-driven): K=4, H=1 (one step), base=32, batch 64,
  Adam lr 1e-3, weight decay 0, L1 loss, 6,000 steps (~5 epochs over the 69,453
  training windows), seed 0, bf16 autocast (residual add kept in fp32),
  `num_workers=2` + pinned memory.
* **Checkpoint selection**: lowest validation L1 (one-step, teacher-forced), never
  touching the test split.
* **Speed observed** (laptop RTX 5070, 8 GB, shared with the display): ~49 steps/s at
  64×48, ~18 steps/s at 96×72. A full 64×48 run is ~2 minutes; 96×72 is ~6 minutes.

| run | resolution | conditioning | best val L1 | params |
|---|---|---|---|---|
| baseline_64 | 64×48 | channels | 0.01213 | 708,323 |
| baseline_96 | 96×72 | channels | 0.01241 | 708,323 |
| baseline_64_film | 64×48 | film | 0.01175 | 735,715 |

A separate, investigation-only model — `fut10` / `fut10_mirror`, trained by
`train_future.py` with an H=10 teacher-forced lookahead objective — is the one that
proved the action usable; its (H-step, not one-step) val L1 and ablation are reported
in §7.7, and it is *not* the adopted one-step baseline.

The train/val gap is tiny (final train L1 0.01247 vs val 0.01269 at 64×48) — the
pilot's consecutive frames are so redundant that there is almost no generalization
gap; the model is under-parameterized relative to the one-step task, not overfit.

---

## 3. Acceptance checks — results

### 3.1 Sanity overfit (check 1)

Training on 64 windows only (`--limit 64`) drove training L1 to 0.0057 within ~1
minute (2,000 steps), while val L1 stayed at 0.048. The model, loss and training loop
can memorize. ✓

### 3.2 Beats the reference (check 2)

One-step, teacher-forced metrics on the val split (8,843 windows), model vs the
persistence reference (repeat the last context frame):

| metric | persistence | model (baseline_64) |
|---|---|---|
| L1 | 0.01973 | **0.00639** |
| MSE | 0.00298 | 0.000344 |
| PSNR (dB) | 25.49 | **35.01** |
| SSIM | 0.9281 | 0.9866 |

The model beats persistence by ~3× on L1. ✓

The same recipe at 96×72 val (8,843 windows) gives model one-step L1 **0.00655**
(PSNR 34.27) vs persistence **0.02036** — again a ~3.1× margin. So the beat is
resolution-independent. (The one-time test-split number is in §3.7.)

### 3.3 Alignment / action-use ablation (check 3)

The schema's off-by-one rule is tested by scoring the same val windows with (i) true
actions, (ii) actions zeroed, (iii) actions shuffled across windows, (iv) actions
shifted one frame earlier, and (v) one frame later. If the rule is right *and* the
model uses the action, true should score best.

**One-step L1, overall (8,843 windows):**

| variant | L1 |
|---|---|
| true | 0.00639 |
| zeroed | 0.00652 |
| shuffled | 0.00643 |
| earlier | 0.00640 |
| later | 0.00640 |

**On the action-relevant subset** (2,647 windows, 30%, where the target-step heading
changes by ≥ 0.5°):

| variant | L1 |
|---|---|
| true | 0.01074 |
| zeroed | 0.01102 |
| shuffled | 0.01081 |
| earlier | 0.01075 |
| later | 0.01077 |

All variants are statistically indistinguishable. **The model ignores the action.**
This is a property of the one-step objective — the action's next-frame effect is
sub-pixel — and is explained, and resolved with an H-step lookahead, in §7 (§7.7). The
result is *not* evidence of a label bug: the one-frame-shifted actions tie rather than
beat the true alignment, and the ground-truth heading correlates strongly with the
observed image shift (r = −0.87, see §3.5), which is exactly what the alignment rule
predicts.

### 3.4 Rollout error vs horizon (check 4)

40-step autoregressive rollouts (model feeds its own predictions, true actions
supplied), averaged over 40 val clips:

* Model L1 grows roughly linearly from 0.0049 (step 1) to 0.235 (step 40) — error
  compounds as geometry melts and blurs.
* Persistence L1 stays near 0.015–0.029 the whole way.
* **The model beats persistence only until about step 3**, then falls behind, because
  compounding prediction error is worse than freezing on the last real frame.

### 3.5 Action-following (check 5a)

Horizontal image shift between successive frames is estimated by phase correlation and
compared with the real frames' shift and the sidecar heading. Over 2,000 val windows:

| quantity | value |
|---|---|
| mean |shift|, real frames | 1.26 px |
| mean |shift|, model frames | 0.77 px |
| mean |shift|, model with zeroed actions | 0.75 px |
| corr(model shift, real shift) | 0.758 |
| corr(model shift, heading change) | −0.795 |
| corr(real shift, heading change) | −0.868 |
| sign agreement, model vs real | 0.664 |
| sign agreement, zeroed vs real | 0.661 |

The model's predicted motion tracks the real heading direction (negative correlation,
matching the "image content moves opposite to the turn" convention), but it
**under-predicts the magnitude** and does it **no better when given the true action
than when the action is zeroed**. The model learns the easy steady-forward motion from
the frames alone; it does not read the action.

### 3.6 Counterfactual direction test (check 5b)

Contexts with reachable doors on both sides of screen center are rolled 15 steps under
"click the left door", "click the right door", and "no click". 50 such contexts:

| quantity | value |
|---|---|
| mean shift, click-left | −1.05 px |
| mean shift, click-right | −0.84 px |
| mean shift, no-click | −0.69 px |
| share: left-click turns view left | 0.12 |
| share: right-click turns view right | 0.60 |
| share: **both** directions correct | 0.06 |

The model shows a **rightward bias**: it tends to turn the view right regardless of
which door is clicked (12% vs 60% correct by side, 6% both-correct, i.e. below the
25% a coin-flip pair would give). The bias traces to the data itself — see §7.

### 3.7 Test split, evaluated once (check 7)

The test split was touched once, at the end, with the adopted 64×48 baseline
(`runs/baseline_64/best.pt`), 8,313 windows:

| metric | persistence | model (true actions) |
|---|---|---|
| one-step L1, overall | 0.0207 | **0.0068** |
| one-step L1, action-relevant | 0.0326 | 0.0114 |
| PSNR, overall (dB) | — | 34.54 |

The test-split ablation and action-following reproduce the val findings exactly:
true ≈ zeroed ≈ shuffled ≈ shifted (the model still ignores the action), the model
beats persistence in rollouts only until ~step 3, and the counterfactual
"both-directions-correct" rate is 0.04 (below the 0.25 chance rate). The model's
one-step quality generalizes; the action-use failure does too.

---

## 4. The evaluation harness (the long-lived part)

`eval_harness.py` evaluates **any** model with the `predict(context, action)`
interface plus a built-in autoregressive rollout driver. It runs all five parts of
§2.4 and writes one JSON:

```bash
.venv/bin/python eval_harness.py --packed data/pilot_v1_64x48 \
    --checkpoint runs/baseline_64/best.pt --split val --out eval_out/baseline_64_val.json
```

* **(a) one-step metrics** — L1 / MSE / PSNR / SSIM (lightweight, no heavy deps),
  overall and on the action-relevant subset (`|target-step heading change| ≥ 0.5°`).
* **(b) action-use ablation** — true / zeroed / shuffled / shifted ±1 frame, no
  retraining.
* **(c) rollouts** — autoregressive ~40-step rollouts, model vs persistence.
* **(d) action-following** — phase-correlation shift vs real frames and heading.
* **(e) counterfactual direction** — left/right/no-click on two-door contexts.

`metrics.py` holds the L1/MSE/PSNR/SSIM and phase-correlation shift estimators;
`analysis.py` wraps the sidecar (heading, pose, angular velocity, doors). The harness
is documented in `README.md` §9 and covered by `tests/test_harness.py` and
`tests/test_metrics.py`.

---

## 5. Speed and resource measurements

`bench.py` measures the model on this GPU (RTX 5070, 8 GB). Numbers are for the
96×72 model (identical architecture to 64×48, so FLOPs/latency are the 96×72 ones):

| quantity | value |
|---|---|
| parameters | 708,323 |
| FLOPs / generated frame (batch 1) | 2.13 GFLOPs |
| numeric precision | bf16 autocast (residual add in fp32) |
| peak GPU memory, training step (batch 64) | 1315 MB |
| peak GPU memory, batch-1 inference | 43.6 MB |
| batch-1 latency | 0.589 ms |
| batch-1 frames/sec | 1699 fps |
| autoregressive rollout-loop fps | 1143 fps |
| rollout fps vs 15 fps target | **76×** |

A single forward pass is far faster than the 15 fps target; the 76× headroom is what
a multi-denoising-step diffusion model would consume.

---

## 6. Learning curve

Trained the final recipe on 10%, 30% and 100% of the training windows (same seed,
same 6,000-step budget) and measured validation L1:

| train fraction | windows | best val L1 |
|---|---|---|
| 10% | 6,945 | 0.01220 |
| 30% | 20,836 | 0.01228 |
| 100% | 69,453 | 0.01213 |

**The curve is flat.** 10% of the data already reaches the same validation L1 as
100% (0.01220 vs 0.01213). The pilot's consecutive frames are so redundant that the
one-step prediction task is **not data-limited** — it is capacity- and
objective-limited. This says two things for the project: (1) generating *more* pilot
data will not improve a deterministic one-step predictor; (2) the action-use problem
documented in §7 is a signal problem, not a data-volume problem, so bulk data
generation is not the fix either. What bulk data *would* buy is coverage of rare
events (doorway reveals, turn variety, the left/right balance) that matter for a
generative model even if they don't move the average one-step loss.

---

## 7. Why action conditioning is hard here — and the objective that fixes it

The one-step ablation (§3.3) shows the model ignores the action. This is not a code
bug — the conditioning *pathway* works — it is a property of the data at 20 Hz. We ran
an investigation to prove it (steps 0–4 below), and it converged on a constructive
answer: the action *is* usable, but only when the objective gives it a pixel-level
effect to predict — an H-step lookahead (§7.7). The one-step and autoregressive
multi-step objectives fail for the reasons in §7.1–§7.4.

### 7.1 Step 0 — how big is the action's effect, really? (`onset_effect.py`)

Using phase correlation on the packed 64×48 frames at turn onsets (a quiet stretch
followed by a sustained turn), the **cumulative** horizontal image shift the action
eventually produces is:

| horizon (steps) | mean \|shift\| at turn onset | steady (no turn) |
|---|---|---|
| 1 | **0.027 px** | 0.006 px |
| 5 | 5.4 px | — |
| 10 | 10.7 px | — |
| 15 | 9.8 px | — |

At the horizon a one-step model is trained on — the *next* frame — a click moves the
image by **0.027 px**, far below pixel noise. The effect only accumulates to ~5 px by
step 5 and ~10 px by step 10 (the median time to align with a door). A one-step L1
loss therefore gives the model *no* gradient signal to read the action: the pixel
error it would avoid is smaller than the texture noise it already fits. The context
frames also already contain the turn in progress, so the frames alone carry most of
the predictable motion.

### 7.2 Step 1 — is the conditioning pathway even working? (`pathway_sanity.py`)

Trained the **same architecture** on a synthetic task where the action obviously
matters: the target is the last context frame shifted horizontally by
`actionX × 8 px`, with K identical context frames (so the frames alone cannot predict
the shift). The model drives L1 to **0.04**, proving the action-injection code and the
training loop can use the action when the signal is large. The problem is the
signal-to-noise ratio in the real data, not the code.

### 7.3 Step 2/3 — multi-step objectives and onset oversampling (`train_multistep.py`)

To reward the action where it actually accumulates, we trained autoregressive
multi-step models: over an H-step horizon the model feeds its own (detached)
predictions and the true actions, and the L1 is summed over the horizon, so ignoring
the action is penalized where the turn accumulates. Turn-onset windows (context shows
no turn yet, a turn develops in the horizon) were oversampled to 50% of each batch —
exactly where the action is the only source of the turn. Then we ablated at the
horizon (true vs zeroed vs shuffled vs shifted on onset windows, ~10 steps):

| model | true | zeroed | shuffled | mean(1..10) true vs zeroed |
|---|---|---|---|---|
| baseline_64 (one-step) | 0.0443 | 0.0458 | 0.0448 | 0.0443 vs 0.0458 |
| ms8 (multi-step H=8, detached) | 0.0380 | 0.0392 | 0.0386 | 0.0380 vs 0.0392 |

The multi-step training **improves rollouts** (the ms8 model holds together longer at
the horizon than the one-step model — 0.038 vs 0.044 mean L1), but the true-vs-zeroed
gap is ~1–3% and true ≈ shuffled ≈ shifted. It still does not make the model clearly
action-responsive.

### 7.4 Step 4 — full backpropagation through time (`ms10_btp`)

The detached multi-step training above cuts off cross-step credit assignment: because
each step's fed-back prediction is detached, the model never learns "use the action
now so my predictions three steps later are right". We therefore tried non-detached
training (full gradient flow through the 10-step rollout, gradient clipping for
stability):

| variant | L1@step1 | L1@step5 | L1@step10 | mean(1..10) |
|---|---|---|---|---|
| true | 0.0155 | 0.0430 | 0.0894 | 0.0501 |
| zeroed | 0.0157 | 0.0449 | 0.0929 | 0.0521 |
| shuffled | 0.0155 | 0.0432 | 0.0900 | 0.0503 |

Even with full credit assignment through the horizon, the true actions beat zeroed by
only ~4% (0.0501 vs 0.0521) and tie shuffled/shifted. For **rollout-based** objectives
the conclusion is robust: the per-step action effect is buried in pixel noise and the
cumulative effect is dominated by the model's own compounding rollout error. But the
deeper fix — drop the rollout and predict the frame H steps ahead in a single forward
pass — works; see §7.7.

### 7.5 The directional bias

The counterfactual's rightward bias (§3.6) has a data-side cause: heading changes in
the pilot are **not balanced** — 42.5% positive vs 57.5% negative. A model that
ignores the action and learns the average turn direction will drift right. This is a
property of the pilot (which corridors and doors the recorder happened to favor), not
a label bug, and it is something the final model will need to be robust to (e.g. via
mirror augmentation or a balanced dataset).

### 7.6 What did and did not fix it

| attempt | result |
|---|---|
| FiLM conditioning (baseline_64_film) | slightly better val L1 (0.01175), still ignores action |
| multi-step H=8, onset oversampling (ms8) | better rollouts, not action-responsive |
| non-detached BPTT H=10 (ms10_btp) | not action-responsive |
| synthetic shift task (pathway sanity) | **works** — proves the pathway, not the data, is the issue |
| **H-step lookahead, teacher-forced (fut10)** | **action-responsive** — see §7.7 |

The failure is a property of the *objective*, not the architecture. Once the model is
asked to predict the frame H=10 steps ahead — where a click has accumulated ~10 px of
motion — the same FiLM U-Net reads the action decisively. The adopted one-step
baseline is unchanged (it still ignores the action, as the brief's harness measures
it), but the action-use question the stage set out to answer has a constructive
resolution in §7.7.

### 7.7 The fix that worked — H-step teacher-forced lookahead (`train_future.py`)

Steps 0–4 share one assumption: the model predicts the *next* frame (directly, or
iterated over the horizon). Step 0 shows that at 20 Hz a click moves the next frame by
0.027 px, so a one-step objective can never reward reading the action; and the
multi-step variants (§7.3–§7.4) fail because the model's own compounding rollout error
dominates the accumulated turn.

`train_future.py` removes both problems at once. The model is trained to predict the
frame **H steps ahead in a single forward pass**, from the K context frames plus the
**K+H action sequence** (the lookahead actions say which way the accumulated turn will
go). FiLM conditions on the full variable-length action vector. Turn-onset windows
(quiet context, a turn develops in the horizon) are oversampled to 50% of each batch,
so the action is the only source of the turn in the loss. H=10 accumulates ~10 px of
motion, well above the pixel-noise floor.

At H=10 (6,000 steps, the same budget as the baseline), the ablation on onset val
windows (n=300) is decisive:

| future actions | H-step L1 | vs true |
|---|---|---|
| true | 0.1013 | — |
| zeroed | 0.1198 | +18% |
| shuffled | 0.1394 | +38% |

Shuffling the lookahead actions makes the prediction 38% worse — the model is
unambiguously using the *direction* of the action sequence, not just "something was
clicked". This is the same FiLM U-Net that ignored the action under the one-step and
rollout objectives (§7.3–§7.4); the only change is the objective. Mirror augmentation
(flip frames + negate `actionX`) gives the same result (true 0.0980 vs zeroed 0.1161
vs shuffled 0.1413) while removing the gross directional bias of §3.6.

**Counterfactual, done right (`probe_click.py`).** The §3.6 probe held one door
position constant for the whole horizon — not how a click works — and measured
sub-pixel noise on a biased, near-duplicate set of contexts.
Redesigned: contexts drawn from 15 distinct val episodes, mid-episode (never the start
of an episode), with two reachable doors on opposite sides that are both close enough
to see (`screenDistance` < 2 world units; 30 such contexts), and a proper "no new
click" baseline (hold the true last action). Predicted t+H frames under
"click left / click right / hold", horizontal shift by phase correlation:

| probe variant | left shift | right shift | hold shift | both-correct |
|---|---|---|---|---|
| click at final context step | +0.72 px | −0.78 px | −0.18 px | 0.30 |
| held across context+future (no change) | −0.15 px | −0.14 px | −0.17 px | 0.27 |

The model turns the correct way **only when the click is a change event** at the
context/future boundary; holding the same X throughout (no change) produces no
directional response. So fut10 keys on the action *changing*, not on the absolute held
position. This is a **weak partial response**, not a clean pass: 0.30 both-correct is
barely above the 0.25 coin-flip for an ordered pair, and the predicted click shift
(±0.7–0.8 px) is **under a tenth of the real accumulated turn** (~10 px at H=10) — a
predicted-vs-real shift ratio of ≈0.08. (Far-door contexts — where the correct turn is
small — reach 0.62 both-correct; n=30 is small, so the ±0.7 px mean shifts are the
robust signal, the per-context accuracies are noisy.) **Note:** the val split has only
15 episodes, so every counterfactual statistic is limited to 15 distinct episodes; n is
a context count, not an episode count.

### 7.8 Deployment-realistic conditioning loses the direction (`train_hold.py`)

fut10 conditions on the **true future actions**, which a real game does not have: at
play time only the held action `a_{t-1}` is known, and it persists until the next
click. `train_hold.py` retrains with **hold-last** future actions (`a_{K-1}` repeated H
times) instead of the true future, under two onset definitions:

| model | future | onset | held-click ablation: true / zeroed / shuffled |
|---|---|---|---|
| fut10_mirror | true | quiet-context | 0.098 / 0.116 / **0.141** (+44% shuffled) |
| hold10 | hold-last | click-at-final-step | 0.157 / **0.212** / 0.202 |
| hold10_quiet | hold-last | quiet-context | 0.105 / 0.117 / 0.112 |

Hold-last collapses the ablation to *turn presence*: zeroing the held click is costly
(the model predicts no turn where a turn exists), but shuffling it is barely worse
than zeroing (0.202 vs 0.212, and 0.112 vs 0.117) — the model does **not** read the
click's *direction*. hold10's click counterfactual is flat (both-correct 0.00), and
on its own training windows the sign of its predicted turn matches the ground-truth
turn below chance (0.26).

Why: the direction signal lives in the **change events** in the true future action
sequence. 31% of onset windows contain a mid-horizon click (the future `actionX`
changes), and "`actionX` changes → the view turns toward the new X" is the one clean,
large-gradient signal the pixel loss can follow. Hold-last deletes every future change,
leaving only the single click at `a_{K-1}` — not enough to learn direction at H=10.
(Compounding it, the click-at-final-step turns are ~10× smaller than the quiet-context
turns: ~1.1 px vs ~10 px accumulated over H=10.)

### 7.9 Left/right asymmetry — world vs pilot

Mirror augmentation was used throughout. Checked against the engine: the **world
generation is essentially symmetric** — reachable door `screenX` is balanced (mean
+0.0065, 48.5% left / 51.5% right, mean 0.81 left / 0.86 right doors per frame). The
asymmetry is in the **pilot**: signed heading changes are skewed negative (≈44%
negative / 42.5% positive over all steps; 57.5/42.5 on turn windows). So the
directional bias the model inherits (§7.5) is the recorder's behaviour, not an
engine-side world-generation asymmetry, and mirror augmentation targets the right
thing.

**Bottom line.** The action *is* usable, and the principle is: supervise at a horizon
where the action has pixel-scale effect (H≈10), condition on the action sequence, and
oversample turn onsets. But the direction comes from the **future change events**, so
the deployment-realistic hold-last form loses it. The one-step baseline the harness
measures remains the "ignores the action" model. Getting directional action-following
out of a play-time model needs an explicit directional signal — see §9, and §7.10–§7.11
for where the direction lives and where it stops.

### 7.10 P4 — the direction is extractable (a direct heading readout)

Before reaching for an engine change or a door detector, we ask the narrower question
the hold-last result (§7.8) left open: is the turn direction *in the pixels + actions*
at all, at this resolution and capacity? A **direct heading predictor** (`head_predict.py`,
no image generation) reads K context frames + K context actions (click at `a_{K-1}`) and
regresses the net heading change over the next H steps from the sidecar. Trained on
click-onset windows, evaluated on val click-onset windows (100 across the 15 val
episodes — the split caps distinct episodes at 15, so the requested ≥20 is not reachable)
against chance, an **oracle** (sign of the nearest reachable door's `screenX` to the
click X — the engine's rule, i.e. the ceiling), and a near/far door-distance breakdown.

| model (binary sign acc) | 64×48 | 96×72 |
|---|---|---|
| chance / majority | 0.50 / 0.56 | 0.50 / 0.56 |
| **oracle (ceiling)** | **0.87** | **0.87** |
| base (mirror) | 0.78 | 0.75 |
| + explicit ΔactionX input | **0.85** | 0.77 |
| base, no mirror | 0.81 | — |

On the clean two-opposite-doors subset the same ranking holds (base 0.64 / +ΔactionX 0.77
at 64×48; 0.59 / 0.72 at 96×72), and near doors are easier than far (oracle 0.90 vs 0.85).
The direction is **fully extractable**: the best model (64×48 + ΔactionX, 0.85) sits
essentially at the oracle ceiling, and the explicit `ΔactionX = a_{K-1} − a_{K-2}` change
input helps, chiefly on the two-door case. Resolution is not the bottleneck — 96×72 is
no better than 64×48. Mirror augmentation is roughly neutral for the readout (0.78 vs
0.81).

### 7.11 P5 — but it does not reach the frames (auxiliary heading head)

P4 shows the direction is there; the question is whether it can be *forced into the
frame output* of the hold-last model. `train_hold_aux.py` retrains the hold-last FiLM
U-Net with the P4 heading readout added as a **training-only auxiliary head** on the
film-conditioned bottleneck (`loss = H-step frame L1 + 10·heading MSE`). At inference
only the frame head is used. Result:

* The auxiliary head **reads the direction at 0.893** sign accuracy (corr 0.81 with the
  true heading change) — at or above the 0.869 oracle. The shared encoder + film
  conditioning therefore *do* compute the turn direction.
* The frame output does **not** inherit it. The held-click ablation is unchanged from
  `hold10` (zeroed 0.216 worst, shuffled 0.209 — *turn presence*, not direction), and
  the `probe_click` counterfactual is flat: left/right/hold shifts all +0.000, both-correct
  0.00.

So the direction signal is present in the bottleneck but stops there: the frame decoder
is trained *only* by the H-step pixel L1, and under hold-last the click's pixel-scale
effect (~1 px) sits below that loss's noise floor, so the decoder never learns to map
"direction feature in the bottleneck" → "directional pixel motion". An auxiliary scalar
head fixes the encoder, not the decoder.

### 7.12 P6 — the generator can express the shift, but only through a structured warp + an explicit direction signal

P5 left two readings: either the direction never reaches the generator, or the
generator cannot render a small global view shift under a pixel loss. P6 separates them.

**P6a — oracle intent as an explicit input (`train_intent.py`).** The hold-last FiLM
U-Net is given the ground-truth turn direction as an explicit input: the screenX of the
reachable door nearest to the click's `actionX` (the same quantity the P4 heading head
is trained to predict), fed both as a scalar into the FiLM conditioning and as a
Gaussian "column map" image channel marking that screen column. Trained with true
values, hold-last, mirror. Result: the model *uses* the intent strongly — zeroing it
raises val click-onset H-step L1 from 17.9 to 25.9 (shuffled 27.4; a 44–53%
degradation) — so direction does reach the generator. But it still does not reach the
*frames as motion*: the click counterfactual stays flat (both-correct 0.17,
predicted-vs-real shift ratio 0.20). The intent improves the predicted frame (content,
brightness) but the pixel-residual decoder never maps it to a coherent horizontal shift.

**P6b — structured warp + residual output.** The output is reparameterized as
`warp(last frame, predicted scalar shift) + residual`. With a full residual the warp is
*shadowed*: the decoder ignores the shift scalar and paints via the residual
(both-correct 0.27, ratio 0.27). Removing the residual — **warp-only**, where the
predicted scalar shift is the only knob — makes the generator follow the intent
decisively (n = 30 val contexts, 15 episodes):

* combined click + intent: mean shift left +3.6 px / right −4.8 px / none −0.02 px,
  **both-correct 0.87**, predicted-vs-real shift ratio **0.65** (GT |shift| 6.8 px);
* intent-only (action held fixed, only the intent varies): **both-correct 0.83**,
  ratio 0.59.

So the generator *can* express the small global shift — when the shift is an explicit
parameter rather than something the residual has to paint. Dropping the oracle intent
(warp-only, `actionX` conditioning alone) is **not** enough: both-correct 0.43 with a
left/right asymmetry (right-correct 0.83 vs left-correct 0.53). The explicit direction
signal is required; `actionX` alone does not carry it.

One side-finding: the sidecar's literal "sticky target" (`targetDoorwayId`) is a laggy
signal — its screenX agrees with the actual turn only 0.55 of the time at click onset
(debounce hysteresis), whereas the door nearest to the click agrees 0.87. Use the
latter as the oracle; the P4 heading head already predicts the same sign.

**Answer:** the failure was in the generator's ability to *express* a small global view
shift under pixel loss, not in getting direction in. The variant that meets the gate
(both-correct ≥ 0.7, ratio ≥ 0.5) is a structured warp output conditioned on an
explicit direction signal — and it needs that signal (the P4 readout), not `actionX`
alone.



---

## 8. Failure modes observed (with evidence)

1. **Action ignored.** Ablation ties across true/zeroed/shuffled/shifted (§3.3), and
   the action-following shift is the same with and without the action (§3.5).
2. **Blurry averages where new content is revealed.** In `predictions.png`, doorways
   and newly-visible corridor regions come out as blurry means rather than crisp
   geometry — the model minimizes pixel error by averaging over the many possible
   next frames. This is the expected failure and the argument for a generative model.
3. **Rollout degradation.** Autoregressive error compounds; the model stops beating
   persistence by step ~3 (§3.4), degrading as blurring, geometry melting and color
   drift. The colour drift is a red tint that grows out of dark regions: the residual
   has no red bias (mean residual ≈ 0 per channel; dark regions get a uniform, slightly
   blue neutral brightening), but in autoregressive rollouts the model fills dark/
   newly-revealed regions with warm-tinted "average" content, and a few pixels' red
   excess compounds from ~0.03 to ~0.12 over 40 steps. No red appears in one-step
   outputs (0 red pixels on the exact `predictions.png` windows) — it is a rollout-only
   manifestation of the blur-average failure, not a bug in the review or training code.
4. **Directional bias.** Rightward turn bias in counterfactuals (§3.6), traced to the
   42.5/57.5 heading imbalance in the pilot (§7.5).

---

## 9. What this suggests for the diffusion stage

* **A generative model is needed for the "blurry average" failure** — when new
  content appears, a deterministic L1/L2 predictor must average over possibilities; a
  diffusion model samples one. This is the single strongest justification in the data.
* **The direction is extractable, but must be fed to the generator explicitly — pixel
  loss cannot transport it.** The effect-size analysis (§7.1) applies to *any* one-step
  frame loss: at 20 Hz a single click moves the next frame by ~0.03 px. Supervising at
  H≈10 steps (§7.7) makes a deterministic model action-responsive — but only when it
  conditions on the *true future* action sequence, whose mid-horizon click "change
  events" carry the turn direction (§7.8); the deployment-realistic hold-last form keeps
  only "turn presence", not direction. The follow-up probe (§7.10–§7.11) closes the
  loop: a direct heading readout from pixels + actions reaches the oracle ceiling
  (0.85 vs 0.87 at 64×48, §7.10), so the information *is* there — yet adding that same
  readout as an auxiliary head on the hold-last frame model does **not** push direction
  into the frames, because the frame decoder is trained only by the H-step pixel L1,
  whose noise floor sits above the hold-last click's ~1 px effect (§7.11).
  **Recommendation:** give the diffusion model the *explicit direction signal as a
  conditioning input*, not as a pixel-loss-only target — e.g. condition the denoiser on
  the heading readout's output (or on `ΔactionX = a_t − a_{t−1}`, or on a supervised
  door-position/heading latent), so the generator is told which way to turn rather than
  asked to rediscover it in a loss that cannot see it. §7.12 adds the second, decisive
  half: the generator must also be given a *structured motion mechanism*. A plain
  pixel-residual decoder cannot render a ~7 px view shift even when told the direction
  (both-correct 0.17); reparameterizing the output as a differentiable warp of the last
  frame by a predicted shift — with a residual reserved for newly-revealed content —
  reaches both-correct 0.87 / ratio 0.65. The residual must not be allowed to *shadow*
  the warp (a full residual collapses back to direction-blind), so either supervise the
  shift head directly, keep the residual low-resolution/low-capacity, or compute the
  shift from the readout rather than through the pixel loss. The direction signal itself
  is required: `actionX` conditioning alone gives only 0.43 and a left/right asymmetry.
  Keep the redesigned click counterfactual (`probe_click.py`), the horizon ablation, and
  the heading-readout accuracy (`head_predict.py`) as the action-responsiveness gate,
  and judge direction with them, not one-step pixel error.
* **The model must be robust to the turn-direction imbalance** (mirror augmentation or
  a rebalanced pilot), or it will inherit the rightward bias. The imbalance is in the
  *pilot* (signed heading changes skewed negative); the engine's world generation is
  symmetric (§7.9), so mirror augmentation — not an engine change — is the right fix.
* **Real-time budget.** A single forward pass is already far faster than 15 fps (§5);
  the room that leaves is what a multi-denoising-step diffusion model will consume.

---

## 10. Shortcuts taken

* No hyperparameter sweeps, no other model families, no heavy dependencies (L1/MSE/
  PSNR/SSIM and phase correlation are implemented locally in `metrics.py`).
* One alternative conditioning scheme (FiLM) tried, per the brief, then abandoned.
* Multi-step and non-detached variants were run only at 64×48 and only enough to
  answer the action-use question (2,500 steps each); the adopted baseline is the
  one-step model.
* P4 ran a small grid (64×48 / 96×72 × ±ΔactionX × ±mirror, 4,000 steps each); P5 ran
  a single auxiliary weight (10), no weight sweep. Both are answer-the-question probes,
  not tuned models.
* P6 ran four configurations at 64×48 (intent-only, warp+residual, warp-only, and
  warp-only without intent), one seed each, 6,000 steps — enough to answer the
  express-vs-ingest question, not a tuned model. The counterfactual has only 30 val
  contexts (15 episodes), so the 0.87 is directionally clear but small-n.
* The learning curve uses the same 6,000-step budget for all three points (fixed-step
  rather than fixed-epoch).

## 11. Changes to Stage 3 code

None — `packer.py`, `dataset.py`, and the existing loader tests were left untouched
and still pass. New **additive** modules only: `analysis.py` exposes the already-packed
`analysis.jsonl` sidecar (heading, pose, doorway candidates, and now the sticky-target
intent fields `targetDoorwayId`/`targetPointX/Z`/`playerX/Z` with `heading_error` and
`target_screen_x` helpers) keyed onto global frame indices for the harness;
`head_predict.py` (P4), `train_hold_aux.py` (P5), and `train_intent.py` (P6) are
standalone probe scripts. None modify the loader, packer, or the Stage 3 model.

## 12. `.gitignore` edits

Added `/neural/runs/` (training checkpoints and logs) and `/neural/review/` (the
three review images). (The packed data, venv and caches were already ignored from
Stage 3.)

## 13. Decisions taken and what remains open

Resolved (from the probe):

1. **The baseline stays the one-step model.** The H-step lookahead is adopted as a
   *principle* — supervise where the action has pixel-scale effect (§7.7) — not as the
   baseline model.
2. **The click-direction counterfactual is the gate**, redesigned (`probe_click.py`):
   distinct mid-episode val contexts, close doors on both sides, a proper "no new
   click" baseline. Under it the true-future model turns the correct way only for a
   *change event*; the deployment-realistic hold-last form does not (§7.8).
3. **Mirror augmentation used throughout**; the engine's world generation is symmetric
   (the turn imbalance is in the pilot, not the engine) — §7.9.
4. **Review images regenerated** with distinct mid-episode contexts and non-overlapping
   labels; visual judgment is the user's.
5. **P4 — direction is extractable.** A direct heading readout from pixels + actions
   reaches the oracle ceiling at 64×48 (0.85 vs 0.87; chance 0.50), so no engine change
   or door detector is needed to *obtain* the direction (§7.10). The explicit
   `ΔactionX` input helps; resolution (96×72 vs 64×48) does not.
6. **P5 — but an auxiliary heading head does not push it into the frames.** The head
   reads direction at 0.893, yet the frame output stays direction-blind (counterfactual
   both-correct 0.00, ablation unchanged from `hold10`) — the frame decoder, trained
   only by H-step pixel L1, cannot map the bottleneck's direction feature to pixel
   motion (§7.11).
7. **P6 — the failure is in *expressing* the shift, and a structured warp + explicit
   direction signal fixes it.** An oracle-intent input is used strongly by the model
   (H-step L1 44% better than zeroed intent) but a pixel-residual decoder still renders
   no shift (both-correct 0.17). Reparameterizing the output as a differentiable warp of
   the last frame by a predicted scalar shift — warp-only, so the residual cannot shadow
   it — reaches **both-correct 0.87, predicted-vs-real shift ratio 0.65** under
   hold-last, and the turn follows the intent alone (0.83). It *needs* the explicit
   direction signal: `actionX` conditioning alone gives 0.43 and a left/right asymmetry.
   So the diffusion stage should condition on the P4 readout's direction *and* render
   motion through a structured warp/flow, not a free residual (§7.12, §9).

P3 (joint prediction of `t+1..t+H` under hold-last) was **not** run — it is
conditional on P1 *and* P2 both being positive, and P2's direction is not (§7.8).

Resolved for the diffusion stage (was open):

* **How to make a play-time model follow the click's direction.** Hold-last
  conditioning — the only thing available at play time — yields "turn presence" but not
  direction (§7.8). P6 closes it: the direction *is* extractable (§7.10) and *can* reach
  the frames, but only when the generator is (a) conditioned on the explicit direction
  signal and (b) given a structured motion output (a differentiable warp by a predicted
  shift) instead of being asked to paint a ~7 px shift through a pixel residual. The
  caveat for the real model — which, unlike the warp-only probe, must also reveal new
  content — is that the residual must be prevented from shadowing the warp (supervise
  the shift, or keep the residual low-resolution/low-capacity) (§7.12, §9).
