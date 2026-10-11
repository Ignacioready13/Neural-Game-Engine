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
This is the central finding of the stage and is explained in §7. The result is *not*
evidence of a label bug: the one-frame-shifted actions tie rather than beat the true
alignment, and the ground-truth heading correlates strongly with the observed image
shift (r = −0.87, see §3.5), which is exactly what the alignment rule predicts.

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

## 7. Why action conditioning is hard here

The ablation (§3.3) shows the model ignores the action. This is not a code bug — the
conditioning *pathway* works — it is a property of the data at 20 Hz. We ran a
four-step investigation to prove it.

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
only ~4% (0.0501 vs 0.0521) and tie shuffled/shifted. The conclusion is robust: **at
this frame rate, action resolution and loss, a deterministic pixel-predictor cannot be
taught to use the action**, because the per-step action effect is buried in pixel
noise and the cumulative effect is dominated by the model's own compounding rollout
error.

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

The one honest fix for *this* architecture would be to make the action signal
stronger or the objective sharper — neither of which is in scope for a baseline. The
adopted baseline is therefore the simplest scheme that beats persistence, and the
"model ignores the action" result is documented as a finding rather than papered over.

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
   drift.
4. **Directional bias.** Rightward turn bias in counterfactuals (§3.6), traced to the
   42.5/57.5 heading imbalance in the pilot (§7.5).

---

## 9. What this suggests for the diffusion stage

* **A generative model is needed for the "blurry average" failure** — when new
  content appears, a deterministic L1/L2 predictor must average over possibilities; a
  diffusion model samples one. This is the single strongest justification in the data.
* **The action will still be hard to use.** The effect-size analysis (§7.1) applies
  to *any* frame-level loss: at 20 Hz a single click moves the next frame by ~0.03 px.
  The diffusion model should be evaluated with the same horizon ablations and
  action-following/counterfactual metrics, not just one-step pixel error, or it will
  look like it uses the action when it doesn't.
* **The model must be robust to the turn-direction imbalance** (mirror augmentation or
  a rebalanced pilot), or it will inherit the rightward bias.
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
* The learning curve uses the same 6,000-step budget for all three points (fixed-step
  rather than fixed-epoch).

## 11. Changes to Stage 3 code

None — `packer.py`, `dataset.py`, and the existing loader tests were left untouched
and still pass. One new **additive** module, `analysis.py`, exposes the already-packed
`analysis.jsonl` sidecar (heading, pose, doorway candidates) keyed onto global frame
indices, for the harness to use; it does not modify the loader or packer.

## 12. `.gitignore` edits

Added `/neural/runs/` (training checkpoints and logs) and `/neural/review/` (the
three review images). (The packed data, venv and caches were already ignored from
Stage 3.)

## 13. Needs user decision

1. **Whether the action-ignored baseline is acceptable**, or whether to invest next in
   an auxiliary heading/action-supervision head, a lower-K variant, or a larger/shifted
   horizon before the diffusion model.
2. **Whether to rebalance or augment the pilot** (mirror augmentation) to remove the
   rightward turn bias before training the diffusion model.
3. **Visual review** of the three review images (see below) — whether the "blurry
   average" failure and the counterfactual bias match expectations.
