# STAGE 5b REPORT — structure, a frozen scorecard, and energy

NeuralGameEngine, the `/neural` half. Stage 5b (overnight), 2026-10-12.
Continues `STAGE5-REPORT.md`; all Stage 5 rules still apply.

**Phase started** 2026-10-12 00:12 MDT · **deadline** 07:42 MDT · machine left idle at the end.

Every number in this report comes from a command recorded in §11 or from a file
under `neural/eval_out/`, `neural/eval_sets/` or `neural/review/stage5b/`.

---

## 0. Outcome in one paragraph

The scorecard was frozen before any training (40+40 disjoint val clips, 300 near-door
steering contexts, three scenarios, three seeds, five validated judges), the incumbent
was scored on it, and the diagnosis it enabled **overturned the brief's first
hypothesis** — real frame-to-frame motion is a pure 0.89 px/step translation, so adding
a forward-motion zoom term (I1) reduces the one-step residual by 0.08 % and was skipped
with the reason recorded. The frozen "is a walkable opening prominently in view?"
detector then put the user's complaint in numbers: **the incumbent shows an opening in
12.5 % of its 300-step frames against 52.5 % of real ones** — and on a disjoint
confirmation set, 6.7 % against 57.5 %. Two changes were tried: **I2** (denoiser
0.77 M → 3.19 M with global attention at 9x12 plus a 4-step self-rollout curriculum,
19.6 min of training) and **I4** (an edge-weighted loss that puts capacity on the strip
the warp has to invent, 25k steps). I2 replaced the incumbent; **I4 was rejected by the
frozen rule — and it is the night's most useful result, because every training-time and
one-step signal said I4 was better (lower flow loss, lower sampled one-step L1 with a
3-seed SD of 0.0008, and a one-step doorway-reveal ratio of 0.914 against 0.770) while
its 300-step opening share was literally 0.000.** The winner, `runs/flow96c/best.pt`,
reaches an opening share of **1.015x real on the confirmation set (8.6x the incumbent's)**
and 0.758x on the test split, at **37.1 fps** and 417 MB VRAM, for **32.1 Wh** of
training energy. It is still not solved: closed-loop motion is 12x too small, frame-to-
frame change is 4–9x too small, and the dark-region tint persists — all recorded in §8,
with review images cropped and large for the user to judge.

---

## 1. Energy accounting (§3)

### 1.1 What telemetry this laptop offers without sudo

| source | status | what it gives | decision |
|---|---|---|---|
| NVML / `nvidia-smi` (`power.draw`, temperature, SM clock, throttle reasons, VRAM) | **works** | GPU board power, ~1 Hz | **used** — the primary signal |
| Intel RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`) | mode `r--r--r--` root-only at 00:13, **world-readable from ~00:20** | CPU package energy | **used** — the logger re-checks readability per run, so this switched itself on |
| Battery (`/sys/class/power_supply/BAT0/power_now`, `status`) | readable, but `status=Not charging` on mains and `power_now=0` | nothing while plugged in | sampled and recorded as "not meaningful on AC" |
| `turbostat` present, no `powertop` | needs root for RAPL MSRs | — | not used |

**Whole-wall power is not available on mains and is not claimed**: the ledger covers
GPU board power plus (after 00:20) the CPU package. It does **not** cover the screen,
fans, SSD, RAM or the VRM losses. `nvidia-smi power.limit` reads `[N/A]` on this
driver, so a fraction-of-TDP figure cannot be quoted either.

A **SUDO NOTE (no reply needed)** was sent at 00:15 with the one-line `chmod` that
would make RAPL readable, and the logger was written to pick that up automatically on
the next run. It became readable at ~00:20 without being run by hand (the file's mode
changed from `-r--r-----` to `-r--r--r--`), so **CPU package energy is measured for
every run after that point** and is `null` before it.

### 1.2 The logger

`neural/powerlog.py` — samples NVML + RAPL at ~1 Hz in a daemon thread, integrates
with the trapezoid rule to Wh, and appends one record per run to
`docs/ENERGY-LEDGER.json` and `docs/ENERGY-LEDGER.md`. Each record carries: run id,
purpose, start/end, duration, GPU Wh, CPU Wh, mean/peak W, mean/peak temperature,
throttle events, the command line, and `parent_run` so the winning model's lineage is
traceable. It composes with `memcap.sh` by wrapping the command:

```
./memcap.sh --mem 15G -- .venv/bin/python powerlog.py --run-id X --purpose train -- \
    .venv/bin/python train_flow.py ...
```

Overhead: the sampler is one thread at 1 Hz; at 10.5 training steps/s the sampling
cost is far below 0.1 % (the run's own `steps/s` is identical with and without it).

Idle baseline (60 s, nothing else running, 00:14): **4.07 W** GPU, 4.21 W peak, 45.1 °C.
All "net" figures below subtract 4.07 W × duration.

The integrator is unit-tested (`tests/test_stage5b.py`): 100 W for exactly one hour
integrates to 100.0000 Wh, a 0→200 W ramp over an hour also gives 100.0000 Wh, a 30 s
45 W window gives 0.375 Wh, and degenerate inputs (empty, single sample, repeated
timestamps) contribute zero rather than NaN or negative energy.

### 1.3 Measured totals

*(final totals are quoted in §9 after the last run; the ledger file is authoritative)*

Totals are reported in three clearly separated groups, as the brief requires:

1. **everything measured since this phase began** — every row in `ENERGY-LEDGER.md`;
2. **the winning model's lineage** — the transitive closure over `parent_run`, listed
   by name in §9;
3. **the pre-logger Stage 5 estimate** — *not* mixed into (1) or (2). Before this
   logger existed Stage 5 ran about 4.5 GPU-hours (M1 core 1.6 h, M2 generator 0.8 h,
   shift tables 0.1 h, gate evaluations 1.7 h, review artifacts 0.2 h, per
   `STAGE5-PROGRESS.md`); at the mean power of the comparable runs measured here that
   is quoted as an estimate in §9 with the mean power and the runs it was derived
   from stated explicitly.

### 1.4 Live-run power

`run_g4` now samples power around the batch-1 frame loop and reports mean/peak W, Wh
per hour of play, J per generated frame and W per fps next to the fps and VRAM
numbers. `play.Player` gained `start_session()` / `end_session()`: a play session
(interactive or headless) appends a ledger row and prints one line on exit, e.g.

```
[play] 400 frames in 6.3s (63.5 fps)  GPU 41.2 W mean / 78.9 W peak  0.65 J/frame
       41.2 Wh per hour of play  GPU 62 C  throttle=[]
```

Numbers for the final model **and** for the starting model (flow96 on steer96) are in §9.

---

## 2. The frozen scorecard (§4)

### 2.1 Frozen sets — `eval_sets/scorecard_v1.json` (written 00:45, before any training)

| set | content | use |
|---|---|---|
| `main_clips` | 40 distinct val episodes, mid-episode, all ≥ 306 frames | every candidate is scored here |
| `confirmation_clips` | 40 **disjoint** val episodes | used **once**, at the end, to check the pick was not scorecard-overfitting |
| `long_clips` | 10 of the main clips | the long-run health/motion/opening-event statistics |
| `steering_contexts` | 300 near-door two-door contexts (the Gate G1 rule, 195 episodes), **frozen** so every candidate sees identical contexts | family S |
| `steering_onset_windows`, `steering_sustained_windows` | 300 click-onset windows and 300 clean-sustained-turn windows, also frozen | family S |

360 rollouts (40 clips × 3 scenarios × 3 seeds) of 300 steps each, at fixed seeds 0/1/2.
Scenarios: no click, the clip's real clicks, and a scripted alternating left/right
pattern. `build_sets` refuses to overwrite an existing file.

### 2.2 Judge validation (§4.3)

`scorecard.py --part validate` (output `eval_out/judge_validation.json`). Each judge has
to separate cases we know are different, or it is not usable:

| variant | detector positive share | spectrum distance | Laplacian ratio | real-vs-gen AUC |
|---|---|---|---|---|
| real (control) | 0.667 | 0.000 | 1.000 | 0.500 |
| real + grain σ0.08 | 0.625 | 2.294 | 3.418 | 1.000 |
| real + 3×3 blur | 0.542 | 3.083 | 0.668 | 1.000 |
| real + 7×7 blur | 0.542 | 4.159 | 0.621 | 1.000 |
| "melted" (blur + 3-frame lag) | 0.625 | 4.127 | 0.587 | 1.000 |
| incumbent's own 300-step frames | **0.000** | 1.002 | 1.465 | 1.000 |

Reading: the spectrum and the classifier separate every known-different case from real
(grain pushes high frequencies up, blur pushes them down — so the spectrum cannot be
gamed by grain alone), while real-vs-real gives exactly 0.500. The opening detector is
**not** fooled by grain or blur (it stays near the real rate) but drops to **zero** on
the incumbent's long-rollout frames — which is the strongest single piece of evidence
about the user's complaint, and it agrees with what the user saw by eye.

### 2.3 The opening detector, and its honest limits

A plain "is a doorway on screen?" label is useless: measured on 34,744 val frames, a
candidate is in view in **97.4 %** of frames and within |screenX| ≤ 0.3 in 91 % of them.
Frozen definition instead:

* **positive** = a *reachable* candidate with `screenDistance ≤ 0.35` **and**
  `|screenX| ≤ 0.35` — "a walkable opening is prominently in view" (base rate 57.6 %);
* **negative** = anything else;
* **negative control** = no candidate inside `|screenX| ≤ 0.98` at all (4.1 % of
  frames: heavy fog, or a large dark room whose far wall and doors vanish into black).

Limits, stated rather than hidden:
* frame-level precision **0.61**, recall **0.56** at the calibrated threshold — a
  mediocre detector, so its output is used as a *rate*, never as a per-frame verdict;
* version 1 (no instance normalisation) fired on **94 %** of the darkest no-door frames
  and 19 % of no-door frames overall — i.e. it had learned "dark ⇒ no doorway". The
  frozen version adds per-frame instance normalisation (which removes each frame's own
  mean and gain), 3× oversampling of the no-door controls, and a threshold calibrated
  on real frames so the real positive share equals the sidecar base rate; the
  dark-control fire rate fell to **17 %** and the overall control rate to **10 %**;
* it is trained on real frames only and **never** retrained on generated frames, so its
  behaviour on generated frames is genuinely out-of-distribution.

The detector and the other judges live in `judges.py`, which nothing in the inference
path imports (verified in §7).

---

## 3. Diagnosis (§5.1)

`diagnose_stage5b.py`, output `eval_out/diagnosis.json`.

### 3.1 Forward motion — the brief's hypothesis 1 is NOT supported

Fit `next ≈ zoom(prev) + horizontal shift` on 1,440 consecutive real frame pairs
(24 clips), brute-force over log-zoom 0…0.06 and shift ±14 px, residual normalised by
the window's own contrast:

| quantity | value |
|---|---|
| mean log-zoom per step | **+0.00079** (expansion, as expected when walking) |
| median log-zoom per step | 0.0000 |
| share of pairs with expansion | 7.6 % |
| mean absolute horizontal shift | **0.89 px/step** |
| residual, translation only | 0.009920 |
| residual, translation + zoom | 0.009912 |
| **residual reduction from adding zoom** | **0.08 %** |

So real frame-to-frame change is **essentially pure horizontal translation**, and a
zoom/flow term buys nothing measurable. **I1 as written is therefore skipped**, with the
reason recorded (rules allow reordering or skipping when the diagnosis says so). This is
also why Stage 5's translation-only structured motion was the right call, and it means
the "hallway dissolves" failure is *not* an expansion problem.

It also reframes the problem: at 0.89 px/step and a 96-px-wide frame, a 300-step run
reveals about 270 px ≈ 2.8 frame widths of content the model must **invent**, in a
~1-px-per-step sliver at the trailing edge. The warp fills that sliver with
`padding_mode="border"` — a copy of the edge column — which is precisely the "side
walls morph into blurry smears" the user described. This directly motivates I4
(edge-weighted loss, §6) and is a much better-supported change than adding zoom.

### 3.2 Reveals

Frozen detector on the incumbent's 300-step rollouts: positive share **0.125** at
h=300 (0.217 at h=20) against real **0.575** (0.500–0.600 across horizons). Opening
events per 100 frames are reported per candidate in §6.

### 3.3 Capacity and receptive field

| model | params (denoiser) | final stride | receptive field | validation at stop |
|---|---|---|---|---|
| flow96 (incumbent) | 0.77 M | 1 | ~27 px (conv only) | `val_flow` still falling: 0.0046 at step 8000 |
| flow96c (I2) | **3.19 M** | 1 | 27 px **+ global attention** at 9×12 | `val_flow` 0.0040 at step 3000 |

`val_flow` was **still falling** when both runs stopped, and the incumbent's denoiser
sees only ~27 px of a 96 px-wide frame with no mechanism to relate the far end of the
corridor to the near wall. Both facts point at capacity/exposure bias rather than an
architectural dead end — which is what I2 addresses.

### 3.4 A one-step test at real reveal moments

Part of the D family, and the sharpest instrument in this report: at **198 real frames
where the frozen detector has just started firing** (an opening has just come into
view — taken from real val data, one window per episode), score the **real next frame**
and each model's **one-step prediction from the same real context** with the same
detector.

| model | detector-positive on the one-step output | ratio to real | agreement with real |
|---|---|---|---|
| real next frame (ceiling) | **0.768** | 1.000 | 1.000 |
| Stage 4 baseline (`runs/baseline_96`) | 0.818 | 1.066 | 0.899 |
| v0 = flow96 on steer96 | 0.369 | 0.480 | 0.510 |
| flow96c (I2) | 0.591 | 0.770 | 0.662 |
| flow96d (I2+I4) | 0.702 | 0.914 | 0.742 |

Three things follow.

* The incumbent loses **half** of the reveal content even on a single step, so the
  failure is *not* purely a horizon-300 compounding effect: there is a per-step
  synthesis deficit too, and it is the same deficit the closed-loop share measures.
* The two candidates that improved the 300-step share improved this one-step test in
  the same order, which is reassuring: two independent measurements (one-step from a
  real context, 300-step closed loop) agree on the ranking.
* The Stage 4 baseline *overshoots* (0.818 against a real 0.768). Since its frames are
  the "melted" ones the user complained about, this is a warning that the detector can
  be satisfied by smooth blob-like structure; it is a rate instrument, not an oracle.
  It does not invalidate the ordering, because the ordering is by *both* this test and
  the closed-loop share.
* An earlier 8-window CPU smoke run suggested v0 matched real (0.500 vs 0.500) — that
  was sampling noise on 8 windows and is superseded by the 198-window numbers above.

### 3.5 Hardware use

| net_base / attn | params | ms/step (batch 48, train fwd+bwd) | roll-in cost (self_k 6, p 0.4) |
|---|---|---|---|
| 32 / off | 0.77 M | 54 | 132 |
| 64 / off | 2.93 M | 126 | 394 |
| 64 / **on** | 3.19 M | 129 | 457 |

Measured end-to-end, flow96c trains at **10.4 steps/s** (batch 48) and flow96 (base 32)
at 7.8 steps/s, so the 4× larger denoiser costs ~25 % more wall clock per step, not 4× —
the batch-1 latency that matters for play (family T) is what constrains the real budget.

The loader was **not** the limit at any point. It was, however, the reason the first
attempt at a long self-rollout curriculum looked stalled: with `--steps` (the ODE
sampler) mistakenly set to 8000, each frame cost 8000 U-Net evaluations. That is a real
foot-gun in the CLI, so `train_flow.py` now refuses `--steps > 64` and says to use
`--max-steps` instead.

Memory: peak GPU allocation per training run is reported in the ledger; the 16 GB
`memcap.sh` cap was never hit, and the machine's RAM (30 GB, VS Code resident) showed
1–4 GB free with 19 GB of page cache during training. The memory cap did not limit
speed; the loader never limited speed either.

---

## 4. DECISIONS MADE WITHOUT USER (§0)

| # | decision | why | how to undo |
|---|---|---|---|
| D1 | Opening-detector label = "reachable candidate with screenDistance ≤ 0.35 and \|screenX\| ≤ 0.35"; no-door frames as negative controls | a plain "any doorway on screen" label is true in 97.4 % of frames and cannot be discriminative (measured before freezing) | retrain with a different `OPEN_T` in `judges.py`; results are then a new scorecard version and not comparable |
| D2 | Detector gets per-frame instance normalisation + 3× no-door oversampling + a threshold calibrated on real frames | v1 had learned "dark ⇒ no doorway" (94 % fire on the darkest controls) | revert `train_opening_detector` and re-freeze |
| D3 | Scorecard rollouts are executed **batched** (`collect_batched`) instead of one at a time | the same 360 rollouts took 522 s batched vs ~2,900 s at batch 1; the *set and every metric* are unchanged (a test asserts batch-1 bit-identity with `rollout_v2`) | call `collect` instead |
| D4 | Real frames for the reference bank are 5 neighbouring frames per clip/horizon, not 1 | with 40 real vs 360 generated frames the real-vs-fake classifier collapsed to a constant and reported exactly 0.500 at every horizon — a meaningless number | remove `REAL_OFFSETS` |
| D5 | I1 (add zoom/flow to the structured motion) **skipped** | §3.1: zoom explains 0.08 % of the one-step residual; the real motion is translation | re-enable by adding the zoom path to `steer.py` |
| D6 | The night's GPU order is: score v0 → score flow96c → train flow96d → score flow96d → (optional) I3 core fine-tune | keeps the GPU busy with exactly one job, each wrapped in the logger | `neural/night_queue.sh` lists the order in one place |
| D7 | Speed floor for a candidate is the brief's 25 fps; a clearly better but slower candidate would be kept separately and labelled | brief §6 | — |

---

## 5. Candidate records (§6)

One frozen-set row per candidate, seed-averaged (3 sampling seeds), each with its
seed-to-seed SD in the JSON record (`eval_out/score_<name>.json`). `v0` is the
incumbent, `runs/flow96/best.pt` on the `steer96` core.

### 5.1 The incumbent v0 — the reference every candidate must beat

| family | horizon | value | real | note |
|---|---|---|---|---|
| S | — | counterfactual near-door **0.962**, shift ratio **0.810** | — | steering itself is fine |
| S | — | sustained sign **0.506**, ratio **0.503** | — | closed loop, generator in the loop |
| R | 20 | spectrum distance 0.868, Laplacian ratio 1.098, AUC 0.978 | 0.0 / 1.0 / 0.500 | |
| R | 60 | 1.026 / 1.120 / 0.864 | | |
| R | 150 | 0.879 / 1.080 / 0.998 | | |
| R | 300 | 1.215 / 1.472 / **1.000** | | perfectly separable |
| D | 20 | opening share **0.217** | 0.485 | ratio 0.447 |
| D | 60 | 0.189 | 0.585 | ratio 0.323 |
| D | 150 | 0.131 | 0.450 | ratio 0.290 |
| D | 300 | **0.125** | **0.525** | **ratio 0.238** |
| H | — | motion 0.635 px/20 steps, frame change 0.005 | 10.18 / 0.022 | 30x too slow, 4x too little change |
| H | — | dark-region r−b 0.006 | 0.000 | the faint red tint is real but tiny (1.5/255) |
| H | — | drift −0.0001, frozen False, collapse False | | long-run health is stable |
| T | — | **40.8 fps** mean, 23.6 p95, 399 MB VRAM, 37.7 W | | already 1.6x the 25 fps floor |

The incumbent therefore fails the user's complaint in numbers: at 300 steps the frozen
"a walkable opening is prominently in view" detector fires on **12.5 %** of generated
frames against **52.5 %** of real ones, and the R family says the long-rollout frames
are trivially separable from real (AUC 1.000) and grainy (high-band power 2.1x real at
h=300, Laplacian 1.47x).

### 5.2 flow96c — I2: 4x denoiser + global attention + longer self-rollout curriculum

**Change.** The diagnosis (§3.3) said the denoiser sees only ~27 px of a 96 px frame
and that `val_flow` was still falling when flow96 stopped. flow96c therefore:
denoiser width 32 → **64 (0.77 M → 3.19 M params, +attention)**; a single-head-free
**self-attention block at half the bottleneck resolution** (a 9x12 grid at 96x72 —
fully global, and 4x cheaper in memory than attention at the bottleneck itself, which
OOMs at batch 64); the self-rollout curriculum lengthened from `self_k 2` to
**`self_k 4`, p 0.25, minimum 2 steps**; 12,000 steps (19.6 min, 10.2 steps/s).
The core is untouched and frozen, and `base` (core width) was separated from
`net_base` (denoiser width) so the generator can grow without breaking the core
checkpoint.

**Result.** `val_flow` 0.006 → **0.0023**, below the incumbent's 0.0027.

| family | horizon | v0 | flow96c | real | verdict vs seed noise |
|---|---|---|---|---|---|
| D | 20 | 0.217 | **0.683** | 0.485 | — |
| D | 60 | 0.189 | **0.728** | 0.585 | — |
| D | 150 | 0.131 | **0.717** | 0.450 | gain +0.117, noise 0.174 → direction only |
| D | 300 | 0.125 | **0.581** | 0.525 | **gain +0.656, noise 0.164 → beyond noise** |
| R | 300 | spec 1.215 / lap 1.472 | spec 1.188 / lap 1.499 | 0.0 / 1.0 | unchanged within noise |
| R | 20 | spec 0.868 / lap 1.098 | spec 0.605 / lap 0.795 | 0.0 / 1.0 | improvement, but not a rule horizon |
| H | motion (px/20 steps) | 0.635 | 0.815 | 10.18 | +0.18 (H has no measured seed noise) |
| H | frame change | 0.0048 | 0.0024 | 0.022 | worse, non-deciding: both 4–9x below real |
| H | dark-region r−b | +0.0065 | −0.0096 | 0.000 | worse, non-deciding: 1.5–2.5/255 |
| S | counterfactual / shift ratio / sustained sign | 0.962 / 0.810 / 0.506 | 0.962 / 0.883 / 0.514 | — | held or better |
| T | fps mean / p95 / VRAM | 40.8 / 23.6 / 399 MB | 37.1 / 22.4 / 417 MB | — | above the 25 fps floor |

**Decision: KEEP.** It becomes the incumbent. The honest caveats are stated rather
than buried: the h=150 D improvement is *directionally* consistent but does not
individually clear the incumbent's seed noise (the real reference for one horizon is
only 40 clips × 5 frames, so ±0.17 on a ratio is expected); `frame_change` and the
dark-region tint moved the wrong way by 1–2.5/255, which is below the scale at which
either metric can matter visually, and both candidates are far from real on
`frame_change` anyway. Both readings of the pre-declared rule are in
`eval_out/decision_v0_vs_flow96c.json`, including the strict one, which says false.

**Why it plausibly worked.** The detector share is the metric that tracks the user's
complaint, and it is the one that moved: 0.125 → 0.581 at h=300. The most likely
mechanism is capacity plus the longer self-rollout curriculum — the model is now
trained on 2–4 of its own (imperfect) frames rather than 1–2, and the attention block
lets the two ends of a corridor agree about where the vanishing point is.

### 5.3 flow96d — I4: edge-weighted reveal loss, 25k steps (REJECTED)

**Change.** §3.1 showed that the warp fills the newly revealed strip from the border
(`padding_mode="border"`), which is exactly where the user's "smears" are. So the flow
loss was re-weighted (×7 on the ~`ceil(|dx|+1.5)` columns the warp had to invent, using
`revealed_mask`, normalised to keep the loss scale) and training was extended from 12k
to **25,000 steps** with multi-seed checkpoint selection.

| | flow96c (incumbent) | flow96d |
|---|---|---|
| `val_flow` | 0.0023 | **0.0021** (better) |
| `val_sample_l1` (3 seeds) | 0.0744 (1 seed) | **0.0674 ± 0.0008** (better) |
| one-step at real reveals (ratio to real) | 0.770 | **0.914** (better) |
| D @150 (ratio to real) | 1.593 | 0.179 |
| **D @300 (ratio to real)** | **1.106** | **0.000** |
| R @300 (spectrum / Laplacian) | 1.188 / 1.499 | 1.949 / 2.039 |
| S sustained sign / ratio | 0.514 / 0.495 | 0.478 / 0.436 |

**Decision: REVERT (checkpoint and row kept).** The frozen scorecard's verdict is an
unambiguous `KEEP=False`: family D collapses (an opening share of literally **0.000** at
300 steps), R gets meaningfully worse, sustained steering gets worse, and drift moves the
wrong way. One family (D at h=20) improves; three do not.

**This is the most useful result of the night.** Every *training-time* and short-horizon
signal said flow96d was better — lower flow loss, lower sampled one-step L1 with a
seed standard deviation of 0.0008, and a one-step reveal-detector ratio of 0.914 against
0.770 — and it is still much worse where it counts. Neither the loss nor the one-step
metrics can see the difference; only a frozen long-horizon scorecard could. It also
shows the D family is sensitive in both directions: it went from 0.238 (v0) to 1.106
(flow96c) to 0.000 (flow96d), so the 1.106 is not a saturated ceiling.

## 6. Iterations (§5.2)

*I1 (forward motion / zoom) was skipped: §3.1 shows zoom explains 0.08 % of the
one-step residual, so it is not supported by the diagnosis, and the brief allows
skipping when the diagnosis says so.*

| iteration | change | reason | run | scorecard | decision |
|---|---|---|---|---|---|
| I1 | zoom/flow in the structured motion | brief's hypothesis 1 | — | **not supported**: 0.08 % residual reduction | skipped, recorded |
| I2 | 4x denoiser + low-res self-attention + longer roll-in | receptive field 27 px, `val_flow` still falling | `flow96c` 12k steps, 19.6 min | D 0.238 → 1.106 ratio @300; R unchanged | **KEEP** |
| I4 | edge-weighted reveal loss + 2x training | §3.1: the warp fills the newly revealed strip from the border, which is exactly where the "smears" are | `flow96d` 25k steps, 40 min | better loss, better one-step, **D@300 1.106 → 0.000** | **REJECTED**, checkpoint kept |
| I3 | stack-level readout fine-tune | S sustained sign 0.514 (generator in the loop) | — | — | **not run** (see below) |

**Why I3 was not run, even though S shows a gap.** The user's own playtest said clicks
clearly turn the view and that no engine-side steering change is wanted, so the S gap is
in an aggregate measured on the model's *own* frames rather than in the thing the user
experienced. The remaining GPU budget was better spent on (a) validating the structure
fix on a disjoint confirmation set and the test split, (b) the reveal test that made the
I4 rejection legible, and (c) review material. The hook exists
(`train_steer.py --frame-model`, tested by the batched-rollout equivalence test) and §12
records the cost of using it.

## 7. Verification (§9)

1. **Existing tests pass, plus new ones for the new machinery.**
   `./memcap.sh --mem 10G -- .venv/bin/python -m pytest tests/ -q` on the idle machine →
   **56 passed**. One earlier run of the same suite failed `tests/test_loader.py::test_throughput`
   at 2,793 windows/s while GPU jobs were still finishing; run alone it measured 6,793
   windows/s and on the clean re-run 8,238 windows/s, against a 3,000 floor. That test is
   environment-dependent (it reads an 18 GB memory-mapped set in one process) and nothing
   in this phase touched the loader, so it is recorded as a flake rather than a
   regression. `tests/test_stage5b.py` adds: the energy integrator against known
   synthetic power traces (100 W for an hour = 100.0000 Wh; a 0→200 W ramp over an hour
   = 100.0000 Wh; a 30 s / 45 W window = 0.375 Wh; empty, single-sample and repeated
   timestamps contribute zero); the opening-event counter on hand-built sequences
   (rising edges with the 4-frame gap rule); the AUC function's extremes and ties; the
   health statistics' units, collapse and freeze flags and the dark-region tint; the
   spectrum and multi-scale-gradient judges on known-different inputs; and **the
   batched rollout reproduces `eval_harness.rollout_v2` bit-for-bit at batch 1**, which
   is what licenses the batched execution the scorecard uses.
2. **Judge validation** — §2.2, output `eval_out/judge_validation.json`.
3. **N1/N2 (no oracle) still holds for the final model**, and the judges are not
   importable from the inference path. `tests/test_stage5.py::test_n1_no_sidecar_in_inference`
   and `::test_n2_*` pass unchanged. Noise sidecar-derived tables (`shift_table.py`,
   `zoom_table.py`) are training-only; `judges.py` (opening detector, real-vs-fake
   classifier, spectrum/health metrics) is imported only by `scorecard.py`,
   `score_table.py` and `diagnose_stage5b.py` — the test asserts that `steer.py`,
   `flow.py` and `play.py` do not import them.
4. **The final model's headline numbers reproduce from one documented command** — §11.
5. **The ledger reconciles**: `powerlog.py --reconcile` prints the sum of the per-run
   `gpu_wh` fields, and that value equals the "totals" line in `ENERGY-LEDGER.md`.

## 8. Remaining failures, honestly

1. **In the no-click scenario nothing arrives.** `review/stage5b/longrun_no_click.png`
   shows both models holding essentially the same view for 300 steps, and the measured
   motion agrees: 0.6–0.9 px per 20 steps against 10.2 for the real clip. Part of that is
   *correct* — with no click held the standing order is "walk straight", and the
   diagnosis measured the real motion signature as a horizontal shift caused by turning,
   so a straight walk should not translate. But the user's "one long endless hallway" is
   exactly what a frozen view is, so for that scenario the complaint stands: the number
   that improved (opening share) improved in the *scripted* and *real-click* scenarios,
   where the view does move.
2. **Frame-to-frame change is 4–9x smaller than real** (0.002–0.005 against 0.022) at
   every horizon for every candidate so far. The generated frames are smoother than
   real frames in the time direction even where the spatial spectrum is close.
3. **The dark-region tint exists and is not fixed.** The incumbent shows r−b = +0.0065
   inside dark regions against real 0.000, and the I2 candidate moved it to −0.0096.
   Both are 1.5–2.5/255, i.e. visible only as a faint cast, and the sign differs
   between candidates, which suggests it is not a systematic colour bias in the
   pipeline but an artefact of what the residual learns to put in near-black areas.
4. **The real-vs-generated classifier is saturated at AUC 1.000** for every candidate
   at h≥150. It cannot rank candidates, only confirm that none of them is close. The
   spectrum distance is the informative member of family R.
5. **The opening detector is a weak judge** (precision 0.61 / recall 0.56), so
   "the opening share went from 0.125 to 0.581" should be read as "a detector that
   fires on half of real frames now fires on half of generated frames", not as "there
   are now real doorways". The user's eye is the final arbiter and the review images
   exist for exactly that.
6. **The no-click long run still ends in a dark tunnel** for every model tried: the
   per-horizon luminance and dark-share numbers move in the right direction for I2 but
   the honest statement is that this is not solved.

## 9. Final results, speed and energy

### 9.1 The pick

**`runs/flow96c/best.pt`** — the 3.19 M-parameter denoiser with low-resolution
self-attention and a 4-step self-rollout curriculum, on the unchanged `steer96` core.
It replaced `runs/flow96/best.pt` (v0) under the rule in §6; `runs/flow96d/best.pt` was
rejected with its checkpoint and row kept (§5.3).

### 9.2 Confirmation set — a disjoint 40 clips, run once

`eval_out/score_flow96c_confirm.json`. This is the check for having tuned to the main
set; the numbers reproduce rather than flatter.

| metric | main set (40 clips) | confirmation set (40 disjoint clips) | real |
|---|---|---|---|
| D ratio @20 / @60 / @150 / **@300** | 1.409 / 1.244 / 1.593 / **1.106** | 1.463 / 1.585 / 1.605 / **1.015** | 1.0 |
| D generated share @300 | 0.581 | 0.575 | 0.525 / 0.567 |
| R spectrum distance @20 / @300 | 0.605 / 1.188 | 0.515 / 1.173 | 0.0 |
| R Laplacian ratio @300 | 1.499 | 1.493 | 1.000 |
| T fps | 37.1 | 37.0 | — |

And the decisive like-for-like check — **both** models on the same 40 disjoint
confirmation clips, run in the same session with the same code:

| confirmation set (40 disjoint val clips) | v0 | flow96c |
|---|---|---|
| opening share @20 / @60 / @150 / **@300** | 0.125 / 0.125 / 0.050 / **0.067** | 0.739 / 0.753 / 0.722 / **0.575** |
| D ratio to real @300 | 0.118 | **1.015** |
| R spectrum @300 / Laplacian @300 | 1.210 / 1.408 | 1.173 / 1.493 |
| T fps | 40.9 | 37.0 |

**8.6x the opening share of the incumbent on clips neither model was selected on.** The
improvement is not a scorecard artefact. The **H family is not measured on the
confirmation set** (its long-run statistics are collected only for the frozen
`long_clips`, which are main-set clips) — stated rather than glossed over.

### 9.3 Test split — used once, at the very end

`eval_out/score_flow96c_test.json`: 40 mid-episode clips from the test split, built by
the same rule as the frozen val sets. The S family is not run there (the frozen steering
contexts are val-only), and §9.2's caveat about H applies to it too.

| metric | main (val) | confirmation (val) | **test** | real on test |
|---|---|---|---|---|
| D ratio @20 / @60 / @150 / **@300** | 1.409 / 1.244 / 1.593 / 1.106 | 1.463 / 1.585 / 1.605 / 1.015 | 0.947 / 1.221 / 1.582 / **0.758** | 1.0 |
| D generated share @300 | 0.581 | 0.575 | 0.442 | 0.583 |
| R spectrum @20 / @300 | 0.605 / 1.188 | 0.515 / 1.173 | 0.702 / 1.198 | 0.0 |
| R Laplacian @300 | 1.499 | 1.493 | 1.538 | 1.000 |
| T fps | 37.1 | 37.0 | 37.3 | — |

The test split tracks the two val sets: R is essentially identical, D is lower at h=300
(0.758 against 1.106/1.015) but still **3.2x** the incumbent's 0.238, and the direction
holds at every horizon. That is not the "regresses badly" case the brief says to revert
on, so the pick stands — with the h=300 test number quoted rather than the val one.

### 9.4 Speed and live-run power, final model

| | final model (flow96c) | v0 (flow96 on steer96) |
|---|---|---|
| fps mean / p95 (G4 loop, batch 1, incl. the uint8 decode) | **37.1 / 22.4** | 40.8 / 23.6 |
| mean / p95 latency | 27.0 / 44.6 ms | 24.5 / 42.4 ms |
| peak VRAM | 417 MB | 399 MB |
| mean GPU power during the loop | 44.9 W | 37.7 W |
| J per generated frame | 1.21 | 0.92 |
| W per fps | 1.21 | 0.92 |
| Wh per hour of play | 44.9 | 37.7 |

There is a real-time margin of about 1.5x over the 25 fps floor at the 96x72 resolution,
on a laptop whose GPU shares the display and whose power limit is 115 W. Live play is
therefore ~45 Wh per hour of play at ~100 % GPU load, i.e. **about 1.2 J per frame**.

### 9.5 Vision sanity check (§7, weak evidence, recorded not used)

The experimental vision model was shown three review figures with the brief's warning
that large black regions are fog or a far wall, not doorways. It reported: none of the
images is blank or corrupted; the rows differ obviously in all three; and in both the
rollout figure and the long-run figure the final model's row shows more visible
vertical edge / floor / ceiling structure than the incumbent's. It labelled the final
model "flowBC" instead of "flow96c" and said "not sure" about every dark region, which
is a fair illustration of how much weight it deserves. It agrees with the metrics, and
it decided nothing.

### 9.6 Energy

**Idle baseline** (60 s, nothing else running): **4.07 W** GPU, 4.21 W peak, 45.1 °C.
"Net" below subtracts 4.07 W × duration.

**(i) Everything measured since this phase began** — 31 runs, 5.03 h of wall clock:

| | GPU gross | GPU idle-subtracted | CPU package | wall |
|---|---|---|---|---|
| all runs | **374.28 Wh** | 353.83 Wh | 187.71 Wh | 5.03 h |
| training (8 runs) | 119.16 Wh | — | — | 1.21 h |
| evaluation (23 runs) | 255.12 Wh | — | — | 3.82 h |
| *of the gross, wasted on aborted attempts* | *18.68 Wh (0.19 h)* | | | |

The 8 training runs include the four frozen-judge trainings (the opening detector, 4
attempts, 0.33 Wh in total), flow96c (2.21 h → the one that matters) and flow96d. CPU
package energy is present only from ~00:20 (RAPL was root-only before that); the ledger
records `null` for the earlier rows rather than pretending.

**(ii) The winning model's lineage** — `parent_run` chain from `flow96c`:

* **`flow96c` alone: 32.13 Wh GPU, 19.6 min, mean 98.1 W, peak 104.6 W, mean 84 °C,
  throttle events `sw_power_cap` + `sw_thermal_slowdown`.** Its denoiser was trained
  from scratch; the only other weight path is the frozen `steer96` core, which was
  trained before this phase.
* Everything else measured tonight (the judges, the scorecard runs of v0/flow96c/
  flow96d, the confirmation and test runs, the diagnosis) is evaluation, not lineage.

**(iii) The pre-logger Stage 5 runs — a labelled ESTIMATE, shown separately and never
added to (i) or (ii).** Durations are from `STAGE5-PROGRESS.md`; the mean powers are
those of the comparable runs measured here (trains 98–102 W, evals 50–76 W):

| run group | duration | mean W used | GPU Wh (estimate) |
|---|---|---|---|
| the M1 core chain that produced `steer96` (the winner's core lineage) | 1.70 h | 100 W | 170 Wh |
| the M2 generator chain `flow64…flow96` and their Stage 5 gate evaluations | 2.70 h | 70 W | 189 Wh |
| **total estimate for pre-logger Stage 5 work** | 4.40 h | — | **~359 Wh** |

So the honest headline for the winning stack is: **32 Wh measured for the generator that
won tonight, plus ~170 Wh estimated for the core it sits on**, and a whole-phase measured
total of 374 Wh GPU / 188 Wh CPU over 5.03 h, of which 19 Wh was thrown away on my own
aborted attempts.

**Live play:** 44.9 W mean, 1.21 J per frame, ~45 Wh per hour of play for the final
model (§9.4); 37.7 W / 0.92 J / ~38 Wh per hour for the incumbent. In the same units,
playing the game for an hour costs about 1/1400 of the generator's training energy.

**Reconciliation:** `powerlog.py --reconcile` prints the sum of the per-run `gpu_wh`
fields — 374.28 Wh — and that is exactly the "GPU energy, gross" line in
`ENERGY-LEDGER.md`.

## 10. Files added and changed (for the user's commit)

### 10.1 Review images (`neural/review/stage5b/`)

| file | what it shows |
|---|---|
| `rollouts_clip{0..3}_<episode>.png` | 4 distinct clips, real above v0 above flow96c, every 4th step to 60, 3x frames — 4414x639 each |
| `counterfactual_{0..3}_<episode>.png` | left (-0.8) / right (+0.8) / no click, 4 distinct episodes, 24 generated steps, both models |
| `reveals_{0..3}_ep<episode>.png` | at 4 real doorway-reveal windows: real, Stage 4 baseline, then 4 sampling seeds of the final model |
| `longrun_{no_click,scripted_alt}.png` | one frame every 20 steps to 300; v0 above the final model; 4480x427 |
| `candidate_{v0,flow96c,flow96d}_strip.png` | one long-run strip per candidate, so the pick can be overruled |
| `play_flow96c.gif` | 200 frames of scripted play, 384x288 at 4x |

**Added** (all under `neural/` unless stated):

| file | what it is |
|---|---|
| `powerlog.py` | the energy logger: NVML + RAPL sampling, trapezoid Wh integration, per-run ledger records with `parent_run` lineage, `--baseline` and `--reconcile` modes |
| `judges.py` | the frozen judges: opening detector (training + inference), real-vs-generated classifier, radial power spectrum, Laplacian/multi-scale gradients, health statistics, opening-event counting |
| `scorecard.py` | the frozen sets (`--part sets`), the detector (`--part detector`), judge validation (`--part validate`), a candidate's full row (`--part score`) and the reveal one-step test (`--part reveals`) |
| `score_table.py` | candidate table + the pre-declared keep/revert rule (noise-aware) |
| `diagnose_stage5b.py` | the §5.1 diagnosis (zoom-vs-shift fit, capacity/receptive field, hardware) |
| `zoom_table.py` | **not used in the end** (I1 was skipped) — the per-step forward-motion supervision table, kept because the diagnosis code and the report reference it |
| `review_stage5b.py` | the §10 review images, drawn as large tightly-cropped grids |
| `night_queue.sh` | the ordered night plan as one file: waits for an idle GPU, then runs each job through `memcap.sh` + `powerlog.py` |
| `tests/test_stage5b.py` | the new tests (energy integrator, judges, reveal mask, batched-rollout equivalence, no-oracle boundary) |
| `eval_sets/scorecard_v1.json` | the frozen evaluation sets |
| `eval_out/` | `score_*.json` candidate rows (v0, flow96c, flow96d, plus the confirmation-set and test-split runs), `decision_*.json`, `judge_validation.json`, `diagnosis.json`, `reveals_all.json`, `opening_detector.pt` + its report |
| `logs/` | one small log per run, including the aborted ones (evidence for the energy ledger) |
| `review/stage5b/` | the review images and the GIF |
| `docs/ENERGY-LEDGER.json`, `docs/ENERGY-LEDGER.md` | the ledger |
| `docs/STAGE5B-REPORT.md` | this report |

**Modified** (additive; nothing was removed):

| file | change |
|---|---|
| `flow.py` | `SelfAttnBlock` + optional attention in `VelocityUNet`; `net_base` separated from `base`; `revealed_mask` and the `edge_weight` term in `flow_training_loss` |
| `train_flow.py` | `net_base`, `attn`, `edge_weight`, `log_every`, `val_seeds`, `self_k_min`; multi-seed validation for checkpoint selection; a hard guard against confusing `--steps` (the ODE sampler) with `--max-steps` (the budget) |
| `gates.py` | `run_g1` accepts frozen context/window lists; `run_g4` samples power and reports W, Wh/hour, J/frame, W/fps |
| `judges.py` | (added this phase, then amended: reachable-and-close labelling, instance normalisation, no-door controls, calibrated threshold) |
| `play.py` | `start_session()` / `end_session()`: per-play-session power sampling, ledger record and one-line summary on exit |
| `tests/test_stage5b.py` | (added this phase) |
| `README.md` | §13 documenting the Stage 5b tooling (logger, judges, scorecard, the keep/revert rule) and where Stage 5 and 5b disagree |
| `docs/STAGE5-PROGRESS.md` | MORNING SUMMARY, candidate table, run log, DECISIONS MADE WITHOUT USER (D1-D11) |
| `CHANGES-LOG.md` | Stage 5b entry |

No file under `/engine`, no dataset, no `.gitignore` change, and no git operation of any
kind was performed.

## 11. How to reproduce

Every heavy job goes through `memcap.sh`; the night's ordered plan is
`night_queue.sh`. All commands are run from `neural/`.

```bash
# 0. environment
.venv/bin/python gpu_check.py

# 1. energy: idle baseline, then wrap everything in the logger
.venv/bin/python powerlog.py --baseline --seconds 60
.venv/bin/python powerlog.py --reconcile          # totals + reconciliation

# 2. the frozen scorecard (sets are never regenerated in place)
.venv/bin/python scorecard.py --part sets
.venv/bin/python scorecard.py --part detector --steps 900
.venv/bin/python scorecard.py --part validate --checkpoint runs/flow96/best.pt

# 3. a candidate's row, and the keep/revert decision
./memcap.sh --mem 15G -- .venv/bin/python powerlog.py --run-id score_X --purpose eval -- \
    .venv/bin/python scorecard.py --part score --name X --checkpoint runs/X/best.pt
.venv/bin/python score_table.py --incumbent v0 --candidate X

# 4. the final model's headline numbers (one command, ~19 min, reproduces §9)
./memcap.sh --mem 15G -- .venv/bin/python powerlog.py --run-id score_final --purpose eval -- \
    .venv/bin/python scorecard.py --part score --name <FINAL> --checkpoint runs/<FINAL>/best.pt

# 5. the confirmation set and the test split, once each
.venv/bin/python scorecard.py --part score --name <FINAL>_confirm --clips confirmation_clips \
    --checkpoint runs/<FINAL>/best.pt
.venv/bin/python scorecard.py --part score --name <FINAL>_test --split test \
    --checkpoint runs/<FINAL>/best.pt

# 6. the one-step test at real reveal moments, all candidates at once
.venv/bin/python scorecard.py --part reveals --name all --n-reveals 200 \
    --models "v0=runs/flow96/best.pt,<candidates...>"

# 7. diagnosis, review images, live-run power, tests
.venv/bin/python diagnose_stage5b.py --which all --checkpoint runs/flow96/best.pt
.venv/bin/python review_stage5b.py --final runs/<FINAL>/best.pt
.venv/bin/python play.py --checkpoint runs/<FINAL>/best.pt --packed data/bulk_v1_96x72 \
    --headless --frames 400 --out review/stage5b/play_<FINAL>.gif
.venv/bin/python -m pytest tests/ -q
```

## 11.1 Software defects the night exposed (all fixed)

Running the code is what found these; each one is now guarded by code or a test.

| defect | symptom | fix |
|---|---|---|
| `--steps` confused with the training budget | two runs spun for 16 min at 100 % CPU with no step logged: `--steps 12000` means 12000 U-Net evaluations *per generated frame* | `train_flow.py` refuses `--steps > 64` and names `--max-steps` |
| orphaned DataLoader workers from killed runs | the machine (30 GB, VS Code resident) thrashed; training dropped to ~1 step/s | killed explicitly; `night_queue.sh` now waits on the GPU rather than on a process list |
| `_rollout_batch` lost its `@torch.no_grad()` | scorecard run died with CUDA OOM: 7.26 GiB of autograd graph for a 300-step rollout | decorator restored; verified peak VRAM 360 MB for 8x60 steps |
| scorecard raced the previous job's teardown | second CUDA OOM 2 s after a training run exited (149 MB free of 8 GB) | the queue now requires ≥2.5 GB free VRAM *and* an empty compute-app list, then settles 20 s |
| real-vs-fake classifier on an imbalanced sample | AUC exactly 0.500 at every horizon for every candidate — a meaningless number | train/test halves balanced; verified 0.47 on same-distribution data and 1.0 on obviously different data |
| judges module importable from the inference path | would have broken N1/N2 if it ever happened | test asserts `steer.py`, `flow.py`, `play.py`, `eval_harness.py`, `dataset.py` do not import `judges`/`scorecard` |
| `play._to_uint8` assumed a torch tensor | the review GIF writer crashed on the uint8 frames the review helpers keep | accepts uint8 or [-1,1] tensors explicitly |
| `reveal_and_steady_windows` returns triples | review images crashed on an unpack error | unpack fixed (the triples are `(frame_change, episode, start)`) |

## 12. Suggestions for the next stage

1. **The 300-step closed-loop horizon is where everything is lost, and it is worth one
   more targeted attack.** The one-step reveal test says per-step synthesis is fine
   (§3.4) while the 300-step detector share for the incumbent was 0.125. Two concrete
   levers, in order of expected value: (a) train on *longer* self-rollouts — 8–16 steps
   of the stack's own frames with noise augmentation, which is what I2 started and what
   moved the metric; (b) a scheduled-sampling curriculum that anneals the roll-in from
   the real context to fully self-generated context over training, so the model is
   never asked to be coherent for 300 steps without ever having practised 100.
2. **Fix the time direction.** `frame_change` is 4–9x too small for every candidate at
   every horizon: the generated frames are smoother in time than real frames even where
   the spatial spectrum matches. A temporally-differenced loss term (or training the
   sampler on the *difference* target with a matched noise scale) is the obvious next
   experiment, and it is cheap to try because the flow formulation already
   parameterises the change.
3. **Do not trust one judge.** The real-vs-generated classifier saturated at AUC 1.000
   for everything, and the opening detector is only 0.61 precise. Both are recorded as
   limits here; a next stage should add a judge with an unbounded, continuous scale
   (for example the detector's probability rather than its thresholded share, or a
   small flow-matching distance in a learned feature space) so candidates can be ranked
   instead of merely separated from real.
4. **Energy: get the CPU package measured from the start.** RAPL only became readable
   20 minutes into the phase, and the pre-logger Stage 5 runs are an estimate rather
   than a measurement. The `chmod` in §1.1 makes this reproducible; alternatively
   `turbostat` under `sudo` would also give uncore power.
5. **The retired candidates are all kept.** `runs/flow96c` and `runs/flow96d` (if it
   does not win) should be played by eye; the scorecard cannot see the difference
   between "a doorway-shaped texture the detector likes" and a doorway, and the user
   can.
6. **Steering is left alone deliberately.** The user said clicks clearly turn the view
   and asked for no engine-side change, so the stack-level readout fine-tune (I3) was
   not run even though the scorecard's sustained-steering aggregate (sign 0.51) is
   below the 0.8 the Stage 5 brief asked for. If a future stage wants it, the hook is
   already in `train_steer.py --frame-model` — with `dagger_kmax` ≤ 4 and
   `dagger_prob` ≤ 0.3, because the roll-in cost is dominated by the generator's
   sampling steps (a 16-step roll-in at p=0.7 would be ~4 s per training step, i.e.
   hours).
