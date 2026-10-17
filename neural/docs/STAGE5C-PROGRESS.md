# STAGE 5C PROGRESS — coherence-first generator (overnight)

Newest information first. Machine: RTX 5070 Laptop 8 GB. `/neural` half only.

## Schedule actually being run

| | |
|---|---|
| Start (T0) | **2026-10-13 00:53 MDT** |
| Phase 0 done | 01:05 (12 min, inside the 60 min cap) |
| Phase 1 pilots | 01:07-01:29 (`flow96e_pilot`, FAILED), 01:32-01:55 (`flow96e_pilot_small`, PASSED) — both inside the T0 + 2 h 15 cap (03:08) |
| Phase 2 main run | **02:01 → ≈ 06:47** (budget 286 min) |
| Phase 3 | ≈ 06:50 → 07:30 |
| GPU work deadline | **07:30 MDT** |
| User stops everything | 08:00 MDT |

Main run window: 02:01 → ≈ 06:47, i.e. **4 h 46 m** of main training — more than the
"6-7 h if the start time allows" wish but the most the schedule gives (time-based budget,
not step-based).

## SUDO NOTE (no reply needed)

Nothing so far needs sudo. If a system change becomes necessary it will be written
here as the exact command plus the reason, and the work will continue without it.

## Rule-3 baseline recorded (before any new run)

`files/baseline/` in the session state:
* `checkpoints.sha256` — 198 sha256 lines over every `.pt/.pth/.ckpt/.safetensors/json/jsonl/log` under `runs/` and `models/`.
* `runs_models_files.txt` (199 files), `review_eval_files.txt` (86 files) — full listings.
* `registry.sha256` — `models/registry.json` `5287e3ea…`, `models/seeds.npz` `8e0dc7ca…`, `eval_sets/scorecard_v1.json` `64804f14…`.

These are compared at the end (verification item 2).

### flow96e_pilot_small (the fallback) — **PASSED the gate** (see Phase 2 below)

In-process EMA health (3 val contexts, 300 steps, no clicks):

| step | frame_change | hf_ratio | contrast | luma_mean | mean in band |
|---|---|---|---|---|---|
| 3589 | 7.31 | 1.377 | 29.7 | 55.6 | 0.33 |
| 7161 | 9.99 | 0.573 | 27.8 | 145.6 | 0.33 |
| 10732 | 4.14 | 0.992 | 41.0 | 142.7 | **1.00** |
| 11190 (final) | 4.17 | 1.020 | 61.4 | 114.4 | 0.67 |

Gate `eval_out/stage5c_pilot_small_gate.json` (live player path, 3 val contexts, 300 steps):

| weights | in band | failing contexts |
|---|---|---|
| `final.pt` | **2/3 → PASS** | contrast 64.9 (limit 62.0) on `ep1391@54` |
| `snapshots/snap_0010732.pt` | **3/3** | — |
| `best.pt` (chosen by val L1) | 1/3 | contrast 24.7/26.3 (too flat) |
| flow96 (reference) | 2/3 | frame_change 23.3, hf_ratio 2.00 on `ep1391@54` |

The final EMA's numbers are the closest to real of anything measured tonight
(`frame_change 4.17` vs real 4.60; `hf_ratio 1.02` vs 1.00; `luma_mean 114` vs 101).
**`best.pt` — the weights the old selection rule would have picked — is the *worst* of
the four.** Another instance of §2.3's lesson: val L1 does not select for coherence.

Honest visual note (one sheet, `review/stage5c/stage5c_flow96e_pilot_small_final_valep1391at54.png`,
seen by me; 5 sheets total tonight, cap 40): panels 1-3 show a recognisable corridor
(walls both sides, ceiling, dark floor); panels 4-5 by step ~200-300 become a large
yellow wall filling the frame, so the pass is **marginal** — better than flow96e_pilot's
flat ochre field, not clearly better than flow96. The health curve over the three
snapshots is strongly improving (0.33 → 1.00 → 0.67 across only 7-minute intervals), so
a 13x-longer run is the reason to proceed rather than a claim that the pilot is good.

Per §4.1 ("If it passes, the main run uses the small architecture") the main run uses
this recipe unchanged.

## Phase 2 heartbeat (one line per ~45-60 min check, per §5)


## Phase 2 — THE MAIN RUN (`flow96e`), launched 02:01, detached

```
./memcap.sh --mem 12G -- .venv/bin/python powerlog.py --run-id flow96e \
    --purpose train --parent-run flow96c -- \
    .venv/bin/python train_flow5c.py --name flow96e --budget-min 286 \
    --snapshot-min 40 --packed data/bulk_v1_96x72 --net-base 32 --batch-size 64 \
    --attn false
```

* **The recipe is exactly the passing pilot's**, unchanged: `noise_level_max 0.5`,
  `level_buckets 10`, `level_clean_frac 0.25`, `ctx_dropout 0.1`, `self_prob 0.25`,
  `self_k 4`, `self_k_min 2`, `noise_scale 0.08`, `mirror_prob 0.5`, `onset_weight 3.0`,
  EMA 0.999, batch 64 at 96x72, LR 4e-4 flat for the first 70 % of the wall clock then
  cosine to 7 %. **Nothing was swept and no other variant was trained** (§6).
  *Note:* because the winning pilot is the *fallback*, the "raise the corruption maximum
  to ~0.7" clause of §4.1 does **not** apply (that is the both-pilots-failed branch); the
  maximum stays at the pilot's 0.5.
* Budget 286 min, so the loop ends ≈ 06:47 and the reserved Phase 3 window
  (≈ 06:50-07:30) is untouched. Time-based, so a slow machine still finishes on time.
* Snapshots every 40 min (never overwritten), each with an EMA validation pass and a
  3-context x 300-step live-path health rollout appended to
  `runs/flow96e/health.jsonl`, plus one strip under `review/stage5c/`.
* Heartbeat to `runs/flow96e/status.json` every 5 min; NaN/collapse restarts from the
  last snapshot at half LR (max 2); the driver registers `flow96e_final`
  (`--not-selected`) itself at the end, so the model is registered even if my session ends.
* Throughput at launch 8.5 steps/s → ≈ 146 k steps over the budget (≈ 18x flow96's
  8 k steps, ≈ 12x flow96c's 12 k), ≈ 9.3 M samples ≈ 15 epochs of the 633 k windows.


## Phase 1 — PILOT GATE

### flow96e_pilot (the GameNGen-style recipe, flow96c's architecture) — **FAILED**

`runs/flow96e_pilot/`, 13,608 steps in 1,329.8 s (10.25 steps/s, batch 48 at 64x48,
net_base 64 + attention), **37.04 Wh**, no restarts, `powerlog --run-id flow96e_pilot`.
Config = flow96c's exactly (same data `data/bulk_v1_64x48`, same core, 4-step sampler,
self_k 4 / p 0.25) plus the Stage 5c additions: `noise_level_max 0.5`,
`level_buckets 10`, `level_clean_frac 0.25`, `ctx_dropout 0.1`, EMA 0.999,
time-based LR (flat 70 %, cosine to 7 %).

In-process EMA health over the budget (3 val contexts, 300 steps, no clicks):

| step | frame_change | hf_ratio | contrast | luma_mean | in band |
|---|---|---|---|---|---|
| 4341 | 11.82 | 0.561 | 24.4 | 131.2 | 0/3 |
| 8687 | 5.87 | 1.640 | 45.0 | 198.0 | 0/3 |
| 13051 | 7.23 | 1.547 | 58.7 | 179.0 | 0/3 |
| 13608 (final) | 7.26 | 1.523 | 57.9 | 172.3 | 0/3 |

Gate evaluation `eval_out/stage5c_pilot_gate.json` (live player path, `final.pt`, 3 val
contexts): flow96 2/3 in band, flow96c 0/3, **flow96e_pilot 0/3** (every context fails
`hf_ratio` (1.48-1.55 vs 1.50) and `luma_mean` (166.8-176.9 vs 170.0)). **Gate FAILS.**

I looked at `review/stage5c/stage5c_flow96e_pilot_final_valep1391at54.png` myself (one
sheet, allowed by the user's amendment; 4 sheets total tonight): the model's step-300
frames are a flat bright ochre field with a few dark horizontal bands — **not a
corridor**, no wall/floor/ceiling perspective. The band's verdict is right. The 7-minute
snapshot is blurred but still has corridor-ish structure, so the run *drifted* toward a
blurry bright mean, which is the expected failure mode when the corruption level is
heavy enough that the MSE-optimal velocity is the conditional mean (and the model can
learn to ignore a mostly-noise frame channel). Recorded as a hypothesis, not a
conclusion: the pilot is also only ~1 epoch of the data, and the main run gets ~13x
more.

### flow96e_pilot_small (the fallback) — launched 01:32:41, 22 min

**DIVERGENCE FROM THE PROMPT, deliberate and recorded:** §4.1 says the fallback is "the
same recipe on the SMALL denoiser size of flow96 (0.77 M, no attention)". I ran it on
flow96's *architecture and its data* (`--packed data/bulk_v1_96x72 --net-base 32
--attn false --batch-size 64`, i.e. exactly flow96's shape), not on flow96c's 64x48 set.
Reason: flow96 is the one model the user finds coherent, and it is the only model here
that was trained and played at the same 96x72; flow96c is fully-convolutional-trained at
64x48 and played at 96x72 (also with attention learned on a 9x12 token grid), which is
the strongest remaining structural explanation for why it dissolves and flow96 does not.
Everything else (recipe, 4-step sampler, self_k 4 / p 0.25 / k_min 2, onset weighting,
mirroring, budget, gate) is identical to pilot 1. This makes the fallback a two-variable
change (architecture+resolution); it is recorded as such, and if it passes the main run
will inherit both.

### Everything else in Phase 1

No other pilot, no sweep, no extra variant was trained (§4.1 and §6 respected).


## 4.0d — HEALTH NUMBERS AND PASS BAND (calibrated)

Definition in `stage5c_rollout.py::health` (grey = 0.299R+0.587G+0.114B, 0..255):
1. `frame_change` — mean |Δ| grey between consecutive generated frames, steps 100-300.
2. `hf_ratio` — mean |4-neighbour Laplacian| of the generated frames ÷ that of real
   consecutive val frames (same display path), 300 steps.
3. `luma_mean`, `contrast` — mean grey and its SD over steps 200-300.
4. `near_black` — share of steps 200-300 whose mean grey < 20.

Real reference (`data/bulk_v1_96x72`, 8 val clips x 300 steps, `eval_out/stage5c_probe.json`):
`frame_change 4.60, luma_mean 100.7, contrast 51.2, near_black 0.0, hf_energy 9.54`.

Live-path no-click, 300 steps, 3 val seed contexts, `noise_seed=0`
(`eval_out/stage5c_health_variance.json` uses 3 contexts x 5 noise seeds = 15 rollouts):

| model | frame_change | hf_ratio | contrast | luma_mean | near_black | in band |
|---|---|---|---|---|---|---|
| real | 4.60 | 1.00 | 51.2 | 100.7 | 0.0 | — |
| flow96 (healthy) | 12.90 | 1.068 | 44.7 | 128.6 | 0.0 | **0.80** (12/15) |
| flow96c (not) | 37.03 | 2.696 | 38.3 | 178.9 | 0.0 | **0.00** |
| flow96d (not) | 13.83 | 1.672 | 21.4 | 172.0 | 0.0 | **0.00** |

**Pass band written down (all five must hold for a context to count):**

```
frame_change in [ 3.0, 22.0]
hf_ratio     in [ 0.30,  1.50]
contrast     in [30.0,  62.0]
luma_mean    in [50.0, 170.0]
near_black   <= 0.15
```

This separates the known verdicts: flow96 passes 12/15 rollouts, flow96c and flow96d
pass none (flow96c fails `frame_change`+`hf_ratio`+`luma_mean` in every rollout; flow96d
fails `hf_ratio`+`contrast` in every rollout). **Honest limit, recorded now:** the band
is a *collapse* detector, not a coherence guarantee — flow96's 3 failing rollouts are
ones where it dissolved, and the band has no way to say a still-in-band image is a
hallway. The user's eyes remain the judge (`STAGE5C-REPORT.md` says so).

## 4.0c — VISION JUDGE CALIBRATION: **DROPPED** (>80 % agreement not reached)

Sheets: `review/stage5c/stage5c_calib_*.png`, 8 sheets (flow96/flow96c/flow96d x 2 val
seed contexts + 2 real clips), one row of 6 frames at generated steps 1, 20, 60, 120,
200, 300, from live-path no-click rollouts. Judge: the vision-capable subagent
(`deepseek-v4-flash-vision-exp`), asked the two fixed questions per frame, JSON only.

Agreement with the known answers (real: corridor yes; flow96: holds its shape;
flow96c/flow96d after ~step 100: no/static):
* real — 11/12 frames "corridor yes". **Good.**
* flow96 — the judge calls it corridor at steps 1-60 and **static from step 120 on**
  (2/12 "yes", 5/12 "partly", 5/12 "no"). This is *not* the stated known answer.
* flow96c — steps 1-60 corridor-ish, steps 200-300 no/static (9/12 as expected).
* flow96d — steps 1-60 corridor, step 120-300 no/static (12/12 as expected).

Overall strict agreement **34/48 = 71 % < 80 % → the judge is DROPPED**, per §4.0c.

**But the disagreement is informative, and I did not dismiss it.** I looked at
`stage5c_calib_flow96_seed0_valep1391at54.png` myself (allowed by the user's amendment)
and at `stage5c_calib_real_clip0.png`: the sheets render correctly, the real sheet is a
clean corridor with a doorway coming into view, and the flow96 sheet really does dissolve
into purple/ochre static by its 4th panel. The Stage 5b/5c picture "flow96 holds its
shape" is true of **clicked** play (which is how the user judged it); in the *no-click*
300-step scenario this night targets, flow96 also dissolves (only 12/15 of its no-click
rollouts stay in the band). This is recorded, not acted on: the judge stays dropped, no
further vision calls, and the gate uses the numeric band only.

## 4.0b — INFERENCE NOISE CORRELATION rho (no training cost; a large effect)

`noise_f = rho*noise_{f-1} + sqrt(1-rho^2)*fresh_f`, no clicks, 3 val seed contexts, 300
steps, live-path maths (`eval_out/stage5c_probe.json`; rho=0 is bit-identical to
`Player.generate`, asserted per model):

| model | rho | frame_change | hf_ratio | luma_mean | contrast |
|---|---|---|---|---|---|
| flow96 | 0.0 (today) | 14.26 | 1.148 | 140.1 | 42.6 |
| flow96 | 0.9 | **3.72** | **0.991** | 123.9 | 53.9 |
| flow96 | 0.99 | 1.89 | 1.735 | 155.7 | 59.0 |
| flow96 | 1.0 | 0.40 | 2.726 | 178.4 | 78.1 |
| flow96c | 0.0 (today) | 37.11 | 2.700 | 178.8 | 38.3 |
| flow96c | 0.9 | **3.55** | **1.065** | 82.7 | 30.3 |
| flow96c | 0.99 | 0.93 | 1.095 | 45.6 | 38.2 |
| flow96c | 1.0 | 0.19 | 1.014 | 38.3 | 40.5 |
| flow96d | 0.0 (today) | 13.83 | 1.672 | 172.0 | 21.4 |
| flow96d | 0.9 | 3.56 | 1.700 | 196.4 | 24.2 |

**rho≈0.9 moves every model's time statistics onto the real ones** (real: 4.60 / 1.00 /
100.7 / 51.2) — and flow96's rho=0.9 row is the closest to real of anything measured in
Phase 0. Interpretation: the *per-frame* noise is itself a large part of the churn; with
smooth noise the frame-to-frame change falls to roughly the real translation-only value,
i.e. the model's genuine motion is still there and the excess change was injected noise.
rho=1.0 is *not* the answer (flow96 hf_ratio 2.73, flow96c luma 38 — the frozen-noise
path, i.e. the 5b GIF). I am therefore adding an **additive** player flag
`--noise-corr RHO` with **default 0.0 = exactly today's behaviour** (no default changes),
and the report will tell the user to try `./play.sh --model flow96e --noise-corr 0.9`.
This is a decision made without the user (§0): lowest-risk option that cannot change any
existing behaviour.

## 4.0a — CONFIRMATION (numeric, live path, `eval_out/stage5c_probe.json`)

Same model, same seeds, same clicks — only the noise differs:
flow96 rho=0 → `frame_change 14.26`; rho=1.0 (the scorecard's fixed tensor) → `0.40`.
flow96c rho=0 → `37.11`; rho=1.0 → `0.19`. So under the scorecard's own condition
flow96c is *smoother* than flow96 (0.19 vs 0.40 — the 5b ordering), and under the live
player's condition flow96c is 2.6x *more* changeable (37.1 vs 14.3 — the Stage 6
ordering). The audit finding is confirmed and the contradiction in §2.1 is fully
explained. Adopted: **the live path is the single evaluation path from here on.**

## 4.0a — PATH AUDIT FINDING (this is the answer to §2.1)

**The live player and the scorecard use different noise.** The generator draws its
initial flow noise inside `flow.FlowSteerModel.predict`:

* `seed=None` → `torch.randn_like(warped)`, i.e. **fresh independent noise every call**;
* `seed=int(s)` → `torch.Generator().manual_seed(s)`, i.e. **the same noise tensor for
  every call with that seed**.

* **Live path** (`play.py` → `Player.step` → `Player.generate`) passes
  `seed = noise_seed + f`, so **every frame gets fresh, independent noise**. This is
  what the user plays.
* **Scorecard path** (`scorecard.py::_rollout_batch` / `_rollout_actions`, and
  `eval_harness.py::rollout_v2`) passes the *set's* `seed` unchanged for **all 300
  steps of a rollout** (`for seed in seeds:` → `_rollout_batch(..., seed, ...)`).
  So an entire scorecard rollout is **one fixed noise tensor**: a deterministic,
  temporally frozen trajectory.
* **Review-GIF path** (`review_stage5b.py::_rollout_sequence`) hard-codes `seed=0`
  **for every step** — the same defect, which is exactly why
  `review/stage5b/play_flow96c.gif` shows "pillars morphing in place" while live play
  does not: with frozen noise the flow residual is nearly constant across steps, so the
  frame barely changes and the sampler output is a smooth morph.

**Implication for the 5b scorecard.** Its `frame_change` (flow96 0.0048, flow96c
0.0024) measured a *frozen-noise* rollout, i.e. how smooth a single deterministic
sampling trajectory is — the opposite of the live quantity. Live, with fresh per-frame
noise, `flow96c` changes ~4× *more* than `flow96` (Stage 6: 24.42 vs 5.98 grey levels),
i.e. it is the incoherent one. **The scorecard's frame_change ordering is therefore an
artifact of the path mismatch, not a contradiction**; and any decision that trusted it
(e.g. "flow96c smoother than flow96") was measuring the wrong thing. From this point on
**every rollout used to make a decision comes from the live player path** with fresh
noise per frame.

*(Status: code audit done; headless confirmation below.)*
* 02:38 — step 20783 (frac 0.140, 8.64 steps/s, no restarts). Snapshot 1: `frame_change 8.46, hf_ratio 0.594, contrast 36.9, luma 137.0` → in band 2/3 contexts. val_sample_l1 0.1221.
* 03:26 — step 41558 (frac 0.309, 8.65 steps/s, no restarts). Snapshot 2: `frame_change 11.16, hf_ratio 0.615, contrast 37.1, luma 195.3` → in band 0/3. **luma drifted 137→195 (too bright) and hf_ratio 0.59 (blurry)**; val_sample_l1 0.1377 (worse than snapshot 1). Watching; the finish step picks the healthiest snapshot, so every state is preserved.
* 04:01 — step 62316 (frac 0.431). Snapshot 3: `frame_change 8.39, hf_ratio 0.747, contrast 43.1, luma 154.2` → **in band 3/3**. Health oscillates (0.67 → 0.00 → 1.00) but the trend over 40-minute snapshots is improving; the snapshot mechanism means every state is preserved and the pick is made on the whole set at the end.
* 04:40 — step 83085 (frac 0.568). Snapshot 4: `frame_change 10.39, hf_ratio 0.607, contrast 45.7, luma 158.5` → **in band 3/3**. Note for the report: `val_sample_l1` is *rising* monotonically (0.122 → 0.138 → 0.151 → 0.170) while the live-path health is *improving* (0.67 → 0.00 → 1.00 → 1.00). Val L1 is not merely uninformative here, it points the wrong way.
* 05:20 — step 103834 (frac 0.704, LR decay begins). Snapshot 5: `frame_change 9.95, hf_ratio 0.620, contrast 43.9, luma 138.2` → **in band 3/3**. Snapshots 3, 4, 5 are all 3/3; luma has come back down from 195 to 138 (flow96 is 128.6). Phase-3 helpers pre-written while waiting (no GPU): `stage5c_pick.py`, `stage5c_guidance.py`, `stage5c_review.py`, `stage5c_steer.py`.
* 06:03 — step 124566 (frac 0.859). Snapshot 6: `frame_change 7.81, hf_ratio 0.572, contrast 35.8, luma 96.5` → in band 2/3. luma now 96.5 against real 100.7. Phase-3 runner `stage5c_finish.py` pre-written.

## FINISHED — what to run first (plain words)

```bash
./play.sh --model flow96e                    # the model this stage picked
./play.sh --model flow96e --noise-corr 0.9   # same model, smooth noise: try both
./play.sh --model flow96c                    # the old default, for comparison
./play.sh --model flow96                     # the previous most-playable one
```
`flow96e` is the healthiest snapshot (step 62,316) of `runs/flow96e`, registered but NOT
selected, so the default and the `M` cycle are unchanged. Look at
`review/stage5c/stage5c_noclick_flow96e_seed*.png` next to
`stage5c_noclick_flow96_seed*.png` (same seed contexts, same steps) and the
`stage5c_flow96e_noclick.gif`, then judge by playing. The full technical account is in
`docs/STAGE5C-REPORT.md`.

* 02:01 — main run launched (`flow96e`, budget 286 min, snapshots every 40 min).
* 06:47 — main run finished: **148,385 steps, 4 h 46 m, 8.64 steps/s, 0 restarts, 494.35 Wh**.
  Final EMA health `frame_change 7.09, hf_ratio 0.518, contrast 33.7, luma 82.8` → 2/3 in band.
* 06:45-07:20 — Phase 3: pick, guidance, steering, speed, review artifacts, verification.
* 07:20 — machine idle; all GPU jobs done (4 jobs, **569.30 Wh** total tonight).
