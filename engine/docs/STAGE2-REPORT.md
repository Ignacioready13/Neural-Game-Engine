# Stage 2 Report — action labels, click-bot personas, resumable runs, pilot

Brief: `stage2-prompt-bots-labels-pilot.txt`. Date: 2026-10-10.

This records what was built for Stage 2 and the findings. The definitive data
contract is [`DATASET-SCHEMA.md`](DATASET-SCHEMA.md); the definitive engine
architecture is the root [`README.md`](../../README.md) and
[`CONTRACTS.md`](CONTRACTS.md). This file is the stage's summary of record.

---

## 1. What was built (Phase B)

| Work item | Where | Notes |
|---|---|---|
| 4.1–4.4 action labels + provenance | `src/core/types.ts`, `src/headless/driver.ts`, `src/headless/manifest.ts`, `src/headless/provenance.ts` | schema v2; `actionX[i] == controlSignalX[i+1]`; final frame `actionX = null`; configHash / engineCommit / dirty / engineSourceHash; deterministic episode dirs + ids |
| 4.6 personas + profiles | `src/headless/bot.ts` | seven personas, three profiles; bot RNG derived from seed, separate from world RNG |
| 4.7 run plans + CLI + resume | `src/headless/plan.ts`, `src/headless/workerPool.ts`, `src/headless/cli.ts` | `runPlan` resumes (skips complete + aborted, deletes/redoes partial), disjoint seed ranges, retry-once, early-stop past `CLIP_STOP_SHARE`; legacy flags kept |
| 4.8 diagnostics report | `src/headless/report.ts` | reads manifests + `aborted.jsonl` only (never decodes PNGs); writes `report.json` + `report.md` |
| 4.9 doc fixes | `README.md`, `CHANGES-LOG.md` | see §6 |
| 4.10 pilot + review images | `engine/data/pilot_v1/` | §5 |
| 4.11 clipping failsafe | `src/headless/driver.ts` | L1 wallOverlap, L2 ≥10 slide frames, L3 report-only; trim 8 s / keep ≥60 frames; `aborted.jsonl` |
| 4.5a overlay | (no change needed) | the click pulse never reaches captured frames — see §3(a) |

Simulation behavior (world generation, steering, reachability, debounce,
collision, timing, RNG) was **not changed**.

## 2. Section 3 finding (signal path)

The 2.6 claim that "every frame fell in three `controlSignalX` bins (0 / +1 /
−1)" **does not reproduce**. Re-running the real headless pipeline with the real
placeholder bot (uniform X every 20–80 frames, signal held between clicks) gives
a held-signal histogram spread across all 0.1 bins, and an accepted-click
histogram spread across all 20 bins — exactly what the held-signal model
predicts (flat, with a spike at 0 only before the first click). The signal path
(click → normalized X → held signal → sticky target) is correct; the old number
was an artifact of the earlier investigation, not an engine bug. No engine
change was made.

## 3. 4.5 findings (frame cleanliness)

- **(a) Overlay** — the click pulse is **not** in captured frames, and no change
  was needed. `Renderer.render` draws pulses to the default framebuffer *after*
  `ResolutionPipeline.blit`, and `captureFrame` reads `pipeline.readPixels()`,
  which reads the downsampled scene target (`outputTarget`) captured before the
  pulse pass. A paired pixel probe over 400 click events (frame *i+1* vs *i* at
  the disc centre vs a control band) gives disc-minus-control Δlum = **−0.43**
  (and Δ(R−B) = +0.12), i.e. the pulse's signal is absent. (An earlier draft of
  this report incorrectly claimed the pulse was baked in and removed; that was
  wrong — the 2026-10-08 `Renderer.ts` diff was only the `EntityLights` type.)
- **(b) Lag** — the headless loop is fixed-timestep and nothing in the
  capture/render path depends on wall-clock time (no rAF timing, timers,
  `performance.now`, or async loading). A double-render at entity-generation
  events is byte-identical, i.e. no shader-compile or half-uploaded frame
  reaches the data.
- **(c) Continuity** — consecutive-frame difference shows no spike at entity
  changes beyond the intended second-degree-doorway black-fill → content swap;
  the transition image is continuous. No rendering change.

## 4. Diagnostics (4.8), pilot v1

Pilot: `engine/data/pilot_v1/` — **150 episodes** (120 train / 15 val / 15 test),
600 frames each, mixture profile, concurrency 8. **145 complete + 5 aborted
(3.3%), 0 failed**, 88,047 frames, 6.70 GB, **130 s** wall-clock on the RTX 5070
(NVIDIA Vulkan confirmed in the episode logs). `report.json` and `report.md` are
written into the pilot directory. **Regenerated in place 2026-10-11** so every
frame carries the `viewUnsafe` label (§4b); the regeneration is behaviour-neutral
and produced the identical dataset (same episodes, aborts, frame counts and
bytes), with only the new field added.

| Item | Result |
|---|---|
| 2. Signal | accepted clicks spread across all 20 bins; held signal spread (spikes at 0 and ±1 are the never-clicked default + prober/spammer informative clicks); `actionHasClicked=false` in **9.8%** of frames |
| 3. Reachable | histogram 0:5929, 1:41359, 2:23343, 3:11647, 4:4656, 5:930, 6:183; **46.3%** of frames have 2+ reachable |
| 4. Straightest-target (per-frame, 2+ reachable) | **88.7%** overall — see §5 for why this is the §3(b) confound |
| 4b. Straightest-**choice** (accepted clicks, 2+ reachable) | **48.8%** overall (decisive 39.9, dither 23.4, late 58.8, prober 21.7, spammer 57.1, uniform 64.0) |
| 5. Intent-honored / fallback | intent-honored **91.7%**; engine fallback **9.7%** |
| 6. Label ambiguity | **11.9%** of frames |
| 7. Collision | **0.2** events / 1,000 frames; **0** wall-overlap frames |
| 8. Per-persona frame share | decisive 31.3%, uniform 24.5%, prober 13.7%, idle 9.6%, late 9.5%, dither 6.3%, spammer 5.0% |
| 9. Integrity | 0 incomplete episodes; no seed shared between splits |
| 10. Clipping | 5 aborted (one each decisive/idle/late/spammer/uniform); 676 frames trimmed; brief-graze (L3) 0.0% |
| 11. View safety | `viewUnsafe` (label-only, margin 0.10) — **357 frames (0.41%)**, **90 / 150 episodes (60.0%)** — see §4b |
**Replay caveat (seed 78).** The diagnostic replays were renderer-free (world +
steering only, no browser). One episode, `seed 78`, diverges from the real engine
after f91 at a debounce boundary, so a renderer-free replay is not bit-exact
there; **any later human-session replay must go through the real headless
engine**, not a renderer-free harness.

## 4b. View safety (`viewUnsafe`) — the 4.11 follow-up

The read-only diagnosis (Check-in 3) found frames where the camera — a floor
point with **no collision radius** — sits close enough to a rendered wall that
the view is compromised, by one of two purely *rendering* mechanisms:

1. **Near-plane clip** — a surface nearer than the camera near plane (**0.02**,
   unchanged) is clipped away, leaving a hole (e.g. `seed 0 f133`, distance
   0.0196).
2. **Single-sided see-through** — walls are front-side only, so a wall whose
   front face points away from the camera is culled; if the camera has crossed
   that plane you look straight through it (e.g. `seed 111 f175`).

Neither is caught by the clipping failsafe (L1 `wallOverlap` fires only when the
player *leaves the floor*; here the player stays on it), and the smallest fix is
not a simulation or render change — it is a **label**. Every frame now records
`viewUnsafe` (`src/sim/viewSafety.ts`): true when the nearest rendered surface of
the active entity or a maintained neighbour is nearer than
`camera near + VIEW_UNSAFE_MARGIN` (**0.10** ⇒ 0.12), or when that nearest
surface is back-facing with geometry in front of the camera (the see-through
case). It is **label-only** — it never aborts, trims or otherwise changes an
episode. Loaders must drop any training window that touches a flagged frame.

Margin chosen (0.10) so that the whole `seed 0` h2→h4 doorway graze flags —
f129–f135 all flag, spanning the f132 (0.059) / f133 (0.020) / f134 (0.034)
frames named at the check-in, while f128 and f136 stay clear. Measured over the
regenerated pilot:

| | flagged | share |
|---|---|---|
| frames | 357 / 88,047 | **0.41%** — under the ~1% budget, no tightening needed |
| episodes | 90 / 150 | **60.0%** |

The frame rate is the number the ~1% budget applies to, and it passes. The
episode share is high only because the flag is per *frame*: 72 of the 90 flagged
episodes have **1–4** flagged frames (a sub-0.2 s doorway graze); the tail is
`seed 55` (38 frames, two long wall-slide runs) and `seed 78` (24). The pattern
matches the diagnosis exactly — the diag's independent geometric count over the
149 non-diverged episodes (333 frames at the same threshold) plus `seed 78`'s 24
equals the 357 the report measures.


## 5. The straightest-target finding (4.6 success criterion)

The literal per-frame criterion ("among frames with 2+ reachable, share where the
current target is the straightest candidate") is **88.7%**, above the ~65%
threshold. That number is misleading, and I did **not** apply the one-time
persona tuning + regeneration, for a reason the brief itself anticipated in
§3(b):

- Once a bot commits to a door and steers toward it, that door **becomes** the
  straightest (minimum |screenX|) door on the next frames — so "current target
  == straightest" is true for almost every post-commitment frame *regardless of
  which door was chosen*. The `uniform` persona (uniform random X, the
  maximally-spread choice) still reads **90.2%** per-frame, which proves the
  per-frame metric tracks "how often the target is the door currently ahead",
  not "how central the choice was". Tuning weights cannot drive it below ~65%
  without gamifying the metric.
- The meaningful measure — the §1 "moments of real choice" — is
  **straightest-choice at accepted clicks with 2+ reachable: 48.8%** (≈1/n, i.e.
  uniform). Per persona: decisive 39.9%, dither 23.4%, prober 21.7%. This is
  "far below the old 90–93%" and meets the intent of the criterion.

**Decision needed (see §7):** accept the choice-moment measurement as the
criterion, or request the literal tune-once + regenerate anyway.

## 6. Doc fixes (4.9)

- `README.md`: removed the non-existent `DOORWAY_TURN_GATE_RADIUS` (1.5) config
  row; corrected bang-bang → P-controller; "h0" → "h1"; `/data` → `engine/data/`;
  corrected the "two-phase turns / gate radii" wording to match the code (only
  bends set `turn`/`approachPoint`); added the Stage 2 CLI, bot framework,
  failsafe, report, and v2 manifest documentation.
- `CHANGES-LOG.md`: added this stage's entry; added a dated correction note to
  the 2026-10-05 "per-doorway gate radii" description.

## 7. Review images (4.10)

Prepared for your visual judgment (I have not judged them myself):

- `engine/data/pilot_v1/_review/legibility.png` (1238×3066) — 12 representative
  frames, each at 320×240 and box-downsampled to 96×72 and 64×48 then
  nearest-enlarged so pixels are visible. Covers: close wall, bend, 4-way/3-way
  junction, view through a doorway into a room and into a hallway, two room
  interiors, a long hallway, a hallway toward a centered door, and two
  2-reachable hallways (second-degree doorway).
- `engine/data/pilot_v1/_review/transitions.png` (2220×1290) — 5 entity
  transitions (h→h, h→r, r→h), 6 consecutive frames around each.

Category labels are honest best-effort heuristics from manifest metadata
(`currentEntityId` + reachable-candidate count); tall vs low room ceiling and
the exact "blacked-out doorway" state are not recorded in the manifest, so two
room interiors are shown without a tall/low claim.

The pilot was regenerated in place (§4b), but the regeneration is
behaviour-neutral and reproduced identical episodes/frames, so these images still
correspond to the same frames on disk.

## 8. Shortcuts taken

- The report and review-image selection read manifests/`aborted.jsonl` only; no
  PNG decoding in the report, and review images are composited in the existing
  headless browser (2D canvas, no new dependency).
- Review frames are chosen by manifest heuristics rather than re-generating the
  world to obtain exact entity types/ceiling heights.
- No persona tuning + pilot regeneration (see §5); the criterion was measured at
  choice moments instead.
- Legacy CLI flags are preserved by folding them into an implicit plan.

## 9. Files changed / added (Stage 2)

Added: `src/headless/plan.ts`, `src/headless/provenance.ts`,
`src/headless/report.ts`, `src/sim/viewSafety.ts` (the `viewUnsafe` probe),
`docs/DATASET-SCHEMA.md`, `docs/historical/README.md`, `docs/STAGE2-REPORT.md`.

Rewritten: `src/headless/workerPool.ts`, `src/headless/cli.ts`.

Modified: `src/headless/bot.ts`, `driver.ts`, `manifest.ts`, `api.ts`,
`src/core/types.ts`, `config/config.ts`, `src/render/Renderer.ts`,
`src/render/scene.ts`, `src/sim/Engine.ts`, `tests/headless.test.ts`,
`README.md`, `CHANGES-LOG.md`.

The working tree is intentionally left dirty (no commit, per rules of
engagement); it also contains pre-existing uncommitted rendering changes from
the 2026-10-08 work that Stage 2 did not touch.

## 10. Verification

`tsc --noEmit` clean; **56/56** vitest tests pass (action-alignment rule, run
plans + disjoint seed ranges, provenance determinism, clipping-failsafe trim and
delete paths). Determinism and resume were confirmed: re-running an episode is
byte-identical, and resume skips completed + aborted episodes while deleting and
redoing partials. The `viewUnsafe` label was checked end to end: the `seed 0`
smoke episode flags exactly f129–f135 (spanning the f132/f133/f134 the check-in
named) and leaves f128/f136 clear, and the regenerated pilot's flagged frames
reproduce the diagnosis's independent geometric count (§4b).

## 11. Follow-up status (all check-in decisions resolved)

1. **Straightest-target criterion** — accepted: the choice-moment rate (48.8%)
   is the criterion; no persona tuning, no regenerating for it.
2. **Review images** — `_review/legibility.png` accepted; the `_review/transitions.png`
   row `h2->h4 @ seed 0 f133` is the clipping graze, which drove the `viewUnsafe`
   label in §4b.
3. **View safety** — implemented as the label-only `viewUnsafe` (§4b), pilot
   regenerated in place, report/README/schema updated. No simulation or render
   change (near plane still 0.02).
4. **Next step**: per the brief, **WAIT** — there is no further Stage 2 work to do.
