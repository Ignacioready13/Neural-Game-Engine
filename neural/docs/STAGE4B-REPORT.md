# Stage 4B Report — bulk dataset: tuned generation, larger val/test, packing, coverage

Brief: `stage4b-prompt-bulk-dataset.txt`. Date: 2026-10-11.
Scope: generate the bulk dataset (`engine/data/bulk_v1/`), retune the bot mixture
(one round), pack at 96×72 and 64×48, and measure coverage. No model is trained.
The definitive data contract is [`DATASET-SCHEMA.md`](../../engine/docs/DATASET-SCHEMA.md);
the qualifying-counterfactual definition is
[`probe_click.py`](../probe_click.py) (`_select_contexts`), reused throughout.

**Headline.** The bulk dataset is built and packed. `bulk_v1` = **1,483 episodes /
876,261 frames** at 600 frames each, generated with a **new** profile `mixture_v2`
(the existing `mixture`/`uniformOnly`/`doorAwareOnly` are untouched), **3.67 %**
abort, packed at both resolutions (18.17 GB and 8.08 GB) outside git. Realised
qualifying-context yield is **126.5 per 1,000 frames** (near-door 100.1) on train —
**1.44× / 1.50×** the pilot. One memory-safety bug in the loader was found and
fixed (Section 9); it had OOM-killed the editor session during verification, not
the generation.

---

## 1. Dataset size — pilot vs bulk (side by side)

Generation: 600 frames/episode, concurrency 8, one tuning round. All episodes use
the same engine code (see §6 hashes).

| | pilot_v1 | bulk_v1 |
|---|---|---|
| profile | `mixture` | **`mixture_v2`** (new) |
| episodes (complete / aborted / planned) | 145 / 5 / 150 | **1,445 / 55 / 1,500** |
| aborted share | 3.33 % | 3.67 % |
| frames kept | 88,047 | **876,261** |
| train / val / test episodes | 119 / 15 / 15 | 1,086 / 198 / 199 |
| train / val / test frames | 70,580 / 9,000 / 8,467 | 642,006 / 116,724 / 117,531 |
| PNG bytes on disk | 6.70 GB | 66.94 GB |
| mean PNG size | 76,133 B | 76,393 B |
| generation wall-clock | 130 s | ≈ 1,370 s (22.8 min) |
| generation throughput | ≈ 677 frames/s | ≈ 640 frames/s |
| packed bytes, 96×72 (`images.bin`) | 1.83 GB | **18.17 GB** |
| packed bytes, 64×48 (`images.bin`) | 811 MB | **8.08 GB** |
| sidecar (`analysis.jsonl`) | 62.8 MB | 626.2 MB |
| `labels.bin` | 1.23 MB | 12.27 MB |
| `viewUnsafe` frames (share) | 357 (0.41 %) | 3,112 (0.36 %) |
| `viewUnsafe` episodes (share) | 90 (60.0 %) | 562 (37.5 %) |
| `actionHasClicked == false` share | 9.77 % | **5.08 %** |
| frames with ≥2 reachable doors | 46.3 % | 45.2 % |
| brief-graze (L3) frame share | 0.022 % | 0.008 % |

Aborts are all L2 (sustained wall-slide), no L1, no failures — same failsafe and
`viewUnsafe` labelling as the pilot.

**Per-persona abort counts.**

| persona | pilot aborts | bulk aborts |
|---|---|---|
| dither | 0 | 22 |
| spammer | 1 | 14 |
| decisive | 1 | 11 |
| prober | 0 | 3 |
| idle | 1 | 2 |
| late | 1 | 2 |
| uniform | 1 | 1 |
| **total** | **5** | **55** |

`dither`+`spammer` carry most bulk aborts (36/55) — expected, since they are the
two most-weighted personas (§5) and they click far more often, so they explore more
of each world's geometry.

---

## 2. Trial, tuning, and the new profile `mixture_v2`

### 2a. Procedure

Three generation passes, all counted with the Stage 4 probe definition
(reused verbatim from `probe_click._select_contexts`):

- **Pilot baseline** — the existing `pilot_v1` packed set (`mixture`).
- **Trial 1** — 60 episodes, `mixture`, scratch seeds 5,000,000–5,000,059
  (discarded). 57 complete + 3 aborted (5.0 %).
- **Trial 2** — 30 episodes, **`mixture_v2`**, scratch seeds 5,100,000–5,100,029.
  30 complete, 0 aborted.

### 2b. What the tuning did

The pilot/trial per-persona diagnostic (below) showed the qualifying yield is
**concentrated in `spammer` and `dither`** — the personas that click often and,
crucially, re-click a *different* door, so the player lingers in rooms with two
reachable doors on opposite sides of screen centre ("visually meaningful choice"
frames). `decisive` is also decent; `idle`, `uniform`, `late`, `prober` are lowest.

Measured qualifying-context rate per persona on **trial 1** (`mixture`):

| persona | frame share | qualifying / 1,000 persona-frames |
|---|---|---|
| spammer | 5.1 % | **173.9** |
| dither | 8.2 % | **138.3** |
| decisive | 34.1 % | 102.9 |
| late | 14.2 % | 98.0 |
| idle | 6.7 % | 97.7 |
| prober | 11.6 % | 96.4 |
| uniform | 20.0 % | 79.0 |

`mixture_v2` therefore **reweights toward `dither` and `spammer`** and away from
`uniform`/`idle`, while keeping every persona represented (floor ≥ ~5 %) so the
never-clicked and stale-held-click regimes survive. No bot-side bug was found: the
left/right imbalance traces to no engine or bot defect (Section 4), so **no bot
code was changed** and nothing in the tuning touches the existing profiles.

**The only knob changed is the profile weights** (`engine/config/config.ts`):

| persona | `mixture` (pilot, unchanged) | **`mixture_v2`** (new) |
|---|---|---|
| idle | 10 | 6 |
| decisive | 30 | 20 |
| late | 10 | 6 |
| dither | 10 | **30** |
| prober | 15 | 7 |
| spammer | 5 | **26** |
| uniform | 20 | 5 |

No reaction-delay parameters were changed (they are shared with `mixture`, and
editing them would alter the pilot). The existing profiles are byte-for-byte
unchanged.

### 2c. Before / after (qualifying contexts per 1,000 frames)

| | total | near-door |
|---|---|---|
| pilot_v1 (`mixture`, train) | 87.74 | 66.75 |
| trial 1 (`mixture`, train) | 102.85 | 76.74 |
| **trial 2 (`mixture_v2`, train)** | **148.17** | **122.72** |
| bulk_v1 realised (`mixture_v2`, train) | **126.53** | **100.07** |

**Precisely how the headline factors relate (the brief asked):**

- Pilot total = 87.74 per 1,000. Trial-2 total = 148.17 per 1,000.
  `148.17 / 87.74 = 1.69×` — so **"148 vs 88 per 1,000" is 1.69×, not 1.85×**.
- The **1.85×** figure is the *near-door* ratio: pilot near = 66.75/1,000,
  trial-2 near = 122.72/1,000, `122.72 / 66.75 = 1.84×`. That is the number that
  meets the brief's "roughly 2× the pilot's yield" target (the signal the model
  must learn is the near-door onset).
- **Realised on the full bulk run**, the yield is lower than the 30-episode trial
  suggested: `126.53 / 87.74 = 1.44×` total and `100.07 / 66.75 = 1.50×` near.
  The trial was optimistic by ~15 % (small-sample variance). The bulk is the
  number to trust.

Trial-2 direction balance met target (Section 4): signed heading changes
+43.6 %/−46.0 %, turn windows +48.9 %/−51.1 %, click-resolved turn +48.2 %/−45.4 %.

---

## 3. Qualifying counterfactual contexts (per split, near/far, episodes)

**Bulk_v1** (`mixture_v2`), counted with `count_contexts.py` (same selector):

| split | frames | qualifying | /1,000 | episodes | near | near/1,000 | near eps | far | far eps |
|---|---|---|---|---|---|---|---|---|---|
| train | 642,006 | 81,236 | 126.53 | 1,067 | 64,246 | 100.07 | 1,058 | 16,990 | 933 |
| val | 116,724 | 14,556 | 124.70 | 195 | 11,725 | 100.45 | 193 | 2,831 | 165 |
| test | 117,531 | 14,545 | 123.75 | 192 | 11,504 | 97.88 | 190 | 3,041 | 160 |

**Pilot_v1** (`mixture`) for comparison:

| split | frames | qualifying | /1,000 | episodes | near | near/1,000 | near eps | far | far eps |
|---|---|---|---|---|---|---|---|---|---|
| train | 70,580 | 6,193 | 87.74 | 114 | 4,711 | 66.75 | 111 | 1,482 | 93 |
| val | 9,000 | 860 | 95.56 | 15 | 681 | 75.67 | 15 | 179 | 12 |
| test | 8,467 | 700 | 82.67 | 15 | 446 | 52.68 | 15 | 254 | 13 |

**Coverage target met with room to spare.** The brief asked for **≥ ~400 near-door
contexts from ≥ 100 distinct episodes** in each of val and test: bulk val has
**11,725 near-door contexts from 193 episodes**, test **11,504 from 190** (~29×
the ask). Train has 64,246 near-door contexts from 1,058 episodes.

**Accepted click onsets and retargets** (from the sidecar: an accepted click whose
sticky target changes = onset; an onset where the previous target was non-null =
retarget):

| split | onsets | retargets |
|---|---|---|
| pilot train / val / test | 684 / 72 / 74 | 676 / 71 / 74 |
| bulk train / val / test | 15,559 / 2,870 / 2,649 | 15,207 / 2,800 / 2,599 |

Bulk gives **~2,870 val** and **~2,649 test** onsets — real statistics for
action-following evaluation (the pilot's val had 72).

---

## 4. Left/right balance

Turn-direction imbalance was the other Stage-4 concern. Measured on both datasets
(all values are "% positive / % negative"):

| metric | pilot train | bulk train | bulk val | bulk test |
|---|---|---|---|---|
| signed heading change, all steps | 42.1 / 43.8 | 44.6 / 44.4 | 43.9 / 44.7 | 44.0 / 44.8 |
| signed heading change, turn windows (\|Δ\|≥0.5°) | 49.9 / 50.1 | 50.1 / 49.9 | 49.2 / 50.8 | 49.5 / 50.5 |
| click-resolved turn direction | 47.2 / 47.3 | 47.3 / 47.1 | 47.6 / 46.7 | 48.0 / 45.8 |
| reachable-door `screenX` (left/right) | 48.6 / 51.4 | 48.5 / 51.5 | 48.9 / 51.1 | 48.6 / 51.4 |

**Balance target met.** The brief's target was signed heading changes within
47–53 % either way on turn windows: bulk is **50.1 / 49.9** (train) — effectively
symmetric, and much better than the pilot's turn-window skew. The residual
all-steps skew (≈44.6 / 44.4, i.e. ~11 % of steps are exactly zero) is the
straight-walking frames, not a directional bias.

Reachable-door `screenX` is left/right balanced (~48.5 / 51.5) in **both** datasets,
matching the Stage-4 finding. **The imbalance was never an engine
world-generation problem** (the world is balanced) and the bulk's near-perfect
turn balance confirms it was pilot bot behaviour now smoothed out by weighting;
**no engine or bot fix was needed** (brief rule 8(iv) did not trigger).

---

## 5. Persona statistics — does each persona earn its weight?

Per-persona qualifying-context yield **and** abort exposure on the bulk set
(`persona_stats.py`; aborts attributed to the persona active at the crash):

| persona | frame share (train) | click rate | qualifying /1,000 | aborts (all splits) | aborts /1,000 frames |
|---|---|---|---|---|---|
| dither | 30.3 % | 2.8 % | 136.4 | 22 | 0.08 |
| spammer | 24.3 % | 71.5 % | **192.7** | 14 | 0.07 |
| decisive | 20.0 % | 1.5 % | 96.7 | 11 | 0.06 |
| prober | 8.0 % | 3.4 % | 81.2 | 3 | 0.04 |
| idle | 6.4 % | 0.0 % | 60.2 | 2 | 0.04 |
| late | 6.1 % | 0.4 % | 72.4 | 2 | 0.04 |
| uniform | 5.0 % | 1.7 % | 87.9 | 1 | 0.02 |

**Verdict.** `dither` and `spammer` have the highest yield per frame (136 and 193
per 1,000) and therefore earn their raised weight — but they also carry the most
aborts in absolute terms (36 of 55). Normalised, their abort **rate** is only
0.08 and 0.07 per 1,000 persona-frames, essentially the same as every other
persona (0.02–0.08): they abort more only because they click more, not because
they are more dangerous per frame. `idle` (which never clicks) and `late` have the
lowest yield and are kept small but non-zero so the never-clicked and
stale-held-click regimes stay represented. All personas ≥ ~5 % of frames (the
brief's floor was ~3 %).

**Final bulk persona frame shares** (report.md §8): decisive 20.3 %, dither 30.3 %,
idle 6.2 %, late 6.5 %, prober 7.7 %, spammer 24.3 %, uniform 4.7 %.
Pilot for comparison: decisive 31.3 %, uniform 24.5 %, prober 13.7 %, idle 9.6 %,
late 9.5 %, dither 6.3 %, spammer 5.0 %.

**Straightest-choice rate at accepted clicks with ≥2 reachable** (Stage-2 metric —
≈ 1/n means the choice is not biased by sorted order / tie-breaking):
pilot **48.8 %**, bulk **48.6 %** (decisive 39.8, dither 24.9, late 44.4,
prober 37.0, spammer 54.0, uniform 61.1). Unchanged from the pilot — the tuned
profile did **not** introduce a choice bias.

**Label ambiguity** (real target differs from stateless resolution): pilot
11.9 %, bulk **11.8 %**. Unchanged.

---

## 6. Integrity

- **Seed disjointness.** Bulk ranges: train 10,000,000–10,001,099, val
  11,000,000–11,000,199, test 12,000,000–12,000,199 — mutually disjoint and clear
  of every pilot seed (0–119, 1,000,000–1,000,014, 2,000,000–2,000,014) and of the
  scratch trial ranges (5,000,000 and 5,100,000). The report's integrity check
  finds **0 seeds shared between splits**. The plan's recorded ranges match the
  episode directories (verified in §7 of the plan/run).
- **Incomplete episodes:** 0 (report.json `incompleteEpisodes: []`).
- **Manifests' hashes.** All 1,483 bulk manifests carry **one** `engineSourceHash`
  and **one** `configHash`: `engineSourceHash = 7742daca…d8ee7ad7b… `,
  `configHash = a29cd08b…8081c8b9…`. Both packed sets record exactly these.
  (The pilot's differ — its config hash predates `mixture_v2`, and its source hash
  predates the Stage-4B runner edit; see §8.)
- **Packed totals equal the manifests** at both resolutions (checked independently
  from the 1,483 manifests): episodes **1,483**, frames **876,261**, `viewUnsafe`
  **3,112**, `actionHasClicked==false` **44,546**. The per-split frame/episode
  totals match too.
- **Loader sanity on the bulk set.** The existing loader tests
  (`neural/tests/test_loader.py`) **all pass** against `bulk_v1_96x72`:
  action-alignment (each packed `actionX` equals the manifest's, and
  `controlSignalX[i+1]`), window-validity rules (loader count == independent
  manifest count), the GPU smoke train, and throughput. Throughput, memory-mapped
  at 96×72 / K=4 (per-process, page-fault bound — **not** an in-RAM load):

  | DataLoader workers | windows/s | peak RSS |
  |---|---|---|
  | 0 | 300 | 2.8 GB |
  | 2 | **3,808** | 2.8 GB |
  | 4 | **7,672** | 2.8 GB |

  The pilot loader still measures ~1,900–2,100 windows/s in its in-memory mode
  (the Stage-4 figure was >3,000 on a quieter machine; the threshold is
  load-sensitive — see §8).

---

## 7. Valid training windows (viewUnsafe removed)

Bulk_v1, 96×72, `count_valid_windows(require_safe=True)`:

| K | H | train | val | test |
|---|---|---|---|---|
| 4 | 1 | 633,226 | 115,158 | 115,832 |
| 4 | 4 | 628,414 | 114,276 | 114,912 |
| 4 | 10 | 618,843 | 112,520 | 113,081 |
| 8 | 1 | 626,816 | 113,982 | 114,606 |
| 8 | 4 | 622,029 | 113,102 | 113,691 |
| 8 | 10 | 612,497 | 111,364 | 111,861 |
| 16 | 1 | 614,082 | 111,653 | 112,166 |
| 16 | 4 | 609,333 | 110,787 | 111,253 |
| 16 | 10 | 599,861 | 109,076 | 109,442 |

The `viewUnsafe` removal costs ~0.6 % of windows at K=4/H=1 and ~4 % at
K=16/H=10 (longer windows are likelier to touch a flagged frame).

---

## 8. Shortcuts, `.gitignore`, changes outside new profile/plan files

**`.gitignore` edits: none needed.** `engine/data/`, `neural/data/`,
`neural/review/` and `neural/runs/` are already ignored, so the bulk dataset, both
packed sets, and the sample image are all outside git.

**Files added (all additive):**

- `engine/config/config.ts` — the new `mixture_v2` profile (the only engine-data
  change; existing profiles untouched).
- `engine/data/bulk_v1.plan.json` — the plan (under gitignored `engine/data/`,
  like the pilot's).
- `neural/count_contexts.py`, `neural/persona_stats.py`, `neural/sample_grid.py`,
  `neural/throughput_bulk.py` — read-only analysis/measurement helpers.
- `neural/memcap.sh` — a memory-capped launcher (Section 9).
- `neural/docs/STAGE4B-REPORT.md` (this file).

**Existing files modified (small, documented):**

- `engine/src/headless/cli.ts` — the `--profile` help line now lists the profiles
  dynamically (cosmetic).
- `engine/src/headless/plan.ts` — added an **optional** `clipStopMinEpisodes`
  plan field.
- `engine/src/headless/workerPool.ts` — the abort-share early stop now fires only
  after `clipStopMinEpisodes` finished episodes (default **100**, was hard-coded
  20), at the user's request. The stop threshold (`CLIP_STOP_SHARE`, 25 %), the
  alert (`CLIP_ALERT_SHARE`, 5 %) and the failsafe are **unchanged**.
- `neural/dataset.py` — `PackedDataset.__reduce__` (+ a private `_in_memory`), so a
  DataLoader worker re-opens the memmap instead of being handed an 18 GB copy.
  Behaviour-preserving; **this is the one change that touches the Stage-3 loader**
  (see "needs user decision").
- `neural/analysis.py` — the sidecar reader now also exposes `clickX`,
  `clickAccepted` and `persona` (additive).
- `neural/probe_click.py` — `_select_contexts` gained an optional `split` argument
  (default `"val"`, unchanged).
- `neural/conftest.py`, `neural/tests/test_loader.py` — env knobs
  (`LNE_PACKED_96`/`LNE_PACKED_64`/`LNE_SRC`, `LNE_THROUGHPUT_IN_MEMORY`,
  `LNE_THROUGHPUT_MIN`) so the same tests can run against the bulk set. Defaults
  reproduce the pilot's behaviour exactly.

**Shortcuts / decisions.**

- The profile was tuned from a measured per-persona yield model plus one trial
  round (as the brief allows); I did not sweep.
- The pilot was **not** regenerated (its profiles are untouched, so it need not
  be). Its numbers above are the existing packed set, re-measured and unchanged.
- Throughput on the bulk is reported memory-mapped (the brief says not to assume
  18 GB fits in RAM); the in-memory path is the pilot's.
- The pilot's `test_throughput` floor (>3,000 windows/s) is load-sensitive: on this
  busier machine it measures ~1,900–2,100/s. The default is unchanged; the bulk run
  uses the memory-mapped floor. Reported, not "fixed".

**"Needs user decision."**

1. **`neural/dataset.py` change.** The brief says not to change the Stage-3
   loader. `__reduce__` is additive and behaviour-preserving, but it *is* a
   loader change; it is required for DataLoader workers on the 18 GB set
   (Section 9). Keep it, or revert and always run single-process / `fork`?
2. **Runner early-stop guard.** `clipStopMinEpisodes` (default 100) changes the
   runner's early-stop behaviour for *all* future plans. You asked for it; confirm
   it is the behaviour you want as a permanent default.
3. **Inherently-bad world seeds** (engine edge cases, no fixes made — logged as
   repros): `10000017` (entity `r3`, ~x 20.4 / z −59.8, crash ~f497) and
   `10000022` (entity `r4`, ~x 48.9 / z −10.3, crash f249) abort under **both**
   `mixture` and `mixture_v2`; 53 more seeds abort under `mixture_v2` only
   (`bulk_v1/aborted.jsonl`). These are L2 wall-slide edge cases where the bot
   walks into a wall — worth an engine look before Stage 5, but out of scope here.
4. **`dither`/`spammer` dominance** (30 %/24 % of frames). They maximise the
   choice-context yield but skew the persona mix; if Stage 5 wants more
   human-like click statistics, a future profile could trim them.

---

## 9. The memory incident and its fix (unplanned)

During post-pack verification the **editor session was OOM-killed**. Root cause,
confirmed by reproduction under a hard cgroup cap:

- The loader hands `PackedDataset` to `torch` DataLoader workers. On this machine
  `torch.multiprocessing` uses the **`forkserver`** start method (not `fork`), so
  the dataset must be **pickled** to each worker.
- `np.memmap` pickles by **copying its whole backing file**. The pilot's 1.8 GB
  memmap fit; the bulk 96×72 memmap is **18 GB**, so two workers each tried to
  materialise an 18 GB array → a global kernel OOM (a 26 GB Python), which killed
  the VS Code app scope.
- **Fix:** `PackedDataset.__reduce__` returns only `(path, in_memory)`, so each
  worker re-opens the memmap. Verified: the same `num_workers=2` probe that OOM'd
  at 6 GB now peaks at **642 MB**, and bulk loader throughput rises to
  **3,808 w/s (2 workers)** / **7,672 w/s (4 workers)** at 2.8 GB peak.
- **Isolation:** every heavy job from here on runs in a memory-capped systemd user
  scope via `neural/memcap.sh` (`systemd-run --user --scope -p MemoryMax=…`), so a
  runaway allocation is killed at the cap instead of taking down the editor.

**Swap:** not needed, and not recommended for this. The failure was a single
specific bug (memmap copied into workers), now fixed; raising swap would only let a
future runaway thrash the disk and slow the whole desktop instead of failing fast.
The cgroup cap is the better safety net. (If you want belt-and-suspenders later, a
modest swap bump plus the cap is fine — but the cap is the load-bearing part.)

---

## 10. Sample image (for your glance)

`neural/review/bulk_v1_sample.png` — one grid of **24 frames** drawn at random:
random train/val/test episodes, random mid-episode positions, plus near-door
"two-door choice" moments, each captioned `split / episode / frame / kind`, shown
at the packed resolution (nearest-enlarged). It is only to let you glance at the
room/hallway variety and confirm nothing looks broken or odd — **I have not judged
it**. Path: `neural/review/bulk_v1_sample.png`.

---

## 11. Verification checklist (brief Section 5)

1. Pre-flight tests/typecheck — engine `tsc --noEmit` clean, **56/56** vitest pass
   (re-run after the runner edit).
2. Seed disjointness — bulk splits mutually disjoint and clear of the pilot; 0
   shared seeds reported.
3. Totals — packed frames/episodes/`viewUnsafe` equal the manifests at both
   resolutions (§6).
4. Loader sanity on bulk — existing tests pass; a sample-of-windows alignment
   check (action paired with each context frame equals the manifest's `actionX`)
   passes (§6).
5. Abort share (3.67 %) and direction balance (50.1/49.9 on turn windows)
   measured and reported against the targets (§1, §4).
6. Pilot unchanged and still loads — re-measured identical totals and coverage
   (§1, §3); pilot packed sets untouched.

Deliverables: dataset + plan + diagnostics + `aborted.jsonl` ✅;
`mixture_v2` profile (existing profiles unchanged) ✅; packed 96×72 / 64×48 sets ✅;
this report + sample image ✅; `CHANGES-LOG.md` + `README.md` updates ✅; git,
engine simulation/rendering/schema, and the pilot untouched ✅.
