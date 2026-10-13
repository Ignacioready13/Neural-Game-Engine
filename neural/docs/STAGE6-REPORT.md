# STAGE 6 REPORT — the player: a real-time client for experiencing and comparing models

NeuralGameEngine, the `/neural` half. Stage 6, 2026-10-12.
Continues `STAGE5-REPORT.md` / `STAGE5B-REPORT.md`; all earlier rules still apply.
No training, no engine change, no dataset change, no git operation.

**Deliverable in one line** — `./play.sh` runs any *named* model in a real-time window
at the training rate, with the engine's click semantics and click pulse, a resizable /
fullscreen letterboxed display, HUD and help, session recording and replay against any
model, and live-session power accounting.

---

## 0. What already existed versus what I changed

`neural/play.py` (Stage 5, Milestone 5) was a ~360-line script that loaded a checkpoint
path, generated frames and blitted them into a fixed `768x576` window. What it did
correctly: the closed loop over a real K-frame seed, the debounce arithmetic (3 frames /
150 ms), and N1/N2 (only frames + click signals reach the model). What it did not do,
relative to the brief:

| # | gap | fix |
|---|---|---|
| 1 | no model selector; `--checkpoint` only, and the documented command loaded the *old* model | name-based registry (`model_registry.py` + `models/registry.json` + `register_model.py`), `--model`, `--list-models`, `M`/`shift+M` hotkeys, newest-selected-is-default; names are the internal run names (§1.1) |
| 2 | the action tensor handed to the model was the *current* action repeated K times, not the real click history | pass `hist[-K:]` (the same rows the features come from) |
| 3 | the click history grew by one row per frame and `steer.click_features` walks it in a Python loop → O(T) per frame, so long sessions slowed down without bound | trim the history to `hist_extra + K` |
| 4 | the seed context's real click history was used for features but *not* to restore the debounce state, so the first click of a session was accepted even 1 frame after a real accepted click | `ClickState.restore()` reconstructs the last-accepted frame and the held value from the history |
| 5 | the "pulse" was a 3-px-wide full-height white bar | the engine's shader disc: radius `0.055·(1+(1−life)·1.5)·W/2` growing as it fades, `alpha = 0.85·(1 − age/0.4 s)`, soft radial gradient — for **every** click, accepted or not |
| 6 | fixed, non-resizable window, no fullscreen, no aspect control; clicks mapped against the *window* width | resizable + `F11`/`alt+enter` fullscreen, aspect-preserving letterbox (integer scale for nearest-neighbour), click X mapped against the **displayed image** with clamping in the margins |
| 7 | a plain `clock.tick(20)` loop: no fixed timestep, no drop accounting | fixed 20 Hz accumulator at the training rate, `MAX_CATCHUP = 2`, explicit dropped-step counter and an on-screen warning when it cannot keep up |
| 8 | no seed control (a random clip each launch), no restart keys, no HUD, no help, no recording, no screenshots | seed bank + `R` / `shift+R` / `--seed-context` / `--noise-seed`, HUD + help overlay, session recording + replay, screenshots, display GIF recording |
| 9 | `--headless --frames N` emitted the seed frame as frame 0 (n_frames included the seed) | generates exactly N new frames, matching the interactive path's click timing |

Also fixed in passing: `Player.generate` is now the single code path behind headless and
replay runs, so "what you played" and "what you replay" cannot drift.

### Files added

- `neural/model_registry.py` — the registry (load/save/list/get/newest/register/resolve),
  no torch import, so listing is instant.
- `neural/register_model.py` — `--list`, and `--run NAME` / `--name/--generator` to
  register a new revision and freeze a copy of its checkpoint.
- `neural/build_seeds.py` — builds `models/seeds.npz`, the cached seed contexts.
- `neural/play.sh` — the one obvious way to start the player (handles the venv and
  `memcap.sh`).
- `neural/tests/test_stage6.py` — 41 tests (geometry, click semantics, naming/aliases/
  selection, the `M` cycle, both models load, replay round trips, the window under SDL's
  dummy driver, no-oracle).
- `neural/models/registry.json` (committed) +
  `models/{flow96,flow96c,flow96d}/generator.pt`, `models/_shared/steer96.pt`,
  `models/seeds.npz` (derived, gitignored).
- `neural/docs/STAGE6-REPORT.md`, `neural/docs/STAGE6-PROGRESS.md`, this file.

### Files changed

- `neural/play.py` — rewritten as the real client (the Stage 5 API — `ClickState`,
  `Player`, `_to_uint8`, `write_gif`, `write_strip` — is kept, so `tests/test_stage5.py`
  and the review-image helpers still work).
- `neural/powerlog.py` — added `PowerLogger.latest()`, a non-blocking accessor for the
  HUD's live watts (the sampler thread already keeps the samples).
- `neural/README.md` — a "How to play and compare models" section at the top.
- `CHANGES-LOG.md` — one entry at the top.
- `.gitignore` — see §7.

---

## 1. The model registry, and the naming convention

### 1.1 Naming: the internal run name is the model name (the user's stated preference)

**The user asked for models to be identified by their internal training-run names
(`flow96`, `flow96c`, `flow96d`, and `flow96e` next) rather than by the Stage 5-era
letters, and for the player to display those names; the letters are kept only as
aliases.** So the registry's primary key *is* the run directory name, the HUD, window
title, listings and recordings all show it, and `--model 5b` still resolves (to
`flow96c`) so earlier notes and commands are not broken. The Stage 6 brief's own
`5a`/`5b` labels map as follows, confirmed against the Stage 5/5b reports rather than
assumed:

| internal name | Stage 6 brief's label | what it is |
|---|---|---|
| `flow96` | `5a` | the Stage 5 generator (`v0` in the Stage 5b scorecard): 0.77 M-param denoiser, 4-step flow, 8000 steps at 96x72 |
| `flow96c` | `5b` | the Stage 5b overnight winner: 3.19 M-param denoiser with self-attention, 12 000 steps |
| `flow96d` | (`5b-x`) | the rejected overnight candidate: edge-weighted reveal loss |

Nothing in the reports contradicts those identifications. Only the **flow96 family** is
registered (the user asked not to bother with the `flow64*` runs, which are a different
resolution anyway); the rest of `runs/` is left alone.

### 1.2 The registry

`models/registry.json` maps a name to its generator checkpoint (a frozen copy under
`models/<name>/generator.pt`, so a later training run cannot change what the name means),
the core it was trained on, the resolution, where its evaluation records live, its
aliases, and free-form notes.

| name | alias | generator | selected | status |
|---|---|---|---|---|
| `flow96` | `5a` | `runs/flow96/best.pt` → `models/flow96/generator.pt` | yes | default-eligible |
| `flow96c` | `5b` | `runs/flow96c/best.pt` → `models/flow96c/generator.pt` | yes | **default** |
| `flow96d` | — | `runs/flow96d/best.pt` → `models/flow96d/generator.pt` | no | playable, out of the `M` cycle |

### 1.3 "Selected" versus merely registered — where the selection table lives

The brief asked for a registry, not for a selection concept, but the user's request — *let
`M` iterate over the selected models while any model is reachable by flag* — needs one, so
each entry carries `selected`. **The project already has a machine-readable selection
table, and the registry's flags follow it:**

* `neural/eval_out/decision_*.json` — written by `score_table.py` against the frozen rule
  in `eval_sets/scorecard_v1.json`. Each file has an `incumbent` and per-candidate rows
  with a `keep` verdict. `decision_v0_vs_flow96c.json` has `flow96c → keep: true` (it
  replaced `v0`); `decision_flow96c_vs_flow96d.json` has `flow96d → keep: false`.
* `neural/docs/ENERGY-LEDGER.json` — records `"winner": "flow96c"`.

So **selected = the kept revisions = `flow96` (the incumbent before the overnight run)
and `flow96c` (the winner)**; `flow96d` is registered but not selected. `M`/`shift+M`
cycle `flow96` ↔ `flow96c` only. A non-selected model, or a bare `runs/<name>/best.pt`
that was never registered at all, is still playable by naming it, which is what makes the
rejected candidate inspectable by hand:

```bash
./play.sh --model flow96d     # registered, not selected -> loaded from the frozen copy
./play.sh --model flow96e     # not registered        -> loads runs/flow96e/best.pt
```

Starting the player on a non-selected model and pressing `M` joins the selected cycle at
its first entry (so `flow96d` → `flow96` → `flow96c` → `flow96`), and the HUD labels it
`flow96d [not selected]` while it is active.

The generator checkpoints are self-contained (the flow model embeds its steering core),
so `core` is provenance, not a load-time dependency: `--model flow96c` needs no second
file.

Register the next revision in seconds (reads the training run, copies the checkpoint,
prints the play command):

```bash
.venv/bin/python register_model.py --run flow96e \
    --core runs/steer96/best.pt --description "flow96e: longer self-rollout" \
    --parent flow96c
./play.sh --model flow96e
```

Registering a *rejected* candidate keeps it out of the `M` cycle while leaving it
playable: add `--not-selected`.

## 2. Seed contexts (no 18 GB mount needed)

A run starts from K=4 real frames plus the real click history that precedes them
(`hist_extra = 8`, so `deltaActionX`/`frames-since-click` at the first context frame are
right). `build_seeds.py` caches 160 such clips — 80 val, 40 train, 40 test, labelled —
in `models/seeds.npz` (5.3 MB). The player uses the cache by default and never opens the
packed set unless you pass `--packed`; `--split val` picks which split a random seed
comes from; `--seed-context EPISODE:START` pins an exact clip. Seeds are drawn from the
first 10 % of an episode (the brief's "early-episode frames"), and the seed's clip id and
the noise seed are shown in the HUD.

The cache is only valid at 96x72, the resolution both registered models play at; playing
a model at another resolution means `--packed <that resolution>`.

## 3. Click semantics, pulse and pacing (the details that matter)

* **Mapping** — `x = clamp((clientX − imageLeft) / imageWidth · 2 − 1, −1, 1)`, relative
  to the *displayed image*, so a click in the letterbox margin clamps to ±1 exactly as
  the engine's canvas does. Y is ignored.
* **Debounce / latch** — the held signal changes only on an accepted click; a click
  within 3 frames (150 ms at 20 Hz) of the last accepted one is ignored *for the signal*
  and still pulses; `actionHasClicked` latches to 1 after the first accepted click. The
  state is reconstructed from the seed's real history, so a first click that lands 1–2
  frames after a real accepted click is correctly ignored.
* **Features** — computed by `steer.click_features`, the same function training used, over
  the last K rows of the (trimmed) real click history.
* **Pulse** — the engine's disc (see §0 row 5), drawn in image space, clipped to the image
  rect, on top of the frames, never fed back (a test asserts the composed image returns
  to the un-pulsed pixels once the pulses expire).
* **Pacing** — one frame per 50 ms of real time, from a wall-clock accumulator. The
  display redraws at up to 240 Hz only when a new frame arrived or a pulse is alive, so
  the pulse fades smoothly without generating faster than 20 Hz. If the model falls more
  than 2 frames behind, the excess is dropped, a transient "cannot keep up (N fps):
  dropped N step(s)" message appears, and the HUD's fps goes red.
* **Noise seed** — frame `f` is sampled with `noise_seed + f`, so a whole run is
  reproducible from (model, seed context, noise seed, clicks) — which is what makes
  replay exact.

## 4. Recording, replay and comparing two models

`C` (or `--record FILE`) writes `models/recordings/<model>_session_<stamp>.json`:
model, checkpoint, seed context (episode/start/split), noise seed, resolution, and the
full click log `[[frame, x], ...]`, flushed on stop or on exit.

Replay drives a run from that log, so the same clicks and the same start can be seen on
any model:

```bash
.venv/bin/python play.py --model flow96 --replay models/recordings/flow96c_session_<stamp>.json
```

During a replay live clicks are ignored (they would break the comparison); press `M` to
switch models *mid-replay*, which restarts from frame 0 of the same recording on the
same seed context — the fairest hand comparison available without a side-by-side mode.
Headless replay writes a GIF (`--out`) for the review folder. Replaying a recording on
the model that recorded it reproduces the frames bit for bit (a test asserts
`torch.equal` on every frame).

**Side-by-side was deliberately not built** (the user said it was unnecessary): two
models at 96x72 in one process would fit the budget (~6 ms/frame each against 50 ms),
but the replay + `M` path already answers "the same clicks on flow96 and flow96c", and it
keeps the display, HUD and pulse code single-model and simple.

## 5. Frame-time statistics (this machine, RTX 5070, under `memcap.sh --mem 12G`)

500 generated frames per model at the real 20 Hz pacing (`--headless --pace`,
`--report-frames`), 5 warm-up frames discarded first:

| model | mean ms | p95 ms | worst ms | implied fps | budget | dropped steps |
|---|---|---|---|---|---|---|
| `flow96` | 4.29 | 4.65 | 5.25 | 233 | 50 ms | **0** |
| `flow96c` | 6.18 | 7.51 | 8.37 | 162 | 50 ms | **0** |
| `flow96d` | 4.96 | 6.92 | 8.40 | 201 | 50 ms | **0** |

So the real-time margin is roughly **8–11x** (the brief quoted 37–68 steps/s for the
same models in the G4 benchmark loop, which includes a `uint8` decode and per-step
`.cpu()` of every frame plus 1 Hz power sampling; the player's per-frame cost is lower).
A live windowed session reports the same thing from its own loop: mean 5–7 ms, p95
< 8 ms, worst < 9 ms, 0 dropped steps over 400+ frames.

**Design choice: no worker thread.** With a 6 ms frame inside a 50 ms budget, a producer
thread and a queue would add synchronisation risk for nothing, so generation, input and
drawing share one loop; the only "pacing" state is a small accumulator. This is the
measurement behind that choice, not an assumption. If a future model needs > ~35 ms per
frame, the threaded variant is the obvious next step (noted, not built).

## 6. What I could verify without eyes, and what only the user can judge

Verified:

1. Both registered models load from their **frozen copies** and predict (a test, and the
   paced runs above).
2. The click mapping, including letterbox margins and clamping, and the debounce/latch
   semantics — unit tests.
3. The window can be created, resized and toggled to fullscreen **on this desktop**:
   windowed 1536x1152 (scale 16, default), forced `--fullscreen` gives the full
   3840x2400 desktop with the image letterboxed to 3168x2376 centred at (336,12). Tested
   with the default SDL driver (Wayland), forced `wayland`, forced `x11`, and under the
   `dummy` driver (where the tests also drive a resize and both fullscreen toggles).
4. Replay round trip: byte-identical GIFs for the same model, and a valid GIF on the
   other model.
5. The display recording GIF contains the frames *and* the pulses (frames written at
   native resolution, upscaled).
6. Power accounting: a live session appends one ledger row and prints
   `60 frames in 3.0s (20.3 fps)  GPU 20.27 W mean / 21.41 W peak  0.69 J/frame
   20.27 Wh per hour of play`. (The rows this verification produced were removed from
   `docs/ENERGY-LEDGER.json/md` afterwards so the Stage 5b accounting is untouched.)
7. No-oracle (N1/N2): the player reads no engine or sidecar data. `play.py`,
   `model_registry.py`, `build_seeds.py` and `register_model.py` import neither the
   judges nor the sidecar; a test runs a rollout with `analysis.load_analysis` patched to
   raise, and another asserts the model call carries only context frames, click signals,
   features and a seed.
8. Existing tests still pass (see §8).
9. Review artifacts, generated end to end by the player: `models/recordings/demo_flow96c.json`
   (240 frames, 4 recorded clicks, seed `val ep1384@56`, noise seed 7) replayed on all
   three models produced `review/stage6/demo_{flow96,flow96c,flow96d}.gif`. A programmatic
   check of those files (no eyes needed) confirms the player's output is real and the three
   models behave differently on identical input:

   | model | mean brightness | near-black frames | frame-to-frame mean |Δ| |
   |---|---|---|---|
   | `flow96` | 131.5 | 0 / 240 | 5.98 |
   | `flow96c` | 138.7 | 0 / 240 | 24.42 |
   | `flow96d` | 145.9 | 0 / 240 | 14.92 |

   Mean |Δ| between the flow96 and flow96c *runs* is 69.95 grey levels and between flow96c
   and flow96d 51.5, so the same clicks clearly drive different rollouts. flow96c's
   frame-to-frame change is 4x flow96's, the same ordering the Stage 5b report measured
   (the incumbent's
   closed-loop motion was ~12x too small; the winner's is far larger), so the player
   reproduces the known difference rather than hiding it.

Only the user can judge: how the pulse *looks* in motion, whether 20 Hz feels right for
clicking, whether fullscreen behaves on their compositor/window manager (I saw 3840x2400
correct from SDL, but not how the WM animates it), and whether a 1536x1152 default window
suits their screen. I used no vision subagent.

## 7. .gitignore edit

Added, with a comment, at the end of the file:

```
/neural/models/*/*.pt
/neural/models/seeds.npz
/neural/models/recordings/
```

Rationale: `models/registry.json` is committed (it is the source of truth for names →
lineage → evaluations), but the checkpoint copies (~22 MB), the seed cache (5.3 MB) and
the recorded sessions are derived and re-creatable (`register_model.py`, `build_seeds.py`)
— the same convention the repo already uses for `/neural/runs/` and `/neural/data/`.
No other `.gitignore` change.

## 8. Verification runs

```
# unit + integration tests (41 new, Stage 1-6 suite unchanged)
./memcap.sh --mem 12G -- .venv/bin/python -m pytest tests/ -q
  -> 97 passed                        (final run, machine idle)
  -> 96 passed, 1 failed              (earlier runs while Chrome/OnlyOffice/VS Code were
     busy: tests/test_loader.py::test_throughput, measured at 485-2200 windows/s against a
     3000 floor. Pre-existing and load-dependent — it imports only dataset.py, untouched
     by this stage (load average 9.1 at the worst point), and the Stage 4B/5b reports
     already record it as load-sensitive on this machine: 1151 windows/s alone while busy,
     ~6800-8200 windows/s when quiet.)

# paced headless runs, every registered model, frame-time JSON per model
for m in flow96 flow96c flow96d; do
  ./memcap.sh --mem 12G -- .venv/bin/python play.py --model $m --headless --frames 500 \
      --pace --clicks "15:0.8,60:-0.6,140:0.9,260:-0.3" --report-frames /tmp/frames_$m.json
done

# replay round trip: same model -> bit-identical, other model -> valid
.venv/bin/python play.py --model flow96c --headless --frames 120 --clicks "5:0.7,40:-0.9,90:0.2" \
    --record models/recordings/roundtrip.json --out /tmp/a.gif --seed-context 1435:25
.venv/bin/python play.py --model flow96c --headless --replay models/recordings/roundtrip.json --out /tmp/b.gif
.venv/bin/python play.py --model flow96 --headless --replay models/recordings/roundtrip.json --out /tmp/c.gif
cmp /tmp/a.gif /tmp/b.gif     # identical

# the window, on the real desktop and under the dummy driver
./play.sh --no-memcap --model flow96c --no-power --exit-after 2
./play.sh --no-memcap --model flow96c --no-power --fullscreen --exit-after 2
./play.sh --no-memcap --model 5a --no-power --exit-after 2          # alias still works
./play.sh --no-memcap --model flow96d --no-power --exit-after 2     # not selected
SDL_VIDEODRIVER=dummy ./play.sh --no-memcap --model flow96c --exit-after 3 --power

# naming / selection semantics
.venv/bin/python play.py --list-models
.venv/bin/python -m pytest tests/test_stage6.py -q
```

## 9. Judgment calls and things I did not do

* **Naming is the internal run name, per the user's explicit preference** (§1.1): the
  player displays `flow96c`, not `5b`, everywhere (title, HUD, listings, recordings,
  `--report-frames`), and the letters are aliases only. The one place the letters still
  appear is where the Stage 6 brief itself used them (§1.1's mapping table) so older
  notes remain checkable.
* **`selected` is a registry field, not a new source of truth**: it mirrors the frozen
  scorecard's `keep` verdicts already in `eval_out/decision_*.json` (§1.3). No new
  selection logic was invented.
* **A non-selected or unregistered model is reachable by flag but never by `M`**, which
  is exactly the split the user asked for; a bare `runs/<name>/best.pt` also loads
  without registration (a warning says it is unpinned).
* **`--exit-after SECONDS`** was added to the windowed mode so a session can be scripted
  (it is how the window checks above run). Light, and useful for demos.
* **`build_seeds.py` samples whole episodes where possible** rather than 160 random
  draws from the same episode, so the bank is not one clip repeated.
* **Not built, deliberately**: side-by-side (user's call), a click-magnitude/direction
  editor, per-model on-screen tuning, video encoding beyond GIF (Pillow only —
  `imageio`/`ffmpeg` would be new dependencies), any read of the engine or the sidecar.
* **The G4/benchmark loop was not refactored** to share a single "model runner" with the
  player. `Player.step` and `gates.run_g4` both go through
  `eval_harness.model_predict` + `features_from_history`, so the *model call* cannot
  drift; the surrounding loops differ (pacing, power, display) by design. The brief
  suggested sharing where cheap; this is the cheap sharing point.
* **A model that cannot keep up** is still playable: the loop degrades with a warning
  rather than silently running in slow motion (the accumulator would otherwise stretch
  each step).

## 10. Known limitations

* The seed cache is 96x72-only (see §2). Only the flow96 family is registered; other
  families (`flow64*`) are a different resolution and were left alone as asked — they
  are still reachable as bare run names, but the 96x72 seed cache would have to be
  rebuilt with `--packed` for them.
* `models/` copies are gitignored; a fresh clone must run `register_model.py` (the
  registry's `checkpoint_source` points at the training runs) and `build_seeds.py`, or
  pass `--packed`.
* The player holds at most one model per registered name in memory; switching between
  two 3.19 M-parameter models costs ~0.2 s (measured) and is done lazily.
* Fullscreen restores the *previous window size* on exit, not the previous position.
