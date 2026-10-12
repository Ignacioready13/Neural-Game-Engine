# NeuralGameEngine — Phase 1

A procedurally generated, backrooms-style first-person navigation engine. Its
only purpose is to generate a large, high-quality dataset of
`(sequence of frames, sequence of control signals)` pairs that will later train
a neural network (the "dynamics model"). The classical engine is a *teacher*,
not a permanent game: every design decision favors clean, consistent,
information-rich training signal.

The definitive architecture and data contract are this file, `CHANGES-LOG.md`,
`engine/docs/DATASET-SCHEMA.md`, and `engine/docs/CONTRACTS.md`. (The original
Phase 1 build prompt is historical; see `engine/docs/historical/`.)

## Two-phase architecture

1. **`/engine`** (this phase) — a classical, fully deterministic 3D engine
   (TypeScript + three.js) with two operating modes that share the exact same
   core:
   - **Human play** — point-and-click navigation through procedurally generated
     hallways and rooms, via the Vite dev server.
   - **Headless generation** — the same engine driven by a persona-based click-bot
     framework in headless Chromium (Puppeteer), writing frames + labels to
     `engine/data/` under resumable run plans.
2. **`/neural`** (future, out of scope) — a Python/PyTorch neural network that
   learns to generate the frames itself, conditioned on the control signal and
   a short window of its own prior frames. It is fully independent of this
   engine and never calls back into it; this engine is discarded once enough
   training data exists.

## Repository layout

```
/engine/      everything built in Phase 1 (TypeScript)
  /config/    config.ts — the single source of every tunable parameter
  /src/
    /core/    shared types, math, and deterministic RNG
    /world/   procedural hallway/room generator (lazy + disposal)
    /control/ point-and-click steering, spline/jerk, debounce
    /render/  three.js scene, lighting/flicker, fog, resolution pipeline
    /sim/     Engine.ts — the shared orchestrator used by BOTH modes
    /app/     main-interactive.ts (human) and main-headless.ts (bot)
    /headless/ Puppeteer driver, plan runner, click-bot framework, manifest
               writer, diagnostics report, CLI
  /tests/     vitest rule-verification + unit tests
  /docs/      CONTRACTS.md, DATASET-SCHEMA.md, PLANNING-ANSWERS.md
/neural/      empty placeholder for the future neural codebase
/data/        default output root for generated datasets (gitignored)
```

The default output root is `engine/data/` (gitignored), not `/data`.
```

## Requirements

- Node.js >= 20 and npm
- Google Chrome (used by Puppeteer for headless capture)
- A GPU with working WebGL for headless acceleration (see "GPU" below)

## Human play mode

```bash
cd engine
npm install
npm run dev          # prints a localhost URL — open it and click to steer
```

The player always walks forward at a fixed speed. Clicking anywhere on the
canvas steers toward the doorway whose on-screen X position is nearest the
click (only the click's X is used). Clicking near the screen center continues
straight ahead. A fading pulse marks every click.

### Human playtest recording

```bash
cd engine
npm run record      # opens a headful Chrome window at record.html; play, press Q to save
```

`npm run record` launches a **headful** Chromium window driving the same shared
`Engine` with a real mouse, and streams every simulated frame (full state +
PNG) back over the Puppeteer bridge to disk as `data/playtest_<timestamp>/` —
the same shape as a headless training episode, plus richer per-frame debug
fields (target doorway/phase, candidate reachability, collision events, wall
overlap). It's the tool for reproducing a human-discovered bug: the manifest
records the world `seed` and the exact per-frame click sequence, which a test
or replay script can drive deterministically.

```bash
npm run record -- --help       # full option list
npm run record -- --seed 123   # pin the world seed
npm run record -- --build      # force a vite build even if dist/ is current
```

Press **Q** (or Escape) to finish and save; closing the window keeps whatever
was last drained. The recorder auto-rebuilds `dist/` when source/config is newer
than the built bundle (see `src/headless/stale.ts`).

## Headless data generation

```bash
cd engine
npm run build         # produce dist/ (index.html + headless.html)
npm run generate      # run a plan (or a single episode) with the click-bot framework
npm run report -- engine/data/pilot_v1   # write report.json + report.md for a dataset
```

Generation is **plan-based and resumable**. A plan names a dataset, its mixture
profile, frames per episode, and an explicit list of `(seed, split, profile)`
episodes with disjoint seed ranges per split. The CLI can generate a plan and
run one:

```bash
# Write a plan (no generation yet):
npm run generate -- --make-plan pilot_v1 --train 120 --val 15 --test 15 \
  --frames 600 --profile mixture --output engine/data

# Run it (resumable; skips complete episodes, deletes+redoes partial ones,
# retries an infra crash once):
npm run generate -- --plan engine/data/pilot_v1.plan.json --concurrency 8
```

Options (see `npm run generate -- --help`):

```
--make-plan <name>   write a plan JSON at <output>/<name>.plan.json and exit
--plan <path>        load a plan JSON and run it (resumable)
--train/--val/--test <n>  per-split episode counts (for --make-plan)
--profile <name>     mixture profile: mixture | uniformOnly | doorAwareOnly
--frames <n>         frames per episode (default 600 = 30 s @20 Hz)
--concurrency <n>    simultaneous headless Chromium instances (default 8 for plans)
--output <dir>       output root (default: <repo>/engine/data)
--seed <n>           base seed for the legacy single-split mode
--episodes <n>       legacy single-split episode count (no --plan/--make-plan)
--build              rebuild dist/ before generating
```

Episodes land in `<output>/<plan>/<split>/seed_<seed>/` (deterministic names):

```
pilot_v1/
  train/seed_0/frames/000000.png ...   lossless PNGs at OUTPUT_RESOLUTION
              manifest.json            schema-v2 metadata (see DATASET-SCHEMA.md)
  aborted.jsonl                        clipping-failsafe records (4.11)
  report.json / report.md              diagnostics report (4.8)
```

The `seed_<seed>` directory names and the `episodeId` (`<plan>-<split>-<seed>`)
are deterministic, so a batch can be interrupted and resumed, and a given seed
always produces byte-identical frames on the same machine.

### Click-bot framework and mixture profiles

The single placeholder bot is replaced by a small **persona framework**. A bot
reads the same per-frame state snapshot the logger reads, keeps its own memory
(previous active entity, pending delayed clicks), and draws randomness from a
generator seeded from the episode seed but **separate** from the world generator
(so bot choices never perturb world generation).

Seven personas — `idle` (no clicks), `decisive`, `late`, `dither`, `prober`,
`spammer`, `uniform` — are mixed per ~100–300-frame segment under a named
profile. Profiles: `mixture` (default, the Stage-2 pilot), `uniformOnly`,
`doorAwareOnly`, and `mixture_v2` (the Stage-4B bulk profile — weights tuned
toward `dither`/`spammer` for more two-door counterfactual contexts while keeping
every persona and the never-clicked/stale-held regimes; see
[`neural/docs/STAGE4B-REPORT.md`](neural/docs/STAGE4B-REPORT.md)). The
`mixture`/`uniformOnly`/`doorAwareOnly` weights are never edited, so the pilot
stays reproducible. The
active persona per step is logged in the manifest. Full semantics and weights
are in `engine/docs/DATASET-SCHEMA.md` §9 and `engine/config/config.ts`.

### Clipping failsafe

Headless agent episodes are passed through a clipping failsafe (a safety net, not
a fix): on a hard wall-overlap (L1) or 10 consecutive wall-slide frames (L2) the
episode stops, the 8-second lead-in is deleted from disk, and the tail is either
trimmed (with `truncated`/`crashFrameIndex`/`trimmedFrames`) or the whole episode
is deleted if too little remains. Every aborted episode is recorded in
`aborted.jsonl`; brief grazes (L3) are reported but not a crash. Details in
`engine/docs/DATASET-SCHEMA.md` §7.

### View safety (`viewUnsafe`)

The clipping failsafe watches the *simulation*; a separate **label-only** flag
watches the *view*. The player is a floor point with no collision radius, so the
camera can end up close enough to a wall that (a) the surface is clipped by the
camera near plane, or (b) the camera crosses a single-sided wall plane and looks
through it. Each frame records `viewUnsafe: true` when either holds (closest
rendered surface nearer than camera near + `VIEW_UNSAFE_MARGIN`, default 0.10
world units, or a back-facing plane with geometry in front).

Unlike the failsafe, `viewUnsafe` **never aborts or trims an episode** — the
frames stay on disk and in the manifest. A training loader should **drop any
window that touches a flagged frame** (including its action target frame). The
share of flagged frames/episodes is reported in `report.md` §11. Details in
`engine/docs/DATASET-SCHEMA.md` §5b.

### Datasets

- **`engine/data/pilot_v1/`** — the Stage-2 pilot: 150 episodes (120/15/15),
  600 frames each, `mixture` profile. 88,047 frames; 3.3 % abort.
- **`engine/data/bulk_v1/`** — the Stage-4B bulk training set: 1,500 episodes
  (1,100/200/200), 600 frames each, **`mixture_v2`** profile. 1,445 complete +
  55 aborted (3.67 %), 876,261 frames; packed at 96×72 and 64×48 under
  `neural/data/`. See [`neural/docs/STAGE4B-REPORT.md`](neural/docs/STAGE4B-REPORT.md)
  for coverage numbers.

The `engine/data/` subtree is gitignored (data is regenerated, not committed); the
plan JSON for each dataset lives beside it and is copied into the dataset dir.

### GPU acceleration

Headless Chromium silently falls back to a software rasterizer (SwiftShader)
unless told otherwise. The driver launches Chrome with
`--use-gl=angle --use-angle=vulkan` plus GPU-forcing flags, and sets hybrid-
graphics env vars (`__NV_PRIME_RENDER_OFFLOAD=1`,
`__GLX_VENDOR_LIBRARY_NAME=nvidia`) to route to an NVIDIA dGPU on Optimus
laptops (harmless no-ops elsewhere). Before each episode it runs a self-check
that reads the WebGL `UNMASKED_RENDERER` string and **fails loudly** if it
contains any software-rasterizer marker (`swiftshader`, `llvmpipe`, `softpipe`,
`software`, `basic render driver`) — it will not silently generate data slowly
on a software renderer.

### Worker pool

`--concurrency` controls how many headless Chromium instances run at once, each
writing its own episode directory. The default is 4 (`WORKER_CONCURRENCY`) for
the legacy single-split mode and 8 for plan runs. Tune it up or down based on
your GPU memory and observed throughput. It is expected (and fine) that
large-scale generation is slow in wall-clock terms — what matters is that each
episode's simulated motion is correct and consistent, which the fixed timestep
guarantees regardless of load.

## Dataset format (`manifest.json`)

The manifest is **schema v2** and self-describing. The definitive field-by-field
contract (types, the action-alignment rule, held-signal semantics, provenance,
truncation) is `engine/docs/DATASET-SCHEMA.md`. The short version:

```jsonc
{
  "schemaVersion": 2,
  "episodeId": "pilot_v1-train-0",
  "seed": 0, "split": "train", "profile": "mixture",
  "outputWidth": 320, "outputHeight": 240, "simTimestep": 0.05,
  "configHash": "…", "engineCommit": "…", "dirty": true, "engineSourceHash": "…",
  "frames": [
    {
      "frameIndex": 0, "simulatedTime": 0.0,
      "imagePath": "frames/000000.png",
      "controlSignalX": 0.0,   // held click-X AT capture time, -1..1
      "actionX": 0.0,          // held click-X during step i -> i+1 (the training action)
      "actionHasClicked": false,
      "clickX": null, "clickAccepted": null,
      "playerX": 0.5, "playerZ": 1.5, "playerHeading": 1.5708,
      "currentEntityId": "h1",
      "viewUnsafe": false     // camera too close to a rendered wall/jamb
    }
  ],
  "debugFrames": [ /* persona, targetDoorwayId, candidates, collisionEvents, … */ ],
  "truncated": false, "complete": true
}
```

**The alignment rule:** the action for the transition from frame `i` to `i+1` is
`actionX[i]` (= `controlSignalX[i+1]`). Training must use `actionX`, never
`controlSignalX`, to predict the next frame. `controlSignalX` is the held
normalized screen-space click X (a step function; `0` = screen-center default
before the first click), kept for compatibility and live-client semantics.
`playerX/Z/heading` and the whole `debugFrames` section are debug context, not
fed to the network. `viewUnsafe` is not fed to the network either — it is a
filter: drop any window containing a flagged frame (see *View safety* above).
Frames are PNG (lossless) so compression artifacts don't pollute the training
signal.

## Config

Every tunable lives in `engine/config/config.ts`, with an inline comment. Key
values and their defaults:

| Constant | Default | Controls |
|---|---|---|
| `SIM_TIMESTEP` | 1/20 s | fixed simulated timestep (20 Hz) |
| `WALK_SPEED` | 4.0 | forward speed, world units/s |
| `EYE_HEIGHT` | 1.6 | camera height (fixed regardless of room height) |
| `FOV_HORIZONTAL_DEG` | 85 | horizontal field of view |
| `HALLWAY_WIDTH` / `HALLWAY_HEIGHT` | 4.5 / 3.5 | corridor width / ceiling height |
| `HALLWAY_LENGTH_MIN` / `_MAX` | 8 / 30 | segment length range |
| `HALLWAY_TERMINATION_WEIGHTS` | bend .35 / junction .35 / room .30 | how segments end |
| `JUNCTION_WEIGHTS` | 2-way .45 / 3-way .40 / 4-way .15 | fork arity |
| `HALLWAY_SIDE_DOORWAY_CHANCE` | 0.6 | probability a segment has any side doorways |
| `ROOM_WIDTH/DEPTH_MIN/MAX` | 10–20 | room footprint ranges |
| `ROOM_HEIGHT_MIN/MAX` | 2.8–15 | room ceiling range (min ≥ `DOORWAY_HEIGHT`) |
| `WALL_THICKNESS` | 0.5 | outward wall extrusion (doorway reveal; never changes walkable space) |
| `DOORWAY_WIDTH` / `_HEIGHT` | 2.7 / 2.8 | opening size |
| `DOORWAY_SPACING_FACTOR` | 3 | min center-to-center = 3× doorway width |
| `SIDE_DOORWAY_START_CLEARANCE` | 3.0 | distance from entry wall before the first side doorway |
| `DOORWAY_GATE_RADIUS` | 2.5 | proximity to commit to turning through a non-bend doorway |
| `TURN_GATE_RADIUS` | 2.5 | proximity to a bend corner to commit to the ~90° turn |
| `TURN_WAYPOINT_DEPTH` | 3.0 | how far past a turn corner/opening its waypoint sits |
| `JERK_THRESHOLD` | 6.0 rad/s² | max angular acceleration ("jerk") |
| `MAX_TURN_RATE` | 2.5 rad/s | soft cap on turn rate (see judgment calls) |
| `TURN_RATE_GAIN` | 3.0 (1/s) | proportional steering gain (turn rate = gain × heading error) |
| `REACHABILITY_MAX_ANGLE_DEG` | 75 | front-cone half-angle for reachability |
| `DEBOUNCE_SECONDS` | 0.150 | min time between accepted re-targets |
| `CLICK_PULSE_SECONDS` | 0.400 | click pulse fade duration |
| `RENDER_SCALE` | 4 | supersampling multiplier |
| `OUTPUT_WIDTH` / `_HEIGHT` | 320 / 240 | final frame resolution |
| `FOG_NEAR` / `FOG_FAR` | 4 / 12 | fog range |
| `FLICKER_ENABLED` / `_FREQUENCY_HZ` / `_INTENSITY` | false / 9 / 0.08 | subtle light flicker (currently off) |
| `LIGHT_SPACING` | 5.0 | overhead fixture spacing (world units) |
| `FIXTURE_LIGHT_INTENSITY` / `AMBIENT_LIGHT_INTENSITY` | 3.6 / 0.35 | point-light / ambient floor brightness |
| `WORKER_CONCURRENCY` | 4 | headless instance count |
| `EPISODE_FRAMES` | 600 | default frames per episode |

`RENDER_RESOLUTION` is always `OUTPUT_RESOLUTION × RENDER_SCALE` in both
dimensions. To raise the output resolution later, change **only**
`OUTPUT_WIDTH`/`OUTPUT_HEIGHT` — no other code needs to change.

## Testing

```bash
cd engine
node node_modules/vitest/vitest.mjs run      # all tests
```

- `tests/world.rules.test.ts` — deterministic rule-verification (Section 12.4):
  generates many sample worlds and checks axis-aligned (90°-increment)
  direction changes, constant hallway width, room dimension ranges, doorway
  spacing, ≥2 doorways per room, and that hallways only connect via
  bends/junctions.
- `tests/control.test.ts` — steering/jerk/debounce/resolution unit tests.
- `tests/headless.test.ts` — placeholder bot + manifest schema tests.

> Note: `npx` may be broken in some environments by an npm/Node version
> mismatch; invoke binaries directly as above if so.

## Judgment calls (documented per the spec)

Where the spec left a decision to engineering judgment, these choices were made
and are documented here rather than as scattered magic numbers:

- **Hallway ceiling height** (`HALLWAY_HEIGHT = 3.5`) — the spec fixes room
  ceiling ranges but is silent on hallway height; kept constant.
- **`MAX_TURN_RATE`** — an additional soft cap on turn rate beyond the mandated
  jerk threshold, so turning never becomes unrealistically fast during a long
  turn. `desiredOmega` is a proportional controller:
  `omega = clamp(turnRateGain · headingError, ±maxTurnRate)`, whose change is
  then limited by the jerk threshold; it converges monotonically without
  overshoot (not a bang-bang profile).
- **Reachability** — an approximate, conservative geometric feasibility test
  (front-cone angle + required-arc-length check); it is intentionally
  hand-tunable via `REACHABILITY_MAX_ANGLE_DEG`. Candidates are projected and
  reachability-tested against a doorway's **approach point** (the corner for a
  bend, the opening center otherwise) — never its `entryPoint`, whose waypoint
  sits past the corner and can falsely fail the front cone (see CHANGES-LOG).
- **Episode length** — `EPISODE_FRAMES = 600` (30 simulated seconds) by
  default; CLI-overridable.
- **Second-degree doorways** — implemented with deterministic solid black quads
  (not fog alone), so the "can't see through two doorways" rule (Section 5.5)
  is guaranteed, not just visually implied.
- **PNG capture** — uses the browser's native PNG encoder (no extra dependency);
  capture is done *before* each step so frame `i`'s image, timestamp
  (`i·timestep`), and held control signal are mutually consistent.
- **Surface materials** — procedurally generated canvas textures (no internet
  fetches), applied by the world generator as *lit* materials so the renderer's
  point/ambient lights illuminate the scene.
- **Hybrid-graphics routing** — headless Chrome is pointed at the NVIDIA dGPU
  via Vulkan; the GPU self-check accepts any hardware driver (NVIDIA/AMD/Intel)
  and rejects only software rasterizers.
- **Junction arity** (`junctionWeights`) — the spec names "2-way fork / 3-way
  T-junction / 4-way junction" without pinning the geometry. Implemented as:
  *2-way* = continue straight **or** turn one way (2 exits); *3-way* = turn
  left **or** right (a T, corridor ends); *4-way* = straight + left + right (a
  cross, 3 exits). No branch ever reverses 180° — the fourth arm of a cross is
  the entry the player came from — so every direction change stays a multiple
  of 90°.
- **Re-entering a disposed parent** — a hallway's `start` (entry) doorway whose
  parent entity has since been disposed becomes a permanent dead-end instead of
  regenerating a backwards hallway. In practice the player never targets it:
  it sits directly behind the heading, outside the forward reachability cone,
  so this is invisible during normal forward navigation.
- **Room exits face the player** — the spec's no-dead-ends rule (5.3) allows the
  extra doorway "on the same wall if there's room." That clause is what causes a
  real dead-end: the player walks into a room perpendicular to the entry wall and
  can only turn within the ~75° reachability cone (never 180°), so an extra
  doorway on the entry wall — or on a perpendicular wall near the entry corner —
  ends up behind them, and they walk straight into a blank wall. The generator
  therefore always places at least one extra doorway on the wall **opposite** the
  entry (always ahead), then optionally adds 0–2 more on the *other* walls for
  variety.
- **No extra doorway on the entry wall** — the entry wall is shared with the
  parent hallway, which has exactly one opening (the doorway the player came
  through). Cutting a second opening there would be occluded by the parent's solid
  wall: the player sees no doorway, but the floor is contiguous on both sides (the
  engine's collision is floor-based, not wall-based), so they "phase" through the
  phantom opening with no collision. The generator therefore never places an
  additional doorway on the wall it entered a room through — the player can never
  be forced to enter and exit through the same wall.
- **Bend doorways turn in two phases** — only a hallway *bend* (a doorway on the
  side wall of a corridor, the ~90° continuation) sets `turn: true` and an
  `approachPoint` (the corner). It is steered as: approach the corner (keeping
  straight down the corridor), then commit at `TURN_GATE_RADIUS` (2.5) and steer
  through a waypoint `TURN_WAYPOINT_DEPTH` (3) past the corner, which carries the
  player through the ~90° turn instead of crossing still heading straight and
  clipping the new corridor's far wall.
- **All other doorways are normal two-leg doorways** — a side doorway, a room
  exit, and a junction branch that turns left/right are steered the same way: on
  the approach leg steer at the **opening center** (the pursuit curve from
  anywhere in the convex entity toward the boundary point at the opening stays
  inside the entity, so it cannot clip a wall), then commit at `DOORWAY_GATE_RADIUS`
  (2.5) and steer through the entry point. `Doorway.gateRadius` exists in the type
  but is never assigned in the generator, so every doorway uses the two config
  radii above (see CHANGES-LOG for the correction note; the earlier "smaller gate
  for tight turns" description did not match the code).

## Out of scope (Phase 1)

No neural network code, no internet textures/props, no world-blending/biome
morphing, no persistent room memory, and no dependency of the trained model on
this engine. The click-bot personas (Stage 2) are coverage generators — they
make the signal information-rich — not learned or "intelligent" policies.
