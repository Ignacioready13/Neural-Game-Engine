# NeuralGameEngine — Phase 1

A procedurally generated, backrooms-style first-person navigation engine. Its
only purpose is to generate a large, high-quality dataset of
`(sequence of frames, sequence of control signals)` pairs that will later train
a neural network (the "dynamics model"). The classical engine is a *teacher*,
not a permanent game: every design decision favors clean, consistent,
information-rich training signal.

The full build specification lives in `backrooms-engine-build-prompt.txt`.

## Two-phase architecture

1. **`/engine`** (this phase) — a classical, fully deterministic 3D engine
   (TypeScript + three.js) with two operating modes that share the exact same
   core:
   - **Human play** — point-and-click navigation through procedurally generated
     hallways and rooms, via the Vite dev server.
   - **Headless generation** — the same engine driven by a minimal placeholder
     bot in headless Chromium (Puppeteer), writing frames + labels to `/data`.
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
    /headless/ Puppeteer driver, worker pool, placeholder bot, manifest writer
  /tests/     vitest rule-verification + unit tests
  /docs/      CONTRACTS.md — subsystem integration contract
/neural/      empty placeholder for the future neural codebase
/data/        default output root for generated datasets (gitignored)
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

## Headless data generation

```bash
cd engine
npm run build        # produce dist/ (index.html + headless.html)
npm run generate     # run one episode with the placeholder bot, write to /data
```

`npm run generate` forwards arguments to `src/headless/cli.ts`:

```
--concurrency <n>   simultaneous headless Chromium instances (default 4)
--episodes <n>      number of episodes to generate (default 1)
--frames <n>        frames per episode (default 600 = 30 simulated seconds @20Hz)
--seed <n>          RNG seed (default: random per episode)
--output <dir>      output root (default: <repo>/data)
--serve-port <n>    port for the built-page static server (default 4587)
--build             rebuild dist/ before generating
```

Each episode writes to `data/episode_<uuid>/`:

```
episode_<uuid>/
  frames/000000.png ...   lossless PNGs at OUTPUT_RESOLUTION
  manifest.json           episode + per-frame metadata (see below)
```

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
writing its own episode directory. The default is conservative (4); tune it up
or down based on your GPU memory and observed throughput. It is expected (and
fine) that large-scale generation is slow in wall-clock terms — what matters is
that each episode's simulated motion is correct and consistent, which the fixed
timestep guarantees regardless of load.

## Dataset format (`manifest.json`)

`manifest.json` is self-describing and unambiguous:

```jsonc
{
  "episodeId": "…uuid…",
  "renderWidth": 1280, "renderHeight": 960,   // supersampled render size
  "outputWidth": 320,  "outputHeight": 240,   // final frame size
  "simTimestep": 0.05,                          // seconds per simulated frame
  "seed": 12345,
  "config": { /* jerkThreshold, hallway/room size ranges, doorway spacing,
                 debounceSeconds, walkSpeed, fovHorizontalDeg, … */ },
  "frames": [
    {
      "frameIndex": 0,
      "simulatedTime": 0.0,                     // frameIndex * simTimestep
      "imagePath": "frames/000000.png",
      "controlSignalX": 0.0,                    // held click-X, -1..1 (the ONE signal)
      "playerX": 0.5, "playerZ": 1.5,           // debug-only, not fed to the model
      "playerHeading": 1.5708,
      "currentEntityId": "h0"
    }
    // …
  ]
}
```

`controlSignalX` is the **held** normalized screen-space click X (step-function;
`0` = screen-center default before the first click). It is the one and only
per-frame signal the future model is trained on, and it is sampled on every
simulated frame. `playerX/Z/heading` and `currentEntityId` are debug fields for
verifying engine behavior — they are not intended to be fed to the network.
Frames are stored as PNG (lossless) so compression artifacts don't pollute the
training signal.

## Config

Every tunable lives in `engine/config/config.ts`, with an inline comment. Key
values and their defaults:

| Constant | Default | Controls |
|---|---|---|
| `SIM_TIMESTEP` | 1/20 s | fixed simulated timestep (20 Hz) |
| `WALK_SPEED` | 2.0 | forward speed, world units/s |
| `EYE_HEIGHT` | 1.6 | camera height (fixed regardless of room height) |
| `FOV_HORIZONTAL_DEG` | 85 | horizontal field of view |
| `HALLWAY_WIDTH` / `HALLWAY_HEIGHT` | 4.0 / 3.0 | corridor width / ceiling height |
| `HALLWAY_LENGTH_MIN` / `_MAX` | 8 / 24 | segment length range |
| `HALLWAY_TERMINATION_WEIGHTS` | bend .35 / junction .35 / room .30 | how segments end |
| `JUNCTION_WEIGHTS` | 2-way .45 / 3-way .40 / 4-way .15 | fork arity |
| `ROOM_WIDTH/DEPTH_MIN/MAX` | 4–10 | room footprint ranges |
| `ROOM_HEIGHT_MIN/MAX` | 2.5–5 | room ceiling range |
| `DOORWAY_WIDTH` / `_HEIGHT` | 2.0 / 2.2 | opening size |
| `DOORWAY_SPACING_FACTOR` | 2.5 | min center-to-center = 2.5× doorway width |
| `DOORWAY_GATE_RADIUS` | 1.25 | proximity to commit to turning through a doorway |
| `TURN_GATE_RADIUS` | 2.5 | proximity to a bend corner to commit to the ~90° turn |
| `DOORWAY_TURN_GATE_RADIUS` | 1.5 | turn gate radius for tight 2-wide turns (side doorways, junction turns) |
| `TURN_WAYPOINT_DEPTH` | 3.0 | how far past a turn corner/opening its waypoint sits |
| `JERK_THRESHOLD` | 6.0 rad/s² | max angular acceleration ("jerk") |
| `MAX_TURN_RATE` | 2.5 rad/s | soft cap on turn rate (see judgment calls) |
| `DEBOUNCE_SECONDS` | 0.150 | min time between accepted re-targets |
| `CLICK_PULSE_SECONDS` | 0.400 | click pulse fade duration |
| `RENDER_SCALE` | 4 | supersampling multiplier |
| `OUTPUT_WIDTH` / `_HEIGHT` | 320 / 240 | final frame resolution |
| `FOG_NEAR` / `FOG_FAR` | 8 / 40 | fog range |
| `FLICKER_*` | on, 9 Hz, 0.08 | subtle light flicker |
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

- **Hallway ceiling height** (`HALLWAY_HEIGHT = 3.0`) — the spec fixes room
  ceiling ranges but is silent on hallway height; kept constant.
- **`MAX_TURN_RATE`** — an additional soft cap on turn rate beyond the mandated
  jerk threshold, so turning never becomes unrealistically fast during a long
  turn. `desiredOmega` uses a bang-bang profile
  `sign(err)·sqrt(2·jerk·|err|)`.
- **Reachability** — an approximate, conservative geometric feasibility test
  (front-cone angle + required-arc-length check); it is intentionally
  hand-tunable via `REACHABILITY_MAX_ANGLE_DEG`.
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
- **Turn doorways turn in two phases** — a doorway on a *side* wall (a hallway
  bend, a side doorway into a room, or a junction branch that turns left/right)
  sits on a wall the player drives parallel to, so it is not a normal "approach the
  opening then step through." These are steered as: approach a point on the
  corridor centerline (a bend: the corner; a side doorway: centerline abreast of
  the opening; a junction turn: the node center) so the player keeps driving
  straight down the corridor, then commit to the ~90° turn at a gate radius and
  steer through a waypoint `TURN_WAYPOINT_DEPTH` (3) past the opening/corner. This
  is what makes the player actually turn instead of crossing still heading straight
  and clipping the neighbor's far wall.
- **Tight turns use a smaller gate** — a bend turns through a full-width opening
  (the child corridor has no wall at the corner), so it commits early at
  `TURN_GATE_RADIUS` (2.5). A side doorway or junction turn threads a
  `DOORWAY_WIDTH` (2) opening with wall on both sides; committing that early turns
  the player toward the wall beside the opening, and they penetrate it at a shallow
  (~28–45°) diagonal and clip the wall perpendicular to the doorway. Those set
  `doorwayTurnGateRadius` (1.5), committing late enough that the turn arc stays
  inside the 2-wide opening and the player crosses essentially perpendicular to it.

## Out of scope (Phase 1)

No neural network code, no intelligent bot policies (only the minimal
placeholder bot), no internet textures/props, no world-blending/biome morphing,
no persistent room memory, and no dependency of the trained model on this
engine.
