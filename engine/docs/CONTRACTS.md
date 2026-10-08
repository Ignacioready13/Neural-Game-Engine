# Module Contracts (Phase 1)

This file is the integration contract. Subagents implement the modules below
against these exact signatures; the orchestrator wires them together in
`src/sim/Engine.ts`. If you need to change a signature, coordinate — do not
silently diverge, or integration will break.

Coordinate system (see `src/core/types.ts`): X/Z are floor axes, +Y up.
Forward vector = `(cos(heading), 0, sin(heading))`; heading 0 = +X.
Cardinals: east=+X, west=-X, north=-Z, south=+Z.

## File ownership (no two subagents touch the same file)

- Orchestrator: `config/config.ts`, `src/core/*`, `src/sim/Engine.ts`,
  `src/app/main-interactive.ts`, `src/app/main-headless.ts`,
  `tests/world.rules.test.ts`, `index.html`, `headless.html`.
- Subagent A (world): `src/world/**`
- Subagent B (render): `src/render/**`
- Subagent C (control): `src/control/**`, `tests/control.test.ts`
- Subagent D (headless): `src/headless/**`

Shared imports: everyone imports `config/config.ts` (as `CONFIG` / `EngineConfig`)
and `src/core/types.ts`, `src/core/math.ts`, `src/core/rng.ts`.

## 1. World generation — `src/world/` (Subagent A)

```ts
// src/world/World.ts
export class World {
  constructor(seed: number, config: EngineConfig)
  /** Create the starting hallway (Section 5.6) and return it. */
  init(): Entity
  /** Ensure every doorway-neighbor of `activeId` is generated; dispose every
   *  entity that is neither active nor a direct neighbor of active. Returns
   *  what changed so the renderer can sync. */
  maintain(activeId: string): { generated: Entity[]; disposed: string[] }
  get(id: string): Entity | undefined
  /** All currently-live entities. */
  all(): Entity[]
  /** Doorways of the ACTIVE entity's generated neighbors whose own doorways
   *  lead to not-yet-generated entities (second-degree => opaque black). */
  blackDoorways(activeId: string): Doorway[]
  /** The entity whose interior contains a floor point (for enter/exit tests). */
  entityAt(x: number, z: number): EntityDescriptor | undefined
}
```

`Entity` = `{ descriptor: EntityDescriptor; group: THREE.Group }`.
`Entity.group` is the instantiated three.js Group (built by A via a helper
`buildEntityGroup(descriptor, config)` in `src/world/geometry.ts`). It is
created ONCE at generation and disposed (geometry + materials) on `disposed`.

Descriptor rules enforced (tested by `tests/world.rules.test.ts`):
- Hallway segments axis-aligned; direction changes are multiples of 90 degrees
  (bends turn exactly 90 degrees; a junction branch may continue straight).
- Constant hallway width. Length in [min,max].
- Rooms rectangular; width/depth/height in ranges; ceiling height independent.
- Doorways: min center-to-center spacing `doorwaySpacingFactor * doorwayWidth`
  on the same wall/side; every room has >= 2 doorways (entry + at least 1);
  hallway side doorways only ever lead to rooms; hallway↔hallway only via
  bend/junction.
- Starting hallway has no entry doorway.

## 2. Rendering — `src/render/` (Subagent B)

```ts
// src/render/Renderer.ts
export class Renderer {
  constructor(canvas: HTMLCanvasElement, config: EngineConfig)
  /** Position the camera from the player (BEFORE projectToScreenX / render). */
  positionCamera(player: PlayerState): void
  /** Project a world point to normalized screen X (-1..1); null if behind the
   *  camera or not projectable. */
  projectToScreenX(wx: number, wz: number, wy: number): number | null
  /** Sync live entities into the scene (add new groups, remove disposed). */
  setEntities(entities: Entity[]): void
  /** Set the current second-degree doorway black-fill geometry. */
  setBlackDoorways(doorways: Doorway[]): void
  /** Draw one frame (scene -> supersampled target -> box-filter downsample -> screen). */
  render(snapshot: FrameSnapshot): void
  /** Return the final OUTPUT-resolution frame as a PNG data URL. */
  captureFrame(): { dataUrl: string; width: number; height: number }
  /** Advance the light-flicker state by dt (simulated seconds). */
  updateFlicker(dt: number): void
  dispose(): void
}
```

Requirements:
- Resolution pipeline: render at `renderWidth x renderHeight`, downsample to
  `outputWidth x outputHeight` using an area/box filter (NOT nearest-neighbor).
  A fullscreen quad that averages a `renderScale x renderScale` texel box is
  the intended approach.
- Materials: seamless tiling for floor/wall/ceiling, generated procedurally
  (canvas textures — no internet fetches). No obvious seams or stretching.
- Lighting: overhead fixtures (point/area lights on a grid) + gentle flicker;
  ambient floor so corners aren't pitch black.
- Fog for depth; second-degree doorways additionally get deterministic solid
  black quads (from `setBlackDoorways`).
- FOV: config `fovHorizontalDeg` is horizontal; derive vertical from aspect.

## 3. Control — `src/control/` (Subagent C)

```ts
// src/control/steering.ts
export class Steering {
  constructor(config: EngineConfig)
  /** Feed a new click (screen X, -1..1). Debounce applies; always returns the
   *  pulse info (every click pulses, accepted or not). */
  handleClick(screenX: number, simTime: number): { accepted: boolean }
  /** Recompute the current target each frame from live candidates.
   *  Uses held lastClickedX if set, else screen center (0). */
  update(candidates: CandidateDoorway[], simTime: number): string | null
  /** Advance the player one timestep toward a target point (null = straight). */
  step(player: PlayerState, target: Vec2 | null, dt: number): PlayerState
  /** Feasibility: can a jerk-limited turn from `player` reach `entry`? */
  isReachable(player: PlayerState, entry: Vec2): boolean
  // state for snapshot:
  readonly lastClickedX: number | null
  readonly targetDoorwayId: string | null
  readonly lastClickTime: number
  readonly lastClickScreenX: number | null
}

// src/control/doorwayResolution.ts
export function resolveTarget(
  candidates: CandidateDoorway[],
  clickX: number | null,
  previousTargetId: string | null,
): string | null;
```

Algorithm (Section 6.2/6.6):
- Sort candidates by |screenX - referenceX| ascending (referenceX = clickX or 0).
- Start at the nearest; if unreachable, continue outward preferring candidates
  on the SAME side of the reference as the nearest (before crossing to the
  other side), taking the first reachable one.
- If none reachable, return previousTargetId if it is still a live candidate,
  else null (caller keeps straight-ahead steering).

Motion (Section 6.3): kinematic steering — heading θ, angular velocity ω.
Each step: ω += clamp(desiredOmega - ω, ±jerkThreshold·dt); |ω| capped at
`maxTurnRate`; θ += ω·dt; position += forward·walkSpeed·dt. `desiredOmega` is a
proportional controller toward the target heading (bang-bang sqrt profile is
fine). Position/heading/ω are all continuous across retargets — never snap.

`isReachable`: target ahead (within ~`reachabilityMaxAngleDeg` of heading), and
turning to face it is feasible within the remaining straight-line distance at
the jerk limit. Keep it conservative + documented; it is hand-tunable.

## 4. Headless pipeline — `src/headless/` (Subagent D)

```ts
// src/headless/staticServer.ts
export function serveDir(dir: string, port: number): Promise<{ url: string; close(): Promise<void> }>
// src/headless/driver.ts
export interface EpisodeResult { episodeId: string; frameCount: number; dir: string }
export function assertGpuAcceleration(page: Page): Promise<string> // throws if software
export async function runEpisode(opts: {...}): Promise<EpisodeResult>
// src/headless/workerPool.ts
export async function runPool(spec: PoolSpec): Promise<EpisodeResult[]>
// src/headless/bot.ts
export function makePlaceholderBot(seed: number): (frameIndex: number) => number | null
// src/headless/manifest.ts
export function writeManifest(dir: string, manifest: EpisodeManifest): Promise<void>
// src/headless/cli.ts  — arg parsing + orchestration entry
```

The page (`headless.html` -> `src/app/main-headless.ts`) exposes a global:
```ts
window.__backrooms = {
  init(configOverride?): Promise<void>
  step(clickX: number | null): void        // one fixed timestep
  captureFrame(): { dataUrl; width; height }
  getState(): { player, activeEntityId, frameIndex, simulatedTime, controlSignalX }
  gpuInfo(): { renderer: string; vendor: string } // WebGL UNMASKED_RENDERER/VENDOR
};
```
`runEpisode` drives: init -> per frame, ask the bot for a clickX, call step,
capture the frame, write `frames/%06d.png` and append a `FrameRecord`, then
write `manifest.json`. It must verify GPU acceleration (fail loudly on
SwiftShader/llvmpipe/software) before starting, and advance exactly one fixed
simulated timestep per frame regardless of wall-clock time.

## 5. Orchestration — `src/sim/Engine.ts` (orchestrator)

`Engine` owns a `World`, a `Steering`, a `Renderer`, player state, and sim time,
and exposes `submitClick(screenX)` + `step()` + `render()` + `captureFrame()`
used identically by both app mains. Per step: position camera -> collect live
candidates -> project to screen X -> reachability -> resolve target -> advance
player -> detect entity crossing -> `world.maintain` -> sync renderer meshes &
black doorways -> update flicker -> build `FrameSnapshot`.

`src/app/main-interactive.ts`: rAF + fixed-timestep accumulator; mouse click
handler normalizes canvas X to -1..1 and calls `engine.submitClick`; renders each
frame. `src/app/main-headless.ts`: builds `window.__backrooms` over an `Engine`.

## Key judgment calls (document these in the root README)

- Hallway ceiling height constant `HALLWAY_HEIGHT` (spec silent; kept constant).
- `MAX_TURN_RATE` soft cap in addition to the mandated jerk threshold.
- `EPISODE_FRAMES` default 600 (30 simulated seconds); CLI-overridable.
- Reachability is an approximate, conservative geometric feasibility test.
- Headless capture uses the browser's PNG encoder (no extra dep).
