/**
 * SHARED CORE TYPES
 * =================
 * The contract every subsystem builds against. World generation, control, and
 * simulation all import from here; rendering consumes the same structures.
 *
 * Coordinate system (three.js convention):
 *   - X / Z are the horizontal (floor) axes, Y is up.
 *   - A plan-view position is a Vec2 {x, z}.
 *   - Player heading is an angle in radians around +Y. Forward direction is
 *     (cos(heading), 0, sin(heading)); heading 0 faces +X (east), and positive
 *     heading rotates counter-clockwise seen from above (+Y).
 *
 * Cardinal directions use unit vectors in the XZ plane:
 *   east = +X, west = -X, north = -Z, south = +Z.
 */

/** A point in the horizontal (floor) plane. */
export interface Vec2 {
  x: number;
  z: number;
}

/** The four cardinal directions (axis-aligned, 90-degree increments). */
export type Cardinal = 'east' | 'west' | 'north' | 'south';

/** Unit vector in the XZ plane for a cardinal direction. */
export const CARDINAL_VEC: Record<Cardinal, Vec2> = {
  east: { x: 1, z: 0 },
  west: { x: -1, z: 0 },
  north: { x: 0, z: -1 },
  south: { x: 0, z: 1 },
};

/** The opposite cardinal direction. */
export const OPPOSITE: Record<Cardinal, Cardinal> = {
  east: 'west',
  west: 'east',
  north: 'south',
  south: 'north',
};

export type EntityKind = 'hallway' | 'room';

/**
 * A doorway is a rectangular opening in one wall of an entity, leading to a
 * neighboring entity. It carries enough geometry for BOTH steering (entry
 * point) and rendering (opening position/size + facing for the black-fill
 * effect on second-degree doorways).
 */
export interface Doorway {
  /** Stable unique id for this doorway within the run. */
  id: string;
  /** World-space center of the opening, on the floor plane. */
  x: number;
  z: number;
  /** Opening width (along the wall) and height (vertical). */
  width: number;
  height: number;
  /** Outward-facing wall normal (unit Vec2), i.e. the direction you walk to
   * leave the entity through this doorway. */
  facing: Cardinal;
  /** Which wall of the owning entity this doorway sits in.
   *  - hallway: 'left' | 'right' (relative to travel direction)
   *  - room:    a Cardinal (which wall of the box) */
  side: string;
  /** The id of the neighbor entity this doorway leads to. Assigned at the
   * moment that neighbor is generated; empty until then. */
  targetEntityId: string;
  /** True once the neighbor entity has actually been generated. */
  targetGenerated: boolean;
  /** True for a room's entry doorway: the one it was generated through, whose
   * wall is coplanar (back to back) with the parent hallway's wall. The parent
   * owns that seam's wall-thickness reveal, so the room does not build one. */
  entry?: boolean;
  /** World position the player steers toward to pass through (floor-level). For
   * a normal doorway this is just past the opening; for a bend doorway (`turn`) it
   * is a waypoint several units past the corner so the ~90 degree turn completes
   * before the player crosses into the neighbor. */
  entryPoint: Vec2;
  /** For a turning doorway (a hallway bend, see `turn`): the point to steer
   * toward while still inside the parent entity, so the player keeps driving
   * straight down the corridor — instead of drifting toward the turn-side wall —
   * until they are abreast of the corner, then commits to the ~90 degree turn. A
   * hallway bend sets this to the corner. */
  approachPoint?: Vec2;
  /** For a turning doorway (a hallway bend, see `turn`): override for the turn
   * gate radius (the distance from `approachPoint` at which the player commits to
   * the through phase). A bend turns through a full-width corridor and leaves this
   * unset to use TURN_GATE_RADIUS. */
  gateRadius?: number;
  /** True when the player approaches this doorway PARALLEL to the wall it sits in
   * and must turn ~90 degrees to pass through it perpendicular. Applies to hallway
   * bends (the end doorway on a side wall). For these, `entryPoint` is a waypoint
   * past the corner (see navigation.doorwayTargetPoint) that carries the player
   * through the turn instead of clipping the far wall. */
  turn?: boolean;
}

/**
 * Pure-data description of a generated entity. This is what the rule-
 * verification tests inspect — it contains no three.js objects, so the tests
 * run in plain Node without a WebGL context.
 */
export interface EntityDescriptor {
  id: string;
  kind: EntityKind;

  // ----- hallway only -----
  /** Axis the hallway centerline runs along: 'x' or 'z'. */
  axis?: 'x' | 'z';
  /** Centerline endpoints (floor plane). */
  start?: Vec2;
  end?: Vec2;
  /** Direction of travel (cardinal). */
  travelDirection?: Cardinal;
  /** Corridor width (world units) — constant, = config.hallwayWidth. Recorded
   * so rule-verification tests can assert width is constant across hallways. */
  corridorWidth?: number;
  /** How this hallway segment terminates. */
  termination?: 'bend' | 'junction' | 'room' | 'start';
  /** For a hallway that entered via a bend corner: which side (relative to this
   * hallway's travel direction) the parent corridor lies on. Side rooms on that
   * side must back off W/2 from the start so they don't poke back into the
   * parent corridor. Only set on bend children. */
  startCorridorSide?: 'left' | 'right';

  // ----- room only -----
  /** Room footprint center, width (X), depth (Z), ceiling height. */
  center?: Vec2;
  width?: number;
  depth?: number;
  ceilingHeight?: number;

  /** All doorways cut into this entity (including the one the player entered
   * through). */
  doorways: Doorway[];
}

/**
 * Fully-instantiated entity = pure descriptor + its three.js scene object.
 *
 * NOTE: this module otherwise stays free of three.js so the rule-verification
 * tests can run descriptors in plain Node; `group` is typed loosely as a plain
 * object to avoid a hard import here. The world generator casts it to a real
 * THREE.Group when it builds meshes, and the renderer receives the real group.
 */
export interface Entity {
  descriptor: EntityDescriptor;
  /** three.js Group containing this entity's geometry (real THREE.Group). */
  group: unknown;
}

/** Player state at a given simulated frame (see types for units). */
export interface PlayerState {
  /** Floor-plane position. */
  x: number;
  z: number;
  /** Heading angle (radians); forward = (cos, 0, sin). */
  heading: number;
  /** Angular velocity (rad/s). */
  angularVelocity: number;
}

/** A candidate steering target as seen by the doorway-resolution algorithm. */
export interface CandidateDoorway {
  doorwayId: string;
  /** On-screen X in normalized device coords, -1..1 (ignoring Y, per spec). */
  screenX: number;
  /** Whether a jerk-limited spline to this doorway is feasible this frame. */
  reachable: boolean;
  /** On-screen X distance from the click position (for tie-breaking). */
  screenDistance: number;
}

/** Steering/target state carried between frames (the "held" control signal). */
export interface SteeringState {
  /** Last accepted click X, normalized -1..1; null before any click. */
  lastClickedX: number | null;
  /** Currently selected target doorway id; null when none yet. */
  targetDoorwayId: string | null;
  /** Simulated time of the last click of any kind (for the pulse UI). */
  lastClickTime: number;
  /** Screen X of the last click (for the pulse UI). */
  lastClickScreenX: number | null;
}

/** Snapshot handed to the renderer each frame (browser + headless). */
export interface FrameSnapshot {
  /** Entities that should be visible (active + direct neighbors). */
  entities: Entity[];
  /** Descriptor of the entity the player currently occupies. */
  activeEntity: EntityDescriptor;
  /** Doorways that must render as opaque black (second-degree openings). */
  blackDoorways: Doorway[];
  player: PlayerState;
  steering: SteeringState;
  /** Pending click pulses to draw as UI feedback (screen X + age in seconds). */
  clickPulses: Array<{ screenX: number; ageSeconds: number }>;
  simulatedTime: number;
  frameIndex: number;
}

/**
 * One frame's record in the training manifest (schema v2).
 *
 * Alignment rule: the action for the transition from frame i to frame i+1 is
 * `actionX[i]`. `controlSignalX[i]` is the signal held AT capture time; training
 * must use `actionX`.
 */
export interface FrameRecord {
  frameIndex: number;
  simulatedTime: number;
  imagePath: string;
  /** The held control signal AT capture time: normalized screen-space click X,
   * -1..1 (0 before the first accepted click). */
  controlSignalX: number;
  /** Held signal in effect during the step i -> i+1 (i.e. controlSignalX on the
   * next frame). Null on the final frame (no step follows). */
  actionX: number | null;
  /** Whether >=1 click had been accepted as of the step i -> i+1 (false = the
   * engine is in its "never clicked" default mode). Null on the final frame. */
  actionHasClicked: boolean | null;
  /** Raw click X submitted for the step i -> i+1, or null. */
  clickX: number | null;
  /** true = engine accepted the click; false = debounce ignored it; null = no
   * click submitted for that step. */
  clickAccepted: boolean | null;
  // Lightweight context (NOT fed to the neural network):
  playerX: number;
  playerZ: number;
  playerHeading: number;
  currentEntityId: string;
  /** Stage 2 follow-up. True when the camera at capture time was close enough to
   * a rendered wall/jamb that the view is compromised (a surface inside the
   * camera near plane + margin, or the camera behind a single-sided wall plane
   * = see-through). This is a LABEL ONLY — it never aborts or truncates an
   * episode. Data loaders MUST drop any training window that contains a frame
   * with `viewUnsafe: true` (including the window's action target frame). */
  viewUnsafe: boolean;
}

/** One frame's debug/telemetry record (schema v2), kept separable from the
 * minimal training fields so a data loader can ignore them cheaply. */
export interface DebugFrame {
  frameIndex: number;
  playerOmega: number;
  lastClickedX: number | null;
  targetDoorwayId: string | null;
  targetPointX: number | null;
  targetPointZ: number | null;
  targetPhase: string;
  collisionEvents: number;
  wallOverlap: boolean;
  candidates: Array<{ doorwayId: string; screenX: number; reachable: boolean; screenDistance: number }>;
  /** Persona active when this step's click decision was made. */
  persona: string;
  /** Doorway the bot aimed at for this step's click, or null. */
  intentDoorwayId: string | null;
  /** Whether the wall-slide backstop engaged during the step i -> i+1 (aligned
   * with actionX). */
  slid: boolean;
}

/** Episode-level manifest metadata (schema v2, Section 10.2). */
export interface EpisodeManifest {
  schemaVersion: 2;
  episodeId: string;
  seed: number;
  /** Split name from the run plan (train / val / test). */
  split: string;
  /** Mixture profile name this episode was generated under. */
  profile: string;
  renderWidth: number;
  renderHeight: number;
  outputWidth: number;
  outputHeight: number;
  simTimestep: number;
  // Provenance (Section 4.4):
  configHash: string;
  engineCommit: string;
  dirty: boolean;
  engineSourceHash: string;
  config: {
    jerkThreshold: number;
    hallwayWidth: number;
    hallwayLengthMin: number;
    hallwayLengthMax: number;
    roomWidthMin: number;
    roomWidthMax: number;
    roomDepthMin: number;
    roomDepthMax: number;
    roomHeightMin: number;
    roomHeightMax: number;
    doorwayWidth: number;
    doorwaySpacingFactor: number;
    debounceSeconds: number;
    walkSpeed: number;
    fovHorizontalDeg: number;
    /** Stage 2 follow-up: the `viewUnsafe` clearance margin (world units). The
     * flag fires when a rendered surface is closer than camera near + this. */
    viewUnsafeMargin: number;
  };
  frames: FrameRecord[];
  debugFrames: DebugFrame[];
  // Clipping failsafe (Section 4.11):
  truncated: boolean;
  truncationReason: string | null;
  crashFrameIndex: number | null;
  trimmedFrames: number;
  /** Written LAST; a partial episode never carries it. */
  complete: boolean;
}

/** One line of the plan's `aborted.jsonl` (Section 4.11). */
export interface AbortRecord {
  episodeId: string;
  seed: number;
  split: string;
  persona: string;
  level: 'L1' | 'L2';
  reason: string;
  crashFrameIndex: number;
  framesKept: number;
  clickSequence: Array<{ frameIndex: number; clickX: number }>;
  stateDump: Array<{
    frameIndex: number;
    x: number;
    z: number;
    heading: number;
    entityId: string;
    targetDoorwayId: string | null;
  }>;
}
