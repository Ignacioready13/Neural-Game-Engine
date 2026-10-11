/**
 * CENTRAL CONFIG MODULE
 * =====================
 * Every tunable parameter described in the Phase 1 build spec lives here, in
 * one place, with a default and a one-line comment. Nothing else in the
 * codebase should contain a magic number that changes engine behavior — read
 * from this module instead.
 *
 * The resolved config object is frozen so no subsystem can mutate shared state
 * behind another subsystem's back.
 */

// ---------------------------------------------------------------------------
// Simulation timing & motion
// ---------------------------------------------------------------------------

/** Fixed simulated timestep, in simulated seconds per simulated frame.
 * 20Hz by default. Headless mode advances EXACTLY this much per frame,
 * independent of wall-clock render/capture time. */
export const SIM_TIMESTEP = 1 / 20;

/** Constant forward walk speed, world units per simulated second. The player
 * never stops; only steering direction changes. */
export const WALK_SPEED = 4.0;

// ---------------------------------------------------------------------------
// Camera / eye
// ---------------------------------------------------------------------------

/** Fixed eye height (camera Y offset off the floor). NEVER varies with a
 * room's ceiling height — only the ceiling itself moves. */
export const EYE_HEIGHT = 1.6;

/** Horizontal field of view, degrees. three.js uses vertical FOV, so the
 * renderer derives vertical FOV from this + aspect ratio. */
export const FOV_HORIZONTAL_DEG = 85;

// ---------------------------------------------------------------------------
// Hallways
// ---------------------------------------------------------------------------

/** Constant hallway width (world units), identical for every hallway. */
export const HALLWAY_WIDTH = 4.5;

/** Hallway ceiling height (world units). The spec fixes room ceiling heights
 * but leaves hallway ceiling height to judgment; kept constant for hallways. */
export const HALLWAY_HEIGHT = 3.5;

/** Random hallway segment length range (world units) before it terminates. */
export const HALLWAY_LENGTH_MIN = 8;
export const HALLWAY_LENGTH_MAX = 30;

/** Weighted choice of how a hallway segment terminates. Weights need not sum
 * to 1 (they are normalized). bend = 90deg turn (same logical hallway);
 * junction = fork into 2/3/4 directions; room = open into a room. */
export const HALLWAY_TERMINATION_WEIGHTS = {
  bend: 0.35,
  junction: 0.35,
  room: 0.30,
} as const;

/** Relative weights of 2-way fork / 3-way T-junction / 4-way junction. 2-way
 * and 3-way are most common; 4-way is rarer. */
export const JUNCTION_WEIGHTS = {
  twoWay: 0.45,
  threeWay: 0.40,
  fourWay: 0.15,
} as const;

/** Probability that a given hallway segment has any side doorways at all. */
export const HALLWAY_SIDE_DOORWAY_CHANCE = 0.6;

// ---------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------

/** Room footprint dimensions (world units), each randomized independently. */
export const ROOM_WIDTH_MIN = 10;
export const ROOM_WIDTH_MAX = 20;
export const ROOM_DEPTH_MIN = 10;
export const ROOM_DEPTH_MAX = 20;

/** Room ceiling height range (world units), independent of footprint.
 * ROOM_HEIGHT_MIN must be >= DOORWAY_HEIGHT so every doorway always fits its
 * wall (a doorway taller than its wall produces a negative-height lintel). */
export const ROOM_HEIGHT_MIN = 2.8;
export const ROOM_HEIGHT_MAX = 15;

// ---------------------------------------------------------------------------
// Walls
// ---------------------------------------------------------------------------

/** Wall extrusion depth (world units), visible as the doorway jamb / reveal
 * when walking through an opening. Extrudes OUTWARD (away from the entity's
 * interior) so it never narrows walkable space or changes floor-based
 * collision. */
export const WALL_THICKNESS = 0.5;

// ---------------------------------------------------------------------------
// Doorways
// ---------------------------------------------------------------------------

/** Doorway opening width (world units), along the wall it is cut into. */
export const DOORWAY_WIDTH = 2.7;

/** Doorway opening height (world units). Must be <= ROOM_HEIGHT_MIN and
 * <= HALLWAY_HEIGHT so every doorway always fits its wall. */
export const DOORWAY_HEIGHT = 2.8;

/** Minimum center-to-center spacing between two doorways on the same wall,
 * expressed as a multiple of DOORWAY_WIDTH. Default 2.5x -> 5 world units. */
export const DOORWAY_SPACING_FACTOR = 3;

/** How far the first side doorway must sit from a hallway's entry wall (along
 * the corridor). The player spawns 1.5 units in and can only steer into a
 * doorway inside the ~75 degree reachability cone, so the first doorway must be
 * placed far enough ahead to be clickable the moment the player appears — kept
 * a bit over the ~2.04 minimum so the player always has room to commit. */
export const SIDE_DOORWAY_START_CLEARANCE = 3.0;

// ---------------------------------------------------------------------------
// Steering / control (point-and-click)
// ---------------------------------------------------------------------------

/** Maximum angular acceleration ("jerk") allowed on the player's turning rate,
 * rad/s^2. This is the single constant the spec mandates; tune it by feel in
 * human-play mode until motion is smooth and never nauseating. */
export const JERK_THRESHOLD = 6.0;

/** Soft cap on instantaneous turn rate (rad/s), an extra safety knob on top of
 * the jerk limit so turning never becomes unrealistically fast even during a
 * long-duration turn. Documented as a judgment call. */
export const MAX_TURN_RATE = 2.5;

/** Proportional steering gain (1/s): desired turn rate = gain * heading error,
 * then jerk-limited. A P-controller converges monotonically (no bang-bang
 * overshoot), so turning is smooth and never oscillates. Tune by feel in
 * human-play mode. */
export const TURN_RATE_GAIN = 3.0;

/** Minimum time between accepted re-target events, seconds. Clicks arriving
 * inside this window are ignored for steering (but still show the click
 * pulse). */
export const DEBOUNCE_SECONDS = 0.150;

/** How long the click visual pulse fades out, seconds (spec: roughly 300-500ms). */
export const CLICK_PULSE_SECONDS = 0.400;

/** Reachability: maximum angle (deg) between current heading and the direction
 * to a candidate doorway for it to be considered reachable this frame. */
export const REACHABILITY_MAX_ANGLE_DEG = 75;

/** How close (world units) the player must get to a doorway's opening center
 * before committing to turn through the doorway (the "through" phase targets
 * the entry point just past the wall). */
export const DOORWAY_GATE_RADIUS = 2.5; //1.25 //Makes me swing wider before entering doorway,
//I like it but will clip doorway if not careful.

/** How close (world units) the player must get to a BEND's corner before
 * committing to the ~90 degree turn into the new corridor. A bend turns through
 * a full-width opening (the child corridor has no wall at the corner), so it has
 * room for a wider, earlier turn. Kept well under hallwayLengthMin (8) so the
 * player is still steering straight down the corridor when the turn begins. */
export const TURN_GATE_RADIUS = 2.5; //2.5

/** How far (world units) past a bend corner its steering waypoint sits, along the
 * new corridor's direction. A bend sits on a side wall, so the player approaches
 * it parallel to the wall and must turn ~90 degrees; a waypoint this far past the
 * corner is what carries them through the turn instead of letting them penetrate
 * at a shallow diagonal and clip the new corridor's far wall. */
export const TURN_WAYPOINT_DEPTH = 3;

// ---------------------------------------------------------------------------
// Rendering / resolution pipeline
// ---------------------------------------------------------------------------

/** Supersampling multiplier: RENDER_RESOLUTION = OUTPUT_RESOLUTION * this
 * scale in both dimensions. Default 4x. */
export const RENDER_SCALE = 4;

/** Final output resolution, pixels. The single value to raise later once
 * training looks good at low res — changing ONLY this (leaving RENDER_SCALE
 * alone) must change output resolution everywhere. */
export const OUTPUT_WIDTH = 320;
export const OUTPUT_HEIGHT = 240;

/** Fog: near/far planes (world units) and color. Keeps far geometry from
 * popping and helps the liminal look; second-degree doorways are additionally
 * filled with solid black geometry (see world generator) for determinism. */
export const FOG_COLOR = 0x0a0a0a;
export const FOG_NEAR = 4;
export const FOG_FAR = 12;

/** Subtle fluorescent-light flicker. Frequency (Hz) and intensity (0..1)
 * kept mild — "slightly unsettling", not seizure-inducing. */
export const FLICKER_ENABLED = false;
export const FLICKER_FREQUENCY_HZ = 9;
export const FLICKER_INTENSITY = 0.08;

/** Overhead fixture light spacing along hallways / grid in rooms (world units). */
export const LIGHT_SPACING = 5.0;

/** Base light intensity and the ambient floor so corners never go pitch black
 * (except where the fog/darkness rule explicitly requires solid black). */
export const FIXTURE_LIGHT_INTENSITY = 3.6;
export const AMBIENT_LIGHT_INTENSITY = 0.35;

// ---------------------------------------------------------------------------
// Headless data generation
// ---------------------------------------------------------------------------

/** Number of simultaneous headless Chromium instances (worker pool size).
 * Conservative default; tune up/down based on GPU memory + observed
 * throughput. */
export const WORKER_CONCURRENCY = 4;

/** Default number of simulated frames per episode (at 20Hz, 600 frames = 30
 * simulated seconds). Overridable via CLI. Documented as a judgment call. */
export const EPISODE_FRAMES = 600;

/** Default output root for generated datasets, relative to the repo root
 * /data directory (gitignored). Episodes land in <root>/episode_<uuid>/. */
export const DATA_OUTPUT_ROOT = 'data';

/** Explicit RNG seed for an episode; null = seed randomly per episode. */
export const EPISODE_SEED: number | null = null;

// ---------------------------------------------------------------------------
// Clipping failsafe (Stage 2, Section 4.11) — headless agent episodes only
// ---------------------------------------------------------------------------

/** Seconds of lead-in trimmed before the crash frame when the failsafe fires. */
export const CLIP_TRIM_SECONDS = 8;

/** If fewer than this many frames remain after trimming, the whole episode dir
 * is deleted (and the episode recorded in aborted.jsonl). */
export const MIN_KEPT_FRAMES = 60;

/** Warn + exit non-zero when the aborted share exceeds this (after 20 episodes). */
export const CLIP_ALERT_SHARE = 0.05;

/** Stop the batch early when the aborted share exceeds this (after 20 episodes). */
export const CLIP_STOP_SHARE = 0.25;

/** Consecutive slide-backstop frames that count as sustained contact (L2). */
export const CLIP_L2_RUN_LENGTH = 10;

/** Stage 2 follow-up (viewUnsafe label). A frame is flagged `viewUnsafe` when
 * the camera (a floor point at eye height, no collision radius) comes within
 * `camera near plane + this margin` of ANY rendered triangle of the active
 * entity or a maintained neighbour — i.e. close enough that the near plane
 * clips the surface, or the camera has crossed a single-sided wall plane and
 * sees through it. Label only: it never aborts or truncates an episode. The
 * loader must drop any training window that touches a flagged frame.
 *
 * 0.10 was chosen so the whole seed-0 h2->h4 doorway graze flags (f129-f135,
 * spanning f132 d=0.059 / f133 d=0.020 / f134 d=0.034). Over the full pilot it
 * flags 0.41% of frames (357 / 88,047 — well under the ~1% budget) spread over
 * 60% of episodes (90 / 150), essentially all of them 1-4 frame doorway grazes
 * plus one long wall-slide run. */
export const VIEW_UNSAFE_MARGIN = 0.1;

// ---------------------------------------------------------------------------
// Click-bot framework (Stage 2, Section 4.6)
// ---------------------------------------------------------------------------

/** Episode segment length range (frames at 20 Hz) between persona switches. */
export const BOT_SEGMENT_LENGTH_MIN = 100;
export const BOT_SEGMENT_LENGTH_MAX = 300;

/** Decisive persona: human-like reaction delay range (frames) before clicking. */
export const BOT_DECISIVE_DELAY_MIN = 4;
export const BOT_DECISIVE_DELAY_MAX = 25;

/** Late persona: long reaction delay range so the click lands when the door is
 * close and reachability is marginal. */
export const BOT_LATE_DELAY_MIN = 60;
export const BOT_LATE_DELAY_MAX = 180;

/** Click-jitter sigma range (normalized screen X) applied by door-aiming
 * personas (decisive/late/dither). */
export const BOT_JITTER_SIGMA_MIN = 0.03;
export const BOT_JITTER_SIGMA_MAX = 0.10;

/** Dither persona: frames before re-clicking toward a different door. */
export const BOT_DITHER_RETHINK_MIN = 10;
export const BOT_DITHER_RETHINK_MAX = 40;

/** Spammer persona: burst size and inter-burst silence ranges (frames). */
export const BOT_SPAMMER_BURST_MIN = 2;
export const BOT_SPAMMER_BURST_MAX = 8;
export const BOT_SPAMMER_SILENCE_MIN = 20;
export const BOT_SPAMMER_SILENCE_MAX = 80;

/** Uniform persona: click interval range (frames) — the old placeholder. */
export const BOT_UNIFORM_INTERVAL_MIN = 20;
export const BOT_UNIFORM_INTERVAL_MAX = 80;

/** Named persona-mixture profiles (weights need not sum to 1). */
export const MIXTURE_PROFILES = {
  mixture: { idle: 10, decisive: 30, late: 10, dither: 10, prober: 15, spammer: 5, uniform: 20 },
  uniformOnly: { uniform: 1 },
  doorAwareOnly: { decisive: 3, late: 1, dither: 1 },
} as const;

// ---------------------------------------------------------------------------
// Resolved config object
// ---------------------------------------------------------------------------

/** Derived render resolution (supersampled), computed from OUTPUT_* and RENDER_SCALE. */
export const RENDER_WIDTH = OUTPUT_WIDTH * RENDER_SCALE;
export const RENDER_HEIGHT = OUTPUT_HEIGHT * RENDER_SCALE;

export interface EngineConfig {
  // timing / motion
  simTimestep: number;
  walkSpeed: number;
  // camera
  eyeHeight: number;
  fovHorizontalDeg: number;
  // hallways
  hallwayWidth: number;
  hallwayHeight: number;
  hallwayLengthMin: number;
  hallwayLengthMax: number;
  hallwayTerminationWeights: typeof HALLWAY_TERMINATION_WEIGHTS;
  junctionWeights: typeof JUNCTION_WEIGHTS;
  hallwaySideDoorwayChance: number;
  // rooms
  roomWidthMin: number;
  roomWidthMax: number;
  roomDepthMin: number;
  roomDepthMax: number;
  roomHeightMin: number;
  roomHeightMax: number;
  // walls
  wallThickness: number;
  // doorways
  doorwayWidth: number;
  doorwayHeight: number;
  doorwaySpacingFactor: number;
  sideDoorwayStartClearance: number;
  // steering
  jerkThreshold: number;
  maxTurnRate: number;
  turnRateGain: number;
  debounceSeconds: number;
  clickPulseSeconds: number;
  reachabilityMaxAngleDeg: number;
  doorwayGateRadius: number;
  turnGateRadius: number;
  turnWaypointDepth: number;
  // rendering
  renderScale: number;
  outputWidth: number;
  outputHeight: number;
  renderWidth: number;
  renderHeight: number;
  fogColor: number;
  fogNear: number;
  fogFar: number;
  flickerEnabled: boolean;
  flickerFrequencyHz: number;
  flickerIntensity: number;
  lightSpacing: number;
  fixtureLightIntensity: number;
  ambientLightIntensity: number;
  // headless
  workerConcurrency: number;
  episodeFrames: number;
  dataOutputRoot: string;
  episodeSeed: number | null;
  // clipping failsafe
  clipTrimSeconds: number;
  minKeptFrames: number;
  clipAlertShare: number;
  clipStopShare: number;
  clipL2RunLength: number;
  /** Stage 2 follow-up: viewUnsafe clearance margin (world units). */
  viewUnsafeMargin: number;
  // click-bot personas
  botSegmentLengthMin: number;
  botSegmentLengthMax: number;
  botDecisiveDelayMin: number;
  botDecisiveDelayMax: number;
  botLateDelayMin: number;
  botLateDelayMax: number;
  botJitterSigmaMin: number;
  botJitterSigmaMax: number;
  botDitherRethinkMin: number;
  botDitherRethinkMax: number;
  botSpammerBurstMin: number;
  botSpammerBurstMax: number;
  botSpammerSilenceMin: number;
  botSpammerSilenceMax: number;
  botUniformIntervalMin: number;
  botUniformIntervalMax: number;
  mixtureProfiles: typeof MIXTURE_PROFILES;
}

export const CONFIG: Readonly<EngineConfig> = Object.freeze({
  simTimestep: SIM_TIMESTEP,
  walkSpeed: WALK_SPEED,
  eyeHeight: EYE_HEIGHT,
  fovHorizontalDeg: FOV_HORIZONTAL_DEG,
  hallwayWidth: HALLWAY_WIDTH,
  hallwayHeight: HALLWAY_HEIGHT,
  hallwayLengthMin: HALLWAY_LENGTH_MIN,
  hallwayLengthMax: HALLWAY_LENGTH_MAX,
  hallwayTerminationWeights: HALLWAY_TERMINATION_WEIGHTS,
  junctionWeights: JUNCTION_WEIGHTS,
  hallwaySideDoorwayChance: HALLWAY_SIDE_DOORWAY_CHANCE,
  roomWidthMin: ROOM_WIDTH_MIN,
  roomWidthMax: ROOM_WIDTH_MAX,
  roomDepthMin: ROOM_DEPTH_MIN,
  roomDepthMax: ROOM_DEPTH_MAX,
  roomHeightMin: ROOM_HEIGHT_MIN,
  roomHeightMax: ROOM_HEIGHT_MAX,
  wallThickness: WALL_THICKNESS,
  doorwayWidth: DOORWAY_WIDTH,
  doorwayHeight: DOORWAY_HEIGHT,
  doorwaySpacingFactor: DOORWAY_SPACING_FACTOR,
  sideDoorwayStartClearance: SIDE_DOORWAY_START_CLEARANCE,
  jerkThreshold: JERK_THRESHOLD,
  maxTurnRate: MAX_TURN_RATE,
  turnRateGain: TURN_RATE_GAIN,
  debounceSeconds: DEBOUNCE_SECONDS,
  clickPulseSeconds: CLICK_PULSE_SECONDS,
  reachabilityMaxAngleDeg: REACHABILITY_MAX_ANGLE_DEG,
  doorwayGateRadius: DOORWAY_GATE_RADIUS,
  turnGateRadius: TURN_GATE_RADIUS,
  turnWaypointDepth: TURN_WAYPOINT_DEPTH,
  renderScale: RENDER_SCALE,
  outputWidth: OUTPUT_WIDTH,
  outputHeight: OUTPUT_HEIGHT,
  renderWidth: RENDER_WIDTH,
  renderHeight: RENDER_HEIGHT,
  fogColor: FOG_COLOR,
  fogNear: FOG_NEAR,
  fogFar: FOG_FAR,
  flickerEnabled: FLICKER_ENABLED,
  flickerFrequencyHz: FLICKER_FREQUENCY_HZ,
  flickerIntensity: FLICKER_INTENSITY,
  lightSpacing: LIGHT_SPACING,
  fixtureLightIntensity: FIXTURE_LIGHT_INTENSITY,
  ambientLightIntensity: AMBIENT_LIGHT_INTENSITY,
  workerConcurrency: WORKER_CONCURRENCY,
  episodeFrames: EPISODE_FRAMES,
  dataOutputRoot: DATA_OUTPUT_ROOT,
  episodeSeed: EPISODE_SEED,
  clipTrimSeconds: CLIP_TRIM_SECONDS,
  minKeptFrames: MIN_KEPT_FRAMES,
  clipAlertShare: CLIP_ALERT_SHARE,
  clipStopShare: CLIP_STOP_SHARE,
  clipL2RunLength: CLIP_L2_RUN_LENGTH,
  viewUnsafeMargin: VIEW_UNSAFE_MARGIN,
  botSegmentLengthMin: BOT_SEGMENT_LENGTH_MIN,
  botSegmentLengthMax: BOT_SEGMENT_LENGTH_MAX,
  botDecisiveDelayMin: BOT_DECISIVE_DELAY_MIN,
  botDecisiveDelayMax: BOT_DECISIVE_DELAY_MAX,
  botLateDelayMin: BOT_LATE_DELAY_MIN,
  botLateDelayMax: BOT_LATE_DELAY_MAX,
  botJitterSigmaMin: BOT_JITTER_SIGMA_MIN,
  botJitterSigmaMax: BOT_JITTER_SIGMA_MAX,
  botDitherRethinkMin: BOT_DITHER_RETHINK_MIN,
  botDitherRethinkMax: BOT_DITHER_RETHINK_MAX,
  botSpammerBurstMin: BOT_SPAMMER_BURST_MIN,
  botSpammerBurstMax: BOT_SPAMMER_BURST_MAX,
  botSpammerSilenceMin: BOT_SPAMMER_SILENCE_MIN,
  botSpammerSilenceMax: BOT_SPAMMER_SILENCE_MAX,
  botUniformIntervalMin: BOT_UNIFORM_INTERVAL_MIN,
  botUniformIntervalMax: BOT_UNIFORM_INTERVAL_MAX,
  mixtureProfiles: MIXTURE_PROFILES,
});
