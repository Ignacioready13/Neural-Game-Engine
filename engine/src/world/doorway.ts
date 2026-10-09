/**
 * Doorway construction and placement logic.
 *
 * Deterministic — all randomness comes from the passed Rng. Doorway placement
 * enforces the minimum center-to-center spacing rule (doorwaySpacingFactor *
 * doorwayWidth) between doorways on the same wall/side.
 */
import type { Doorway, Cardinal, Vec2 } from '../core/types';
import { CARDINAL_VEC } from '../core/types';
import type { EngineConfig } from '../../config/config';
import type { Rng } from '../core/rng';
import { vecRotate90, vecRotateNeg90, vecAdd, vecScale, vecLen, vecSub } from '../core/math';

/** Stable id counters shared by the whole world generation run. */
export interface Ids {
  hallways: number;
  rooms: number;
  doorways: number;
}

export function makeIds(): Ids {
  return { hallways: 0, rooms: 0, doorways: 0 };
}

export function nextEntityId(ids: Ids, kind: 'hallway' | 'room'): string {
  if (kind === 'hallway') {
    ids.hallways += 1;
    return `h${ids.hallways}`;
  }
  ids.rooms += 1;
  return `r${ids.rooms}`;
}

export function nextDoorwayId(ids: Ids): string {
  ids.doorways += 1;
  return `d${ids.doorways}`;
}

/** Minimum center-to-center spacing between doorways on the same wall. */
export function minSpacing(config: EngineConfig): number {
  return config.doorwaySpacingFactor * config.doorwayWidth;
}

export interface DoorwaySpec {
  x: number;
  z: number;
  width: number;
  height: number;
  facing: Cardinal;
  side: string;
  targetEntityId: string;
  targetGenerated: boolean;
  /** Override the default entryPoint (opening + outward half-width). Turning
   * doorways supply a waypoint past the opening/corner here. */
  entryPoint?: Vec2;
  /** For bend doorways: the point to steer toward while still in the parent
   * corridor (see Doorway.approachPoint). */
  approachPoint?: Vec2;
  /** Override the turn gate radius for this doorway (see Doorway.gateRadius). */
  gateRadius?: number;
  /** Mark a perpendicular-turn doorway (see Doorway.turn). */
  turn?: boolean;
}

/**
 * Create a doorway. `entryPoint` is the steering target: the floor point just
 * outside the opening, offset half the opening width along the outward normal —
 * unless the spec overrides it (bends point it past the corner).
 */
export function makeDoorway(ids: Ids, spec: DoorwaySpec): Doorway {
  const facingVec = CARDINAL_VEC[spec.facing];
  const entryOffset = spec.width * 0.5;
  return {
    id: nextDoorwayId(ids),
    x: spec.x,
    z: spec.z,
    width: spec.width,
    height: spec.height,
    facing: spec.facing,
    side: spec.side,
    targetEntityId: spec.targetEntityId,
    targetGenerated: spec.targetGenerated,
    entryPoint:
      spec.entryPoint ?? {
        x: spec.x + facingVec.x * entryOffset,
        z: spec.z + facingVec.z * entryOffset,
      },
    approachPoint: spec.approachPoint,
    gateRadius: spec.gateRadius,
    turn: spec.turn,
  };
}

export interface SidePlacementOpts {
  /** Centerline start of the hallway. */
  start: Vec2;
  /** Centerline end of the hallway. */
  end: Vec2;
  travelDir: Cardinal;
  /** How close a side doorway may come to the entry wall (along-distance). */
  startClearance: number;
  /** How far from the corridor end the last doorway must stay. */
  endClearance: number;
  /** For a bend: the side the child corridor turns onto. Rooms on that side must
   * keep W/2 clear of the end so they don't overlap the crossing corridor. */
  turnSide?: 'left' | 'right' | null;
  /** For a hallway that entered via a bend: the side the parent corridor lies
   * on. Rooms on that side must keep W/2 clear of the start. */
  startCorridorSide?: 'left' | 'right';
}

/**
 * Greedily place hallway side doorways (side = 'left' | 'right') at random
 * positions, keeping every pair on a given side at least `max(minSpacing,
 * minAlong)` apart center-to-center so each side room's corridor cell is at
 * least its minimum along-dimension. Returns [] when the hallway rolls no side
 * doorways, or when a side's usable cell is too short to fit any room.
 */
export function placeSideDoorways(
  ids: Ids,
  rng: Rng,
  config: EngineConfig,
  opts: SidePlacementOpts,
): Doorway[] {
  if (rng() >= config.hallwaySideDoorwayChance) return [];

  const dirVec = CARDINAL_VEC[opts.travelDir];
  const leftVec = vecRotateNeg90(dirVec);
  const rightVec = vecRotate90(dirVec);
  const left: Cardinal = vecCardinal(leftVec);
  const right: Cardinal = vecCardinal(rightVec);

  const length = vecLen(vecSub(opts.end, opts.start));
  const dw = config.doorwayWidth;
  const halfW = config.hallwayWidth / 2;
  // A side-doorway room's "along" dimension runs parallel to the corridor:
  // width for E/W travel, depth for N/S travel. Its cell (the corridor span it
  // must fit in) must be at least this big, otherwise generateRoom would shrink
  // the room below its configured minimum (or degenerate the offset region and
  // poke out of its cell). Spacing + end/start clearances derive from it.
  const minAlong =
    opts.travelDir === 'east' || opts.travelDir === 'west'
      ? config.roomWidthMin
      : config.roomDepthMin;
  // Consecutive doorways must be at least `minAlong` apart (not just
  // `minSpacing`), so every interior midpoint-split cell is >= minAlong.
  const spacing = Math.max(minSpacing(config), minAlong);

  const result: Doorway[] = [];
  const sides: Array<{ normal: Vec2; cardinal: Cardinal; side: 'left' | 'right' }> = [
    { normal: leftVec, cardinal: left, side: 'left' },
    { normal: rightVec, cardinal: right, side: 'right' },
  ];
  for (const side of sides) {
    // A perpendicular corridor crosses one of this hallway's ends on one side
    // (a bend's turn side at the END; a bend child's parent side at the START),
    // reaching W/2 back into this hallway. A side room on that side must keep
    // W/2 clear of that end (mirrors generateNeighbor's backOffCorner logic).
    const startBackOff = opts.startCorridorSide === side.side ? halfW : 0;
    const endBackOff = opts.turnSide === side.side ? halfW : 0;
    // The cell a single room can occupy must be >= minAlong, else no room fits
    // and any doorway we cut here would produce an overlapping/undersized room.
    const cellLength = length - startBackOff - endBackOff;
    if (cellLength < minAlong) continue;
    // Doorway centers must (a) sit far enough from the entry wall for the player
    // to steer into the first one (startClearance), (b) keep the first/last
    // room's cell >= minAlong (back-off + minAlong/2), and (c) keep the full
    // opening inside the corridor. dw/2 terms are defensive if minAlong < dw.
    const tMin = Math.max(opts.startClearance, startBackOff + dw / 2, startBackOff + minAlong / 2);
    const tMax = Math.min(
      length - opts.endClearance,
      length - endBackOff - dw / 2,
      length - endBackOff - minAlong / 2,
    );
    if (tMax < tMin) continue;
    let t = tMin + rng() * Math.min(spacing, tMax - tMin);
    while (t <= tMax) {
      const center = vecAdd(opts.start, vecScale(dirVec, t));
      const pos = vecAdd(center, vecScale(side.normal, halfW));
      // A side doorway is approached from the corridor: steer at its opening
      // center (the default entryPoint is the floor point just past the opening),
      // which is laterally offset from the corridor centerline, so clicking it
      // visibly turns the player toward it. The default 'through' gate radius
      // commits the final turn as the player reaches the opening.
      result.push(
        makeDoorway(ids, {
          x: pos.x,
          z: pos.z,
          width: dw,
          height: config.doorwayHeight,
          facing: side.cardinal,
          side: side.side,
          targetEntityId: '',
          targetGenerated: false,
        }),
      );
      t += spacing + rng() * spacing;
    }
  }
  return result;
}

function vecCardinal(v: Vec2): Cardinal {
  if (v.x > 0) return 'east';
  if (v.x < 0) return 'west';
  if (v.z < 0) return 'north';
  return 'south';
}
