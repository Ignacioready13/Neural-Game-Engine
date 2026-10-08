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
}

/**
 * Greedily place hallway side doorways (side = 'left' | 'right') at random
 * positions, keeping every pair on a given side at least `minSpacing` apart
 * center-to-center. Returns [] when the hallway rolls no side doorways.
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
  const spacing = minSpacing(config);
  const halfW = config.hallwayWidth / 2;
  // Doorway centers must keep the full opening inside [startClearance, length - endClearance].
  const spanMin = Math.max(opts.startClearance, dw / 2);
  const spanMax = Math.max(spanMin, length - opts.endClearance - dw / 2);

  const result: Doorway[] = [];
  const sides: Array<{ normal: Vec2; cardinal: Cardinal }> = [
    { normal: leftVec, cardinal: left },
    { normal: rightVec, cardinal: right },
  ];
  for (const side of sides) {
    let t = spanMin + rng() * Math.min(dw, Math.max(0, spanMax - spanMin));
    while (t <= spanMax) {
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
          side: side.cardinal === left ? 'left' : 'right',
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
