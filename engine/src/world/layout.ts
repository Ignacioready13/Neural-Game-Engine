/**
 * Pure descriptor generation (no three.js) for hallways and rooms.
 *
 * Determinism: the ONLY randomness source is the passed Rng; call order is
 * fixed, so a given seed + traversal sequence reproduces identical descriptors.
 *
 * Geometry conventions (see geometry.ts for the meshes):
 *  - Hallways run axis-aligned with constant width = config.hallwayWidth and
 *    length in [hallwayLengthMin, hallwayLengthMax].
 *  - Hallway termination is always one of bend / junction / room; the starting
 *    hallway uses the same three (never the reserved 'start' value) and simply
 *    has no entry doorway.
 *  - A bend turns left or right (exactly 90deg) at the corridor end; the bend
 *    doorway sits on the turn-side wall so the parent and child corridors
 *    overlap at a W/2 x W/2 corner.
 *  - A junction forks through a W x W node flush at the corridor end. The
 *    config `junctionWeights` picks the fork layout (spec's "2-way / 3-way /
 *    4-way" names):
 *      * twoWay   = straight + one turn (left or right)  -> 2 exits (a fork)
 *      * threeWay = left + right                          -> 2 exits (a T, must turn)
 *      * fourWay  = straight + left + right               -> 3 exits (a cross)
 *    No branch ever reverses 180 degrees (the 4th arm of a cross is the entry).
 *  - Hallway side doorways always lead to rooms; hallway<->hallway only via
 *    bend/junction; room doorways always lead to hallways.
 */
import type { Cardinal, Doorway, EntityDescriptor, Vec2 } from '../core/types';
import { CARDINAL_VEC, OPPOSITE } from '../core/types';
import type { EngineConfig } from '../../config/config';
import type { Rng } from '../core/rng';
import { range, intRange, pick, pickWeighted } from '../core/rng';
import { vecAdd, vecScale, vecRotate90, vecRotateNeg90 } from '../core/math';
import { Ids, nextEntityId, makeDoorway, minSpacing, placeSideDoorways } from './doorway';

function cardinalOfVec(v: Vec2): Cardinal {
  if (v.x > 0) return 'east';
  if (v.x < 0) return 'west';
  if (v.z < 0) return 'north';
  return 'south';
}

export function leftOf(d: Cardinal): Cardinal {
  return cardinalOfVec(vecRotateNeg90(CARDINAL_VEC[d]));
}

export function rightOf(d: Cardinal): Cardinal {
  return cardinalOfVec(vecRotate90(CARDINAL_VEC[d]));
}

function axisOf(d: Cardinal): 'x' | 'z' {
  return d === 'east' || d === 'west' ? 'x' : 'z';
}

function pickTermination(rng: Rng, config: EngineConfig): 'bend' | 'junction' | 'room' {
  return pickWeighted(rng, config.hallwayTerminationWeights as Record<'bend' | 'junction' | 'room', number>);
}

/**
 * Junction branch directions, chosen by `config.junctionWeights` (Section 5.2b:
 * 2-way fork / 3-way T-junction / 4-way junction). Every branch is a multiple
 * of 90 degrees from the incoming corridor (straight = 0, left/right = +/-90);
 * nothing ever reverses 180 degrees — the 4th arm of a cross is the entry the
 * player came from.
 *
 * Layouts (see the file header): twoWay = straight + one turn; threeWay =
 * left + right (a T); fourWay = straight + left + right (a cross).
 */
function junctionBranches(rng: Rng, config: EngineConfig, travelDir: Cardinal): Cardinal[] {
  const arity = pickWeighted(
    rng,
    config.junctionWeights as Record<'twoWay' | 'threeWay' | 'fourWay', number>,
  );
  const straight = travelDir;
  const left = leftOf(travelDir);
  const right = rightOf(travelDir);
  if (arity === 'twoWay') {
    const turn = pick(rng, [left, right] as const);
    return [straight, turn];
  }
  if (arity === 'threeWay') {
    return [left, right];
  }
  return [straight, left, right];
}

/** World position of a junction branch's entry doorway (on the node edge). */
function junctionExitPosition(end: Vec2, travelDir: Cardinal, branch: Cardinal, W: number): Vec2 {
  const dirVec = CARDINAL_VEC[travelDir];
  const leftVec = vecRotateNeg90(dirVec);
  const rightVec = vecRotate90(dirVec);
  if (branch === travelDir) return vecAdd(end, vecScale(dirVec, W));
  if (branch === leftOf(travelDir)) {
    return vecAdd(end, vecAdd(vecScale(dirVec, W / 2), vecScale(leftVec, W / 2)));
  }
  return vecAdd(end, vecAdd(vecScale(dirVec, W / 2), vecScale(rightVec, W / 2)));
}

export interface HallwayOpts {
  isStart: boolean;
  entryTargetId?: string;
  /** Override the entry doorway position (used by bend children, whose entry
   *  sits at the corner rather than the centerline start). */
  entryPos?: Vec2;
  /** Extra clearance between the entry wall and the first side doorway. */
  startClearance?: number;
}

export function generateHallway(
  ids: Ids,
  rng: Rng,
  config: EngineConfig,
  start: Vec2,
  travelDir: Cardinal,
  length: number,
  termination: 'bend' | 'junction' | 'room',
  opts: HallwayOpts,
): EntityDescriptor {
  const id = nextEntityId(ids, 'hallway');
  const dirVec = CARDINAL_VEC[travelDir];
  const W = config.hallwayWidth;
  const end = vecAdd(start, vecScale(dirVec, length));
  const doorways: Doorway[] = [];

  if (!opts.isStart) {
    const entryPos = opts.entryPos ?? start;
    doorways.push(
      makeDoorway(ids, {
        x: entryPos.x,
        z: entryPos.z,
        width: config.doorwayWidth,
        height: config.doorwayHeight,
        facing: OPPOSITE[travelDir],
        side: 'start',
        targetEntityId: opts.entryTargetId ?? '',
        targetGenerated: opts.entryTargetId !== undefined,
      }),
    );
  }

  // Side doorways: keep them clear of the end (bend doorways live on the
  // turn-side wall near the end and share that wall, so keep full spacing).
  const endClearance =
    termination === 'bend'
      ? W / 4 + minSpacing(config) - config.doorwayWidth / 2
      : config.doorwayWidth / 2;
  doorways.push(
    ...placeSideDoorways(ids, rng, config, {
      start,
      end,
      travelDir,
      startClearance: opts.startClearance ?? config.sideDoorwayStartClearance,
      endClearance,
    }),
  );

  if (termination === 'room') {
    doorways.push(
      makeDoorway(ids, {
        x: end.x,
        z: end.z,
        width: config.doorwayWidth,
        height: config.doorwayHeight,
        facing: travelDir,
        side: 'end',
        targetEntityId: '',
        targetGenerated: false,
      }),
    );
  } else if (termination === 'bend') {
    const turnDir: Cardinal =
      pick(rng, ['left', 'right'] as const) === 'left' ? leftOf(travelDir) : rightOf(travelDir);
    const turnVec = CARDINAL_VEC[turnDir];
    // Center of the W/2 corner opening on the turn-side wall.
    const pos = vecAdd(end, vecAdd(vecScale(turnVec, W / 2), vecScale(dirVec, -W / 4)));
    // The bend doorway is on a SIDE wall (the player drives parallel to it, not
    // toward it), so navigation is two-phase (see Doorway.turn):
    //   approach — steer toward the corner (the parent's centerline end), so the
    //     player keeps driving straight down the corridor instead of drifting
    //     toward the turn-side wall and any side doorways/rooms there.
    //   through  — once abreast of the corner, steer toward a waypoint on the new
    //     corridor's centerline a few units past the corner, which turns them ~90
    //     degrees instead of hugging the inner corner and clipping the far wall.
    const waypoint = vecAdd(end, vecScale(turnVec, config.turnWaypointDepth));
    doorways.push(
      makeDoorway(ids, {
        x: pos.x,
        z: pos.z,
        width: config.doorwayWidth,
        height: config.doorwayHeight,
        facing: turnDir,
        side: 'end',
        targetEntityId: '',
        targetGenerated: false,
        entryPoint: waypoint,
        approachPoint: { x: end.x, z: end.z },
        turn: true,
      }),
    );
  } else {
    for (const branch of junctionBranches(rng, config, travelDir)) {
      const pos = junctionExitPosition(end, travelDir, branch, W);
      doorways.push(
        makeDoorway(ids, {
          x: pos.x,
          z: pos.z,
          width: config.doorwayWidth,
          height: config.doorwayHeight,
          facing: branch,
          side: 'end',
          targetEntityId: '',
          targetGenerated: false,
        }),
      );
    }
  }

  return {
    id,
    kind: 'hallway',
    axis: axisOf(travelDir),
    start,
    end,
    travelDirection: travelDir,
    corridorWidth: W,
    termination,
    doorways,
  };
}

function shuffled(rng: Rng, arr: number[]): number[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out;
}

interface RoomWall {
  side: Cardinal;
  spanMin: number;
  spanMax: number;
  fixedCoord: number;
}

/** Find a doorway center on a wall at least `spacing` from all occupied ones. */
function tryPlaceOnWall(
  spanMin: number,
  spanMax: number,
  occupied: number[],
  spacing: number,
  dw: number,
  rng: Rng,
): number | null {
  const lo = spanMin + dw / 2;
  const hi = spanMax - dw / 2;
  if (hi < lo) return null;
  for (let attempt = 0; attempt < 12; attempt++) {
    const c = lo + rng() * (hi - lo);
    if (occupied.every((o) => Math.abs(o - c) >= spacing)) return c;
  }
  for (let c = lo; c <= hi + 1e-9; c += 0.25) {
    if (occupied.every((o) => Math.abs(o - c) >= spacing)) return c;
  }
  return null;
}

/**
 * Additional room doorways beyond the entry doorway (>= 1, rule 6).
 *
 * The first additional doorway is ALWAYS placed on the wall OPPOSITE the entry
 * wall. The player always walks into a room perpendicular to the entry wall (the
 * doorway they came through), and can only steer toward doorways within a ~75
 * degree cone ahead of them — they can never turn around 180 degrees. If every
 * extra doorway lands on the entry wall (literally "back the way they came") or
 * on a perpendicular wall near the entry corner, the player walks straight into a
 * blank wall with no reachable exit and the game dead-ends. An opposite-wall
 * doorway is always directly ahead, so it is always reachable on entry.
 *
 * Separately, NO additional doorway is ever placed on the entry wall itself: that
 * wall is shared with the parent hallway, which only has the one opening the
 * player came through. A second opening there would be occluded by the parent's
 * solid wall (a phantom doorway the player walks through with no collision), and
 * it would force the player to enter and exit through the same wall.
 */
function placeRoomDoorways(
  ids: Ids,
  rng: Rng,
  config: EngineConfig,
  center: Vec2,
  width: number,
  depth: number,
  entrySide: Cardinal,
  entryCoord: number,
): Doorway[] {
  const dw = config.doorwayWidth;
  const spacing = minSpacing(config);
  const walls: RoomWall[] = [
    { side: 'east', spanMin: center.z - depth / 2, spanMax: center.z + depth / 2, fixedCoord: center.x + width / 2 },
    { side: 'west', spanMin: center.z - depth / 2, spanMax: center.z + depth / 2, fixedCoord: center.x - width / 2 },
    { side: 'north', spanMin: center.x - width / 2, spanMax: center.x + width / 2, fixedCoord: center.z - depth / 2 },
    { side: 'south', spanMin: center.x - width / 2, spanMax: center.x + width / 2, fixedCoord: center.z + depth / 2 },
  ];
  const occupied: Record<Cardinal, number[]> = { east: [], west: [], north: [], south: [] };
  occupied[entrySide].push(entryCoord);

  const result: Doorway[] = [];
  const pushDoorway = (w: RoomWall, pos: number): void => {
    occupied[w.side].push(pos);
    result.push(
      makeDoorway(ids, {
        x: w.side === 'north' || w.side === 'south' ? pos : w.fixedCoord,
        z: w.side === 'east' || w.side === 'west' ? pos : w.fixedCoord,
        width: dw,
        height: config.doorwayHeight,
        facing: w.side,
        side: w.side,
        targetEntityId: '',
        targetGenerated: false,
      }),
    );
  };

  // 1. Guaranteed exit ahead: one doorway on the wall opposite the entry. The
  //    opposite wall spans the room's width (N/S entry) or depth (E/W entry),
  //    both >= roomWidthMin/roomDepthMin (4) > doorwayWidth (2), so there is
  //    always room for at least one doorway here.
  const opposite = OPPOSITE[entrySide];
  const oppWall = walls.find((w) => w.side === opposite)!;
  const oppPos = tryPlaceOnWall(oppWall.spanMin, oppWall.spanMax, occupied[opposite], spacing, dw, rng);
  if (oppPos !== null) pushDoorway(oppWall, oppPos);

  // 2. Extra doorways (0..2) on random walls for variety. The entry wall is
  //    EXCLUDED: it is the wall shared with the parent hallway, which has exactly
  //    one opening (the entry doorway itself). Any additional doorway cut there
  //    is occluded by the parent's solid wall — a "phantom" opening the player
  //    phases through with no collision (the floor is contiguous on both sides).
  const otherWallIndices = [0, 1, 2, 3].filter((i) => walls[i]!.side !== entrySide);
  const targetExtra = intRange(rng, 0, 2);
  for (let k = 0; k < targetExtra; k++) {
    let placed = false;
    for (const wi of shuffled(rng, otherWallIndices)) {
      const w = walls[wi]!;
      const pos = tryPlaceOnWall(w.spanMin, w.spanMax, occupied[w.side], spacing, dw, rng);
      if (pos === null) continue;
      pushDoorway(w, pos);
      placed = true;
      break;
    }
    if (!placed) break;
  }

  // Safety net for rule 6 (belt-and-suspenders; the opposite-wall placement
  // already guarantees >= 1 additional doorway in practice). Same entry-wall
  // exclusion as above.
  if (result.length === 0) {
    for (const wi of shuffled(rng, otherWallIndices)) {
      const w = walls[wi]!;
      const pos = tryPlaceOnWall(w.spanMin, w.spanMax, occupied[w.side], spacing, dw, rng);
      if (pos === null) continue;
      pushDoorway(w, pos);
      break;
    }
  }
  return result;
}

/** Corridor span (along the parent hallway's travel axis) that a side-doorway
 * room's footprint must stay inside, so it never overlaps the parent's junction
 * node or its start-side connection. */
export interface RoomAlongRange {
  min: number;
  max: number;
}

/**
 * A perpendicular corridor that crosses one END of this hallway (a bend) spans
 * +/-W/2 along the hallway's axis and reaches W/2 back into the hallway's own
 * stretch. A side room on that corridor's side must back off W/2 from that end
 * so it does not overlap the crossing corridor.
 *
 * `cornerCoord` is the hallway-end coordinate (start or end along `axis`);
 * `interiorSign` points from that end INTO the hallway (+1 or -1); the bound on
 * the corner side is tightened toward the interior by W/2.
 */
function backOffCorner(range: RoomAlongRange, cornerCoord: number, interiorSign: number, W: number): void {
  const nearEdge = cornerCoord + (W / 2) * interiorSign;
  if (interiorSign > 0) range.min = Math.max(range.min, nearEdge);
  else range.max = Math.min(range.max, nearEdge);
}

export function generateRoom(
  ids: Ids,
  rng: Rng,
  config: EngineConfig,
  entry: { x: number; z: number; facing: Cardinal },
  parentId: string,
  alongRange?: RoomAlongRange,
): EntityDescriptor {
  const id = nextEntityId(ids, 'room');
  const ceilingHeight = range(rng, config.roomHeightMin, config.roomHeightMax);
  const entrySide: Cardinal = OPPOSITE[entry.facing];
  const dw = config.doorwayWidth;

  let width = range(rng, config.roomWidthMin, config.roomWidthMax);
  let depth = range(rng, config.roomDepthMin, config.roomDepthMax);

  // For east/west entry the room's depth runs parallel to the corridor; for
  // north/south entry its width does. When a corridor span is supplied (side
  // doorways), keep that "along" dimension inside the span so the room never
  // bleeds into the junction node or the parent's start-side connection.
  const alongIsDepth = entrySide === 'east' || entrySide === 'west';
  const entryCoord = alongIsDepth ? entry.z : entry.x;
  let halfW = width / 2;
  let halfD = depth / 2;
  let alongHalf = alongIsDepth ? halfD : halfW;
  let offset = 0;
  if (alongRange) {
    const span = alongRange.max - alongRange.min;
    if (2 * alongHalf > span) {
      // Shrink the along dimension to fit the corridor. `span` is a corridor
      // length (>= hallwayLengthMin), so the result stays within the room's
      // configured dimension ranges.
      alongHalf = span / 2;
      if (alongIsDepth) {
        depth = span;
        halfD = span / 2;
      } else {
        width = span;
        halfW = span / 2;
      }
    }
    const baseHalf = alongHalf - dw / 2;
    // Feasible offset keeps both the full opening on the entry wall and the
    // whole footprint inside the corridor span (non-empty after the shrink).
    const lo = Math.max(-baseHalf, alongRange.min - entryCoord + alongHalf);
    const hi = Math.min(baseHalf, alongRange.max - entryCoord - alongHalf);
    offset = lo + rng() * (hi - lo);
  } else {
    const baseHalf = alongHalf - dw / 2;
    offset = baseHalf > 0 ? range(rng, -baseHalf, baseHalf) : 0;
  }

  // Position the room so its entry wall passes through the parent doorway.
  let center: Vec2;
  if (entrySide === 'east') center = { x: entry.x - halfW, z: entry.z + offset };
  else if (entrySide === 'west') center = { x: entry.x + halfW, z: entry.z + offset };
  else if (entrySide === 'north') center = { x: entry.x + offset, z: entry.z + halfD };
  else center = { x: entry.x + offset, z: entry.z - halfD };

  const doorways: Doorway[] = [
    makeDoorway(ids, {
      x: entry.x,
      z: entry.z,
      width: dw,
      height: config.doorwayHeight,
      facing: entrySide,
      side: entrySide,
      targetEntityId: parentId,
      targetGenerated: true,
    }),
    ...placeRoomDoorways(ids, rng, config, center, width, depth, entrySide, entryCoord),
  ];

  return { id, kind: 'room', center, width, depth, ceilingHeight, doorways };
}

export interface GeneratedChild {
  descriptor: EntityDescriptor;
  /** True when the child's start wall is coplanar with a parent wall and must
   *  not be rebuilt (hallway children from rooms and junction branches). Rooms
   *  always build their own entry wall — see buildRoom. Bend children build
   *  their own start wall. */
  skipEntryWall: boolean;
  /** Parent's travel direction — bend children need it to place the corner. */
  parentDir?: Cardinal;
}

/** Create the neighbor an ungenerated doorway points at (lazy generation). */
export function generateNeighbor(
  ids: Ids,
  rng: Rng,
  config: EngineConfig,
  parent: EntityDescriptor,
  doorway: Doorway,
): GeneratedChild {
  if (parent.kind === 'room') {
    const length = range(rng, config.hallwayLengthMin, config.hallwayLengthMax);
    const termination = pickTermination(rng, config);
    const descriptor = generateHallway(
      ids,
      rng,
      config,
      { x: doorway.x, z: doorway.z },
      doorway.facing,
      length,
      termination,
      { isStart: false, entryTargetId: parent.id },
    );
    return { descriptor, skipEntryWall: true };
  }

  // Parent is a hallway.
  if (doorway.side === 'left' || doorway.side === 'right') {
    // Keep the room inside its own cell along the corridor: bounded by the
    // parent corridor span AND the midpoints to the nearest same-wall doorway
    // neighbours. The corridor span alone is not enough — two rooms on the same
    // wall could otherwise overlap (their widths can exceed the doorway
    // spacing), which makes `entityAt` ambiguous and lets a player cross into
    // the wrong room through a doorway's opening.
    const axis = parent.axis === 'z' ? 'z' : 'x';
    const corridorMin = Math.min(parent.start![axis], parent.end![axis]);
    const corridorMax = Math.max(parent.start![axis], parent.end![axis]);
    const siblings = parent.doorways
      .filter((d) => d.side === doorway.side)
      .map((d) => d[axis])
      .sort((a, b) => a - b);
    const p = doorway[axis];
    const idx = siblings.indexOf(p);
    // Outer bounds stay the corridor span (as before); only the gap to an
    // adjacent same-wall doorway is halved, so siblings never overlap while
    // the first/last room can still reach its configured min dimension.
    const alongRange: RoomAlongRange = {
      min: idx > 0 ? Math.max(corridorMin, (siblings[idx - 1]! + p) / 2) : corridorMin,
      max: idx < siblings.length - 1 ? Math.min(corridorMax, (p + siblings[idx + 1]!) / 2) : corridorMax,
    };

    // A bend turns 90deg at a corridor end, so a perpendicular corridor crosses
    // that end CENTERED on the end point and reaches W/2 back into this
    // hallway's own stretch. A side room on the crossing corridor's side that
    // extends all the way to the end would overlap it — its wall renders
    // "inside" the room and then vanishes once the player enters and the other
    // corridor is disposed. Back the room off W/2 from that end. There are two
    // such crossings: this hallway's own bend (the turn side, at the END), and —
    // when this hallway itself entered through a bend — the parent corridor (on
    // `startCorridorSide`, at the START).
    const startAlong = parent.start![axis];
    const endAlong = parent.end![axis];
    const W = parent.corridorWidth ?? config.hallwayWidth;
    if (parent.termination === 'bend') {
      const turnDir = parent.doorways.find((d) => d.side === 'end')?.facing;
      const travel = parent.travelDirection as Cardinal;
      const turnSide: 'left' | 'right' | null =
        turnDir === leftOf(travel) ? 'left' : turnDir === rightOf(travel) ? 'right' : null;
      if (turnSide === doorway.side) {
        backOffCorner(alongRange, endAlong, Math.sign(startAlong - endAlong), W);
      }
    }
    if (parent.startCorridorSide === doorway.side) {
      backOffCorner(alongRange, startAlong, Math.sign(endAlong - startAlong), W);
    }

    const descriptor = generateRoom(
      ids,
      rng,
      config,
      { x: doorway.x, z: doorway.z, facing: doorway.facing },
      parent.id,
      alongRange,
    );
    return { descriptor, skipEntryWall: true };
  }

  // End doorway — depends on termination.
  const term = parent.termination;
  if (term === 'room') {
    const descriptor = generateRoom(
      ids,
      rng,
      config,
      { x: doorway.x, z: doorway.z, facing: doorway.facing },
      parent.id,
    );
    return { descriptor, skipEntryWall: true };
  }
  if (term === 'bend') {
    const parentDirVec = CARDINAL_VEC[parent.travelDirection as Cardinal];
    const entryPos = vecAdd(
      { x: parent.end!.x, z: parent.end!.z },
      vecScale(parentDirVec, -config.hallwayWidth / 4),
    );
    const length = range(rng, config.hallwayLengthMin, config.hallwayLengthMax);
    const newTerm = pickTermination(rng, config);
    const descriptor = generateHallway(
      ids,
      rng,
      config,
      { x: parent.end!.x, z: parent.end!.z },
      doorway.facing,
      length,
      newTerm,
      {
        isStart: false,
        entryTargetId: parent.id,
        entryPos,
        // Match the bend end-clearance: the first side doorway must sit far
        // enough from the start that a room on the parent side can still reach
        // its minimum dimension between the parent corridor (which reaches W/2
        // back into this hallway) and the next same-wall doorway. W/2 + dw/2
        // left the gap too small (span could fall below roomDepthMin), forcing
        // a shrink below the configured minimum.
        startClearance: config.hallwayWidth / 4 + minSpacing(config) - config.doorwayWidth / 2,
      },
    );
    // Record which side of this bend child the parent corridor lies on, so its
    // side rooms on that side back off from the start (see generateNeighbor's
    // side-doorway branch). A left turn leaves the parent on the child's left;
    // a right turn leaves it on the child's right.
    descriptor.startCorridorSide =
      doorway.facing === rightOf(parent.travelDirection as Cardinal) ? 'right' : 'left';
    return { descriptor, skipEntryWall: false, parentDir: parent.travelDirection };
  }
  // junction
  const length = range(rng, config.hallwayLengthMin, config.hallwayLengthMax);
  const newTerm = pickTermination(rng, config);
  const descriptor = generateHallway(
    ids,
    rng,
    config,
    { x: doorway.x, z: doorway.z },
    doorway.facing,
    length,
    newTerm,
    { isStart: false, entryTargetId: parent.id },
  );
  return { descriptor, skipEntryWall: true };
}

/** The starting hallway: at the origin, facing a random cardinal, no entry. */
export function generateStartHallway(ids: Ids, rng: Rng, config: EngineConfig): GeneratedChild {
  const dir = pick(rng, ['east', 'west', 'north', 'south'] as const);
  const length = range(rng, config.hallwayLengthMin, config.hallwayLengthMax);
  const termination = pickTermination(rng, config);
  const descriptor = generateHallway(ids, rng, config, { x: 0, z: 0 }, dir, length, termination, {
    isStart: true,
  });
  return { descriptor, skipEntryWall: false };
}

/** True when the floor point lies inside the entity's interior (or its node). */
export function containsPoint(
  desc: EntityDescriptor,
  x: number,
  z: number,
  config: EngineConfig,
): boolean {
  if (desc.kind === 'room') {
    const c = desc.center!;
    return (
      x >= c.x - desc.width! / 2 &&
      x <= c.x + desc.width! / 2 &&
      z >= c.z - desc.depth! / 2 &&
      z <= c.z + desc.depth! / 2
    );
  }
  const axis = desc.axis!;
  const s = desc.start!;
  const e = desc.end!;
  const W = desc.corridorWidth ?? config.hallwayWidth;
  const along = (v: Vec2) => (axis === 'x' ? v.x : v.z);
  const perp = (v: Vec2) => (axis === 'x' ? v.z : v.x);
  let a0 = Math.min(along(s), along(e));
  let a1 = Math.max(along(s), along(e));
  if (desc.termination === 'junction') {
    // The W x W junction node sits PAST the corridor end, in the travel
    // direction. Extend whichever bound is the corridor's END (not always +a1:
    // a north/west-traveling corridor's end is its lower bound). Extending the
    // wrong bound left the node unreachable (an invisible wall at the end) and
    // even attributed it to a spurious region behind the start.
    const dirVec = CARDINAL_VEC[desc.travelDirection as Cardinal];
    const alongDir = axis === 'x' ? dirVec.x : dirVec.z;
    if (alongDir > 0) a1 += W;
    else a0 -= W;
  }
  const p0 = perp(s) - W / 2;
  const p1 = perp(s) + W / 2;
  const a = along({ x, z });
  const p = perp({ x, z });
  return a >= a0 && a <= a1 && p >= p0 && p <= p1;
}
