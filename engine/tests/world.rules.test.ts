/**
 * AUTOMATED RULE-VERIFICATION SUITE (Section 12.4)
 * ================================================
 * Deterministic, non-LLM checks that generate many sample worlds and verify the
 * exact geometric rules from Section 5 programmatically. Runs in plain Node
 * (vitest) — no WebGL, no rendering; it inspects EntityDescriptor data only.
 */
import { describe, it, expect } from 'vitest';
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import { CARDINAL_VEC, OPPOSITE } from '../src/core/types';
import type { Cardinal, Doorway, Entity, EntityDescriptor, PlayerState } from '../src/core/types';
import { createRng } from '../src/core/rng';
import { Steering } from '../src/control/steering';
import { doorwayPhase, doorwayTargetPoint } from '../src/sim/navigation';
import { resolveWallCollision } from '../src/sim/collision';
import { angleDiff } from '../src/core/math';

function dist2(ax: number, az: number, bx: number, bz: number): number {
  return Math.hypot(ax - bx, az - bz);
}

/** True iff two cardinal directions are perpendicular (90 degrees apart). */
function perpendicular(a: Cardinal, b: Cardinal): boolean {
  const va = CARDINAL_VEC[a];
  const vb = CARDINAL_VEC[b];
  return va.x * vb.x + va.z * vb.z === 0;
}

/** Rule 4: doorway spacing on the same wall/side. */
function verifySpacing(d: EntityDescriptor): void {
  const min = CONFIG.doorwaySpacingFactor * CONFIG.doorwayWidth;
  const groups = new Map<string, Doorway[]>();
  for (const dw of d.doorways) {
    // 'start' (entry) and 'end' (bend/junction/room termination) are not
    // "side wall" doorways; the spacing rule (5.4) applies to doorways a player
    // would see side-by-side on one wall — hallway left/right sides and room
    // cardinal walls.
    if (dw.side === 'start' || dw.side === 'end') continue;
    const arr = groups.get(dw.side) ?? [];
    arr.push(dw);
    groups.set(dw.side, arr);
  }
  for (const [side, dws] of groups) {
    for (let i = 0; i < dws.length; i++) {
      for (let j = i + 1; j < dws.length; j++) {
        const a = dws[i]!;
        const b = dws[j]!;
        const d = dist2(a.x, a.z, b.x, b.z);
        expect(d, `doorway spacing on side "${side}"`).toBeGreaterThanOrEqual(min - 1e-6);
      }
    }
  }
}

/** Rules 1-3, 5, 6 for a single entity, plus hallway length/axis checks. */
function verifyEntity(d: EntityDescriptor, world: World): void {
  if (d.kind === 'hallway') {
    // Rule 2: constant corridor width.
    expect(d.corridorWidth).toBeCloseTo(CONFIG.hallwayWidth, 6);

    // Axis-aligned + length range.
    expect(d.axis === 'x' || d.axis === 'z').toBe(true);
    expect(d.start).toBeDefined();
    expect(d.end).toBeDefined();
    expect(d.travelDirection).toBeDefined();
    if (d.axis === 'x') {
      expect(d.start!.z).toBeCloseTo(d.end!.z, 6);
    } else {
      expect(d.start!.x).toBeCloseTo(d.end!.x, 6);
    }
    const len = Math.hypot(d.end!.x - d.start!.x, d.end!.z - d.start!.z);
    expect(len).toBeGreaterThanOrEqual(CONFIG.hallwayLengthMin - 1e-6);
    expect(len).toBeLessThanOrEqual(CONFIG.hallwayLengthMax + 1e-6);

    // Hallway travel direction is a cardinal axis matching `axis`.
    const tv = CARDINAL_VEC[d.travelDirection!];
    expect(d.axis === 'x' ? tv.z === 0 : tv.x === 0).toBe(true);

    // Rules 1 & 6: hallway<->hallway connections must be axis-aligned (i.e. a
    // change of a multiple of 90 degrees) and must happen only via 'start'/'end'
    // (bend/junction), never via a side doorway. A `bend` termination turns
    // exactly 90 degrees; a junction branch may also continue straight (0 deg).
    for (const dw of d.doorways) {
      const target = dw.targetGenerated ? world.get(dw.targetEntityId) : undefined;
      if (dw.side === 'left' || dw.side === 'right') {
        // Rule 6: every side doorway leads to a ROOM, always.
        if (target) {
          expect(target.descriptor.kind, 'side doorway must lead to a room').toBe('room');
        }
      }
      if (target && target.descriptor.kind === 'hallway') {
        expect(['start', 'end'], 'hallway<->hallway only via bend/junction').toContain(dw.side);
        // Rule 1: direction change is a multiple of 90 degrees. Both travel
        // directions are cardinals, so the dot product must be an integer in
        // {-1, 0, 1} (parallel / perpendicular / antiparallel).
        const tv = CARDINAL_VEC[d.travelDirection!];
        const ctv = CARDINAL_VEC[target.descriptor.travelDirection!];
        const dot = tv.x * ctv.x + tv.z * ctv.z;
        expect(
          Number.isInteger(dot) && Math.abs(dot) <= 1,
          'hallway direction change must be a multiple of 90 degrees',
        ).toBe(true);
        if (d.termination === 'bend' && dw.side === 'end') {
          expect(
            perpendicular(d.travelDirection!, target.descriptor.travelDirection!),
            'a bend must turn exactly 90 degrees',
          ).toBe(true);
        }
      }
    }
  } else {
    // Rule 5: every room has at least 2 doorways (entry + one more).
    expect(d.doorways.length, 'room must have >= 2 doorways').toBeGreaterThanOrEqual(2);

    // Rule 3: dimensions within configured ranges.
    expect(d.width).toBeGreaterThanOrEqual(CONFIG.roomWidthMin - 1e-6);
    expect(d.width).toBeLessThanOrEqual(CONFIG.roomWidthMax + 1e-6);
    expect(d.depth).toBeGreaterThanOrEqual(CONFIG.roomDepthMin - 1e-6);
    expect(d.depth).toBeLessThanOrEqual(CONFIG.roomDepthMax + 1e-6);
    expect(d.ceilingHeight).toBeGreaterThanOrEqual(CONFIG.roomHeightMin - 1e-6);
    expect(d.ceilingHeight).toBeLessThanOrEqual(CONFIG.roomHeightMax + 1e-6);
  }

  // Rule 4: spacing (applies to both kinds).
  verifySpacing(d);
}

/** Walk the lazily-generated graph for a while, verifying as entities appear. */
function walkAndVerify(seed: number, steps: number, rng: () => number): number {
  const world = new World(seed, CONFIG);
  let active: Entity = world.init();
  let verified = 0;

  verifyEntity(active.descriptor, world);
  verified++;

  for (let s = 0; s < steps; s++) {
    world.maintain(active.descriptor.id);

    // Verify the active entity (its neighbors are now all generated/live).
    verifyEntity(active.descriptor, world);

    // Verify each entity generated by this maintain step.
    for (const e of world.all()) {
      verifyEntity(e.descriptor, world);
      verified++;
    }

    // Enter a random generated neighbor.
    const exits = active.descriptor.doorways.filter(
      (d) => d.targetGenerated && d.targetEntityId !== '',
    );
    if (exits.length === 0) break;
    const exit = exits[Math.floor(rng() * exits.length)]!;
    active = world.get(exit.targetEntityId)!;
    if (!active) break;
  }
  return verified;
}

describe('world rule verification (Section 12.4)', () => {
  it('enforces all geometric rules across many generated worlds', () => {
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    let total = 0;
    for (const seed of seeds) {
      total += walkAndVerify(seed, 60, rng);
    }
    // Sanity: we actually generated a meaningful number of entities.
    expect(total).toBeGreaterThan(500);
  });

  it('is deterministic for a fixed seed', () => {
    const a = new World(12345, CONFIG);
    const b = new World(12345, CONFIG);
    const da = JSON.stringify(a.init().descriptor);
    const db = JSON.stringify(b.init().descriptor);
    expect(da).toBe(db);
  });

  it('never places two rooms with overlapping floor footprints', () => {
    // Guards a real layout bug: two rooms on the same wall could overlap when
    // their widths exceeded the doorway spacing, making `entityAt` ambiguous and
    // letting a player cross into the WRONG room through a doorway's opening.
    // Rooms may touch (share a boundary edge) but never have positive-area
    // overlap. Rooms on opposite walls are separated by the corridor width.
    const EPS = 1e-6;
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        const rooms = world
          .all()
          .filter((e) => e.descriptor.kind === 'room')
          .map((e) => e.descriptor);
        for (let i = 0; i < rooms.length; i++) {
          for (let j = i + 1; j < rooms.length; j++) {
            const a = rooms[i]!;
            const b = rooms[j]!;
            const overlaps =
              a.center!.x + a.width! / 2 - (b.center!.x - b.width! / 2) > EPS &&
              b.center!.x + b.width! / 2 - (a.center!.x - a.width! / 2) > EPS &&
              a.center!.z + a.depth! / 2 - (b.center!.z - b.depth! / 2) > EPS &&
              b.center!.z + b.depth! / 2 - (a.center!.z - a.depth! / 2) > EPS;
            expect(
              overlaps,
              `seed ${seed} step ${s}: rooms ${a.id} and ${b.id} overlap`,
            ).toBe(false);
          }
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
  });

  it('every room has a reachable exit doorway from its entry', () => {
    // Guards the dead-end bug: a room whose only extra doorway sits on the entry
    // wall (back the way the player came) or on a perpendicular wall near the
    // entry corner. The player walks in perpendicular to the entry wall and can
    // only turn within the reachability cone (never 180 degrees), so such a room
    // leaves them walking into a blank wall with no reachable target — the game
    // becomes unplayable. Every room must have at least one doorway reachable
    // from a player standing at the entry, facing into the room.
    const steering = new Steering(CONFIG);
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    let roomsChecked = 0;
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        for (const e of world.all()) {
          const d = e.descriptor;
          if (d.kind !== 'room') continue;
          roomsChecked++;
          // The entry doorway is always doorways[0]; its side is the wall the
          // player came through, and they enter facing the opposite direction.
          const entry = d.doorways[0]!;
          const entrySide = entry.side as Cardinal;
          const facingVec = CARDINAL_VEC[OPPOSITE[entrySide]];
          const player = {
            x: entry.x,
            z: entry.z,
            heading: Math.atan2(facingVec.z, facingVec.x),
            angularVelocity: 0,
          };
          const hasExit = d.doorways
            .slice(1)
            .some((dw) => steering.isReachable(player, dw.entryPoint));
          expect(
            hasExit,
            `seed ${seed} step ${s}: room ${d.id} has no reachable exit from its entry (entry wall ${entrySide})`,
          ).toBe(true);
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
    expect(roomsChecked).toBeGreaterThan(500);
  });

  it('never places an extra doorway on a room\'s entry wall', () => {
    // Guards the phantom-doorway bug: the entry wall is shared with the parent
    // hallway, which only has the one opening the player came through. A second
    // doorway cut there is occluded by the parent's solid wall — the player sees
    // no opening, yet the floor is contiguous on both sides, so they phase through
    // the "phantom" doorway with no collision. The generator must never put an
    // additional (non-entry) doorway on the wall it entered the room through.
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    let roomsChecked = 0;
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        for (const e of world.all()) {
          const d = e.descriptor;
          if (d.kind !== 'room') continue;
          roomsChecked++;
          const entry = d.doorways[0]!;
          const entrySide = entry.side as Cardinal;
          for (const dw of d.doorways.slice(1)) {
            expect(
              dw.side,
              `seed ${seed} step ${s}: room ${d.id} has an extra doorway on its entry wall (${entrySide})`,
            ).not.toBe(entrySide);
          }
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
    expect(roomsChecked).toBeGreaterThan(500);
  });

  it('never places a room that overlaps a hallway corridor', () => {
    // Guards the bend bug: a side room on the turn side of a bend could extend
    // to the hallway's end and poke into the continuation corridor (centered on
    // the end point, reaching W/2 back into the parent). Its wall then rendered
    // "inside" the room and vanished once the player entered and the corridor
    // was disposed. A room may touch (share an edge with) a hallway wall but
    // never have positive-area overlap with any hallway's corridor band.
    const EPS = 1e-6;
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        const all = world.all();
        const rooms = all.filter((e) => e.descriptor.kind === 'room').map((e) => e.descriptor);
        const halls = all.filter((e) => e.descriptor.kind === 'hallway').map((e) => e.descriptor);
        for (const r of rooms) {
          const rx1 = r.center!.x - r.width! / 2;
          const rx2 = r.center!.x + r.width! / 2;
          const rz1 = r.center!.z - r.depth! / 2;
          const rz2 = r.center!.z + r.depth! / 2;
          for (const h of halls) {
            const W = h.corridorWidth!;
            const hx1 = h.axis === 'x' ? Math.min(h.start!.x, h.end!.x) : h.start!.x - W / 2;
            const hx2 = h.axis === 'x' ? Math.max(h.start!.x, h.end!.x) : h.start!.x + W / 2;
            const hz1 = h.axis === 'z' ? Math.min(h.start!.z, h.end!.z) : h.start!.z - W / 2;
            const hz2 = h.axis === 'z' ? Math.max(h.start!.z, h.end!.z) : h.start!.z + W / 2;
            const overlaps =
              rx2 - hx1 > EPS && hx2 - rx1 > EPS && rz2 - hz1 > EPS && hz2 - rz1 > EPS;
            expect(
              overlaps,
              `seed ${seed} step ${s}: room ${r.id} overlaps hallway ${h.id}`,
            ).toBe(false);
          }
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
  });

  it('lazily keeps only the active entity and its direct neighbors', () => {
    const world = new World(42, CONFIG);
    const start = world.init();
    world.maintain(start.descriptor.id);

    // After maintain, every live entity must be the active entity or a direct
    // neighbor of it.
    const active = start.descriptor;
    const neighborIds = new Set(active.doorways.map((d) => d.targetEntityId));
    for (const e of world.all()) {
      const isActiveOrNeighbor =
        e.descriptor.id === active.id || neighborIds.has(e.descriptor.id);
      expect(isActiveOrNeighbor).toBe(true);
    }
  });

  it('steers through every bend into the new corridor without hitting the far wall', () => {
    // Guards the bend bug: a hallway that turns 90 degrees puts its bend doorway
    // on a SIDE wall, so a player driving toward it parallel to the wall must
    // still make the ~90 degree turn. With the old single waypoint (or a too-tight
    // gate radius) the player kept driving straight, crossed the corner while
    // still heading down the parent corridor, and plowed into the new corridor's
    // far wall. This simulates the REAL steering + collision pipeline (Steering.step,
    // doorwayPhase, doorwayTargetPoint, resolveWallCollision) approaching every
    // generated bend and asserts the player actually turns into the child corridor.
    const steering = new Steering(CONFIG);
    const rng = createRng(0x5eed);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    let bendsChecked = 0;
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        for (const e of world.all()) {
          const d = e.descriptor;
          if (d.kind !== 'hallway' || d.termination !== 'bend') continue;
          const bend = d.doorways.find((x) => x.side === 'end');
          if (!bend || !bend.targetGenerated) continue;
          const child = world.get(bend.targetEntityId);
          if (!child || child.descriptor.kind !== 'hallway') continue;
          bendsChecked++;
          expect(
            steerThroughBend(world, steering, d, bend, child.descriptor),
            `seed ${seed} step ${s}: bend ${bend.id} (${d.id}->${child.descriptor.id}) not navigable`,
          ).toBe(true);
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
    expect(bendsChecked).toBeGreaterThan(100);
  });

  it('steering toward a side doorway visibly turns the player into the room', () => {
    // Guards the "clicks don't steer" regression: a side doorway is a lateral
    // opening the player must visibly turn toward. A previous fix set its approach
    // target to the corridor centerline (straight ahead), so clicking a side
    // doorway produced no visible turn and the player drove straight past it. This
    // simulates the REAL pipeline (Steering.step, doorwayPhase, doorwayTargetPoint,
    // resolveWallCollision) approaching every generated side doorway and asserts
    // the player's heading actually deviates from the corridor direction as they
    // steer toward the opening.
    const steering = new Steering(CONFIG);
    const rng = createRng(0x71a11);
    const seeds = Array.from({ length: 60 }, (_, i) => i * 7919 + 13);
    let sideChecked = 0;
    for (const seed of seeds) {
      const world = new World(seed, CONFIG);
      let active: Entity = world.init();
      for (let s = 0; s < 40; s++) {
        world.maintain(active.descriptor.id);
        for (const e of world.all()) {
          const d = e.descriptor;
          if (d.kind !== 'hallway') continue;
          for (const side of d.doorways) {
            if ((side.side !== 'left' && side.side !== 'right') || !side.targetGenerated) continue;
            const target = world.get(side.targetEntityId);
            if (!target) continue;
            sideChecked++;
            expect(
              maxHeadingDeviationInto(world, steering, d, side) > 0.15,
              `seed ${seed} step ${s}: side doorway ${side.id} (${d.id}->${target.descriptor.id}) did not turn the player`,
            ).toBe(true);
          }
        }
        const exits = active.descriptor.doorways.filter(
          (d) => d.targetGenerated && d.targetEntityId !== '',
        );
        if (exits.length === 0) break;
        active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
        if (!active) break;
      }
    }
    expect(sideChecked).toBeGreaterThan(50);
  });
});

/**
 * Simulate a player approaching a hallway bend and turning into its child.
 * Returns true iff the player ends up inside the child corridor heading within
 * 45 degrees of the child's travel direction (i.e. they turned the corner) and
 * never gets pinned against a wall. Mirrors the Engine's per-frame pipeline.
 */
function steerThroughBend(
  world: World,
  steering: Steering,
  parent: EntityDescriptor,
  bend: Doorway,
  child: EntityDescriptor,
): boolean {
  const dirVec = CARDINAL_VEC[parent.travelDirection!];
  const childDirVec = CARDINAL_VEC[child.travelDirection!];
  const childHeading = Math.atan2(childDirVec.z, childDirVec.x);

  // Start on the parent centerline, hallwayLengthMin units before the corner,
  // heading straight down the corridor (as a player who has clicked the bend).
  let player: PlayerState = {
    x: parent.end!.x - dirVec.x * CONFIG.hallwayLengthMin,
    z: parent.end!.z - dirVec.z * CONFIG.hallwayLengthMin,
    heading: Math.atan2(dirVec.z, dirVec.x),
    angularVelocity: 0,
  };
  let activeId = parent.id;
  let targetId: string | null = bend.id;
  let phase: 'approach' | 'through' = 'approach';
  let holdFrames = 0;

  for (let f = 0; f < 150; f++) {
    const active = world.get(activeId)!.descriptor;
    const liveIds = active.doorways.map((dw) => dw.id);
    if (targetId !== null && !liveIds.includes(targetId)) {
      // The bend doorway is no longer live: we crossed into the child. Re-resolve
      // to the first reachable forward exit, mirroring the held-click fallback.
      const cands = active.doorways.filter((dw) => dw.side !== 'start');
      const reach = cands.find((dw) => steering.isReachable(player, dw.entryPoint));
      targetId = reach ? reach.id : null;
      phase = 'approach';
    }
    const tw = active.doorways.find((dw) => dw.id === targetId);
    let tp = null;
    if (tw) {
      phase = doorwayPhase(player, tw, CONFIG, phase);
      tp = doorwayTargetPoint(tw, phase);
    }
    const next = resolveWallCollision(world, player, steering.step(player, tp, CONFIG.simTimestep));

    // Collision held the position (pinned in a corner / against a wall).
    if (Math.hypot(next.x - player.x, next.z - player.z) < 1e-6) holdFrames++;
    else holdFrames = 0;
    if (holdFrames > 30) return false;

    player = next;
    if (!world.containsEntity(activeId, player.x, player.z)) {
      const here = world.entityAt(player.x, player.z);
      if (here && here.id !== activeId) {
        activeId = here.id;
        world.maintain(activeId);
      }
    }

    if (activeId === child.id) {
      const err = Math.abs(angleDiff(player.heading, childHeading));
      if (err < Math.PI / 4) return true;
    }
  }
  return Math.abs(angleDiff(player.heading, childHeading)) < Math.PI / 4;
}

/**
 * Simulate a player clicking a hallway side doorway and measure how far their
 * heading deviates from the corridor direction while steering toward it. A side
 * doorway's opening center is laterally offset from the corridor centerline, so a
 * working click turns the player (deviation grows to tens of degrees); a broken
 * "drive straight" approach keeps it near zero. Mirrors the Engine's per-frame
 * pipeline. Returns the max |heading - corridor heading| observed.
 */
function maxHeadingDeviationInto(
  world: World,
  steering: Steering,
  parent: EntityDescriptor,
  side: Doorway,
): number {
  const dirVec = CARDINAL_VEC[parent.travelDirection!];
  const corridorHeading = Math.atan2(dirVec.z, dirVec.x);

  const a = parent.axis!;
  const along = (v: { x: number; z: number }) => (a === 'x' ? v.x : v.z);
  const perp = (v: { x: number; z: number }) => (a === 'x' ? v.z : v.x);
  const sign = Math.sign(along(parent.end!) - along(parent.start!)) || 1;
  let sa = along(side) - sign * CONFIG.hallwayLengthMin;
  const a0 = Math.min(along(parent.start!), along(parent.end!));
  const a1 = Math.max(along(parent.start!), along(parent.end!));
  sa = Math.max(a0 + 0.5, Math.min(a1 - 0.5, sa));
  const cp = perp(parent.start!);

  // Start on the parent centerline, a few units before the doorway, heading
  // straight down the corridor (as a player who has clicked the doorway).
  let player: PlayerState = {
    x: a === 'x' ? sa : cp,
    z: a === 'x' ? cp : sa,
    heading: corridorHeading,
    angularVelocity: 0,
  };
  let activeId = parent.id;
  let targetId: string | null = side.id;
  let phase: 'approach' | 'through' = 'approach';
  let maxDev = 0;

  for (let f = 0; f < 150; f++) {
    const active = world.get(activeId)!.descriptor;
    const liveIds = active.doorways.map((dw) => dw.id);
    if (targetId !== null && !liveIds.includes(targetId)) {
      // The side doorway is no longer live: we crossed into the room. Re-resolve
      // to the first reachable forward exit, mirroring the held-click fallback.
      const cands = active.doorways.filter((dw) => dw.side !== 'start');
      const reach = cands.find((dw) => steering.isReachable(player, dw.entryPoint));
      targetId = reach ? reach.id : null;
      phase = 'approach';
    }
    const tw = active.doorways.find((dw) => dw.id === targetId);
    let tp: { x: number; z: number } | null = null;
    if (tw) {
      phase = doorwayPhase(player, tw, CONFIG, phase);
      tp = doorwayTargetPoint(tw, phase);
    }
    const next = resolveWallCollision(
      world,
      player,
      steering.step(player, tp, CONFIG.simTimestep),
    );
    player = next;
    maxDev = Math.max(maxDev, Math.abs(angleDiff(player.heading, corridorHeading)));
    if (!world.containsEntity(activeId, player.x, player.z)) {
      const here = world.entityAt(player.x, player.z);
      if (here && here.id !== activeId) {
        activeId = here.id;
        world.maintain(activeId);
      }
    }
  }
  return maxDev;
}
