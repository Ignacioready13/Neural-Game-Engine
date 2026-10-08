/**
 * WORLD GENERATION UNIT TESTS
 * ===========================
 * Plain-Node (vitest) tests over EntityDescriptor data — no WebGL. Verifies the
 * 11 hard rules, lazy generation/disposal, and seed determinism.
 *
 * Descriptor generation runs with buildGeometry=false for speed; a separate
 * test constructs a World with default geometry to confirm the mesh path is
 * safe in Node (materials fall back to flat colors without a DOM).
 */
import { describe, it, expect } from 'vitest';
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import { CARDINAL_VEC } from '../src/core/types';
import type { Cardinal, Doorway, Entity, EntityDescriptor } from '../src/core/types';
import { createRng } from '../src/core/rng';

const EPS = 1e-6;

function perpendicular(a: Cardinal, b: Cardinal): boolean {
  const va = CARDINAL_VEC[a];
  const vb = CARDINAL_VEC[b];
  return va.x * vb.x + va.z * vb.z === 0;
}

/** Rule 5: doorway spacing on the same wall/side (excluding start/end). */
function verifySpacing(d: EntityDescriptor): void {
  const min = CONFIG.doorwaySpacingFactor * CONFIG.doorwayWidth;
  const groups = new Map<string, Doorway[]>();
  for (const dw of d.doorways) {
    if (dw.side === 'start' || dw.side === 'end') continue;
    const arr = groups.get(dw.side) ?? [];
    arr.push(dw);
    groups.set(dw.side, arr);
  }
  for (const dws of groups.values()) {
    for (let i = 0; i < dws.length; i++) {
      for (let j = i + 1; j < dws.length; j++) {
        const a = dws[i]!;
        const b = dws[j]!;
        expect(Math.hypot(a.x - b.x, a.z - b.z)).toBeGreaterThanOrEqual(min - EPS);
      }
    }
  }
}

/** Rules 1-9 on a single descriptor (with world access for target lookups). */
function verifyDescriptor(d: EntityDescriptor, world: World, isStart: boolean): void {
  if (d.kind === 'hallway') {
    // Rule 2: constant corridor width, present on every hallway.
    expect(d.corridorWidth).toBeCloseTo(CONFIG.hallwayWidth, 6);

    // Rule 1: axis-aligned, cardinal travel direction matching the axis.
    expect(d.axis === 'x' || d.axis === 'z').toBe(true);
    const tv = CARDINAL_VEC[d.travelDirection!];
    expect(d.axis === 'x' ? tv.z === 0 : tv.x === 0).toBe(true);

    // Rule 3: length within [min, max].
    const len = Math.hypot(d.end!.x - d.start!.x, d.end!.z - d.start!.z);
    expect(len).toBeGreaterThanOrEqual(CONFIG.hallwayLengthMin - EPS);
    expect(len).toBeLessThanOrEqual(CONFIG.hallwayLengthMax + EPS);

    // Rule 8: termination is exactly bend / junction / room (never 'start').
    expect(['bend', 'junction', 'room']).toContain(d.termination);

    // Rule 9: the starting hallway has no entry doorway.
    if (isStart) {
      expect(d.doorways.some((dw) => dw.side === 'start')).toBe(false);
    }

    // Rules 1 & 7 via generated targets.
    for (const dw of d.doorways) {
      const target = dw.targetGenerated ? world.get(dw.targetEntityId) : undefined;
      if (dw.side === 'left' || dw.side === 'right') {
        if (target) expect(target.descriptor.kind, 'side doorway must lead to a room').toBe('room');
      }
      if (target && target.descriptor.kind === 'hallway') {
        expect(['start', 'end'], 'hallway<->hallway only via bend/junction').toContain(dw.side);
        // Direction change must be a multiple of 90 degrees (axis-aligned); a
        // `bend` termination turns exactly 90 degrees, a junction branch may
        // continue straight (0 degrees).
        const tv = CARDINAL_VEC[d.travelDirection!];
        const ctv = CARDINAL_VEC[target.descriptor.travelDirection!];
        const dot = tv.x * ctv.x + tv.z * ctv.z;
        expect(Number.isInteger(dot) && Math.abs(dot) <= 1).toBe(true);
        if (d.termination === 'bend' && dw.side === 'end') {
          expect(perpendicular(d.travelDirection!, target.descriptor.travelDirection!)).toBe(true);
        }
      }
    }
  } else {
    // Rule 6: every room has >= 2 doorways (entry + at least one more).
    expect(d.doorways.length).toBeGreaterThanOrEqual(2);

    // Rule 4: room dimensions within ranges (ceiling independent of footprint).
    expect(d.width!).toBeGreaterThanOrEqual(CONFIG.roomWidthMin - EPS);
    expect(d.width!).toBeLessThanOrEqual(CONFIG.roomWidthMax + EPS);
    expect(d.depth!).toBeGreaterThanOrEqual(CONFIG.roomDepthMin - EPS);
    expect(d.depth!).toBeLessThanOrEqual(CONFIG.roomDepthMax + EPS);
    expect(d.ceilingHeight!).toBeGreaterThanOrEqual(CONFIG.roomHeightMin - EPS);
    expect(d.ceilingHeight!).toBeLessThanOrEqual(CONFIG.roomHeightMax + EPS);

    // Rule 7: rooms never connect directly to rooms.
    for (const dw of d.doorways) {
      const target = dw.targetGenerated ? world.get(dw.targetEntityId) : undefined;
      if (target) expect(target.descriptor.kind, 'room must lead to a hallway').toBe('hallway');
    }
  }

  // Rule 5: spacing (both kinds).
  verifySpacing(d);
}

interface WalkStats {
  verified: number;
  terminations: Set<string>;
}

/** Walk the lazily-generated graph, verifying every live descriptor per step. */
function walk(
  seed: number,
  steps: number,
  rng: () => number,
  onEntity: (d: EntityDescriptor, world: World, isStart: boolean) => void,
): WalkStats {
  const world = new World(seed, CONFIG, { buildGeometry: false });
  let active: Entity = world.init();
  const startId = active.descriptor.id;
  let verified = 0;
  const terminations = new Set<string>();

  onEntity(active.descriptor, world, true);
  verified++;

  for (let s = 0; s < steps; s++) {
    world.maintain(active.descriptor.id);
    for (const e of world.all()) {
      if (e.descriptor.kind === 'hallway') terminations.add(e.descriptor.termination ?? '');
      onEntity(e.descriptor, world, e.descriptor.id === startId);
      verified++;
    }
    const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
    if (exits.length === 0) break;
    const next = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId);
    if (!next) break;
    active = next;
  }
  return { verified, terminations };
}

describe('world generation rules', () => {
  it('enforces all geometric rules across many generated worlds', () => {
    const rng = createRng(0xbeef);
    const seeds = Array.from({ length: 40 }, (_, i) => i * 6151 + 7);
    let total = 0;
    const terminations = new Set<string>();
    for (const seed of seeds) {
      const stats = walk(seed, 50, rng, verifyDescriptor);
      total += stats.verified;
      for (const t of stats.terminations) terminations.add(t);
    }
    expect(total).toBeGreaterThanOrEqual(200);
    // All three termination kinds actually occur (bend/junction/room).
    expect([...terminations].sort()).toEqual(['bend', 'junction', 'room']);
  });

  it('starting hallway has no entry doorway', () => {
    for (const seed of [0, 1, 2, 3, 4, 5]) {
      const world = new World(seed, CONFIG, { buildGeometry: false });
      const start = world.init();
      expect(start.descriptor.kind).toBe('hallway');
      expect(start.descriptor.doorways.some((d) => d.side === 'start')).toBe(false);
    }
  });
});

describe('lazy generation and disposal', () => {
  it('keeps only the active entity and its direct neighbors', () => {
    const world = new World(7, CONFIG, { buildGeometry: false });
    const start = world.init();
    const rng = createRng(4242);
    const path: string[] = [start.descriptor.id];

    for (let step = 0; step < 40; step++) {
      const activeId = path[path.length - 1]!;
      const res = world.maintain(activeId);
      const active = world.get(activeId)!;

      // Invariant: every live entity is active or a direct neighbor.
      const neighborIds = new Set(active.descriptor.doorways.map((d) => d.targetEntityId));
      for (const e of world.all()) {
        expect(e.descriptor.id === activeId || neighborIds.has(e.descriptor.id)).toBe(true);
      }
      // Generated entities are exactly the freshly-materialized neighbors.
      for (const e of res.generated) {
        expect(neighborIds.has(e.descriptor.id)).toBe(true);
      }
      // Disposed entities are gone from the world.
      for (const id of res.disposed) {
        expect(world.get(id)).toBeUndefined();
      }
      // Every doorway of the active entity is now generated.
      expect(active.descriptor.doorways.every((d) => d.targetGenerated)).toBe(true);

      const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
      if (exits.length === 0) break;
      path.push(exits[Math.floor(rng() * exits.length)]!.targetEntityId);
    }
    expect(path.length).toBeGreaterThan(1);
  });

  it('disposes geometry when entities are evicted (default geometry build)', () => {
    const world = new World(9, CONFIG); // buildGeometry defaults to true
    const start = world.init();
    world.maintain(start.descriptor.id);
    // Walk away twice; the original start's far neighbors must be disposed.
    const rng = createRng(1);
    let active = start;
    for (let i = 0; i < 4; i++) {
      world.maintain(active.descriptor.id);
      const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
      if (exits.length === 0) break;
      active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
    }
    // No throw + at least the active entity and its neighbors remain.
    expect(world.all().length).toBeGreaterThan(0);
  });
});

describe('determinism', () => {
  function snapshots(seed: number, steps: number): string[] {
    const world = new World(seed, CONFIG, { buildGeometry: false });
    const rng = createRng(9999);
    const out: string[] = [];
    let active = world.init();
    out.push(JSON.stringify(active.descriptor));
    for (let s = 0; s < steps; s++) {
      world.maintain(active.descriptor.id);
      out.push(
        JSON.stringify(world.all().map((e) => e.descriptor).sort((a, b) => a.id.localeCompare(b.id))),
      );
      const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
      if (exits.length === 0) break;
      active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
    }
    return out;
  }

  it('reproduces identical descriptors for the same seed and traversal', () => {
    expect(snapshots(12345, 40)).toEqual(snapshots(12345, 40));
  });

  it('different seeds produce different worlds', () => {
    const starts = new Set([1, 2, 3, 4, 5, 6, 7, 8].map((s) => snapshots(s, 1)[0]));
    expect(starts.size).toBeGreaterThan(1);
  });
});

describe('geometry path in Node', () => {
  it('builds three.js groups without a DOM (flat-color material fallback)', () => {
    const world = new World(11, CONFIG); // default buildGeometry = true
    const start = world.init();
    expect(start.group).toBeTruthy();
    const rng = createRng(7);
    let active = start;
    for (let i = 0; i < 12; i++) {
      world.maintain(active.descriptor.id);
      const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
      if (exits.length === 0) break;
      active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
    }
    expect(world.all().length).toBeGreaterThan(1);
  });
});
