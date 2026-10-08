import { describe, expect, it } from 'vitest';
import { Steering } from '../src/control/steering';
import { resolveTarget } from '../src/control/doorwayResolution';
import { CONFIG, type EngineConfig } from '../config/config';
import { angleDiff } from '../src/core/math';
import type { CandidateDoorway, PlayerState, Vec2 } from '../src/core/types';

const EPS = 1e-9;

function makeConfig(overrides: Partial<EngineConfig> = {}): EngineConfig {
  return { ...CONFIG, ...overrides };
}

function cand(id: string, screenX: number, reachable = true): CandidateDoorway {
  return { doorwayId: id, screenX, reachable, screenDistance: 0 };
}

function player(overrides: Partial<PlayerState> = {}): PlayerState {
  return { x: 0, z: 0, heading: 0, angularVelocity: 0, ...overrides };
}

describe('handleClick (debounce + held control signal)', () => {
  it('accepts the first click and sets the held value', () => {
    const s = new Steering(makeConfig({ debounceSeconds: 0.15 }));
    expect(s.handleClick(0.5, 1.0).accepted).toBe(true);
    expect(s.lastClickedX).toBe(0.5);
    expect(s.lastClickTime).toBe(1.0);
    expect(s.lastClickScreenX).toBe(0.5);
  });

  it('ignores a rapid click for steering but still records it for the pulse', () => {
    const s = new Steering(makeConfig({ debounceSeconds: 0.15 }));
    s.handleClick(0.4, 0.0);
    const r = s.handleClick(-0.7, 0.05);
    expect(r.accepted).toBe(false);
    expect(s.lastClickedX).toBe(0.4); // held value unchanged
    expect(s.lastClickTime).toBe(0.05); // every click pulses
    expect(s.lastClickScreenX).toBe(-0.7);
  });

  it('accepts a click once the debounce window has elapsed (>= boundary)', () => {
    const s = new Steering(makeConfig({ debounceSeconds: 0.15 }));
    s.handleClick(0.4, 0.0);
    expect(s.handleClick(-0.7, 0.15).accepted).toBe(true);
    expect(s.lastClickedX).toBe(-0.7);
  });

  it('holds the signal constant across frames until the next accepted click', () => {
    const s = new Steering(makeConfig());
    s.handleClick(0.25, 0.0);
    for (let i = 0; i < 5; i++) {
      s.update([cand('a', 0.0), cand('b', -0.5)], ['a', 'b'], i / 20);
      expect(s.lastClickedX).toBe(0.25);
    }
  });

  it('clamps screen X to the normalized -1..1 range', () => {
    const s = new Steering(makeConfig());
    s.handleClick(5.0, 0.0);
    expect(s.lastClickedX).toBe(1.0);
    s.handleClick(-99, 1.0);
    expect(s.lastClickedX).toBe(-1.0);
  });
});

describe('resolveTarget (nearest + same-side bias)', () => {
  it('picks the nearest reachable candidate', () => {
    const cs = [cand('far-left', -0.9), cand('near', 0.05), cand('far-right', 0.8)];
    expect(resolveTarget(cs, 0, null)).toBe('near');
  });

  it('defaults the reference to screen center (0) when clickX is null', () => {
    const cs = [cand('left', -0.6), cand('center', 0.02), cand('right', 0.9)];
    expect(resolveTarget(cs, null, null)).toBe('center');
  });

  it('uses the click X as the reference', () => {
    const cs = [cand('a', -0.4), cand('b', 0.6), cand('c', -0.85)];
    expect(resolveTarget(cs, -0.5, null)).toBe('a');
  });

  it('biases the outward search to the same side as the nearest (unreachable) candidate', () => {
    const cs = [
      cand('nearest-left', -0.1, false), // nearest, left of center, unreachable
      cand('near-right', 0.2, true), // closer in distance than far-left, but other side
      cand('far-left', -0.6, true), // same side as nearest, farther out
    ];
    expect(resolveTarget(cs, 0, null)).toBe('far-left');
  });

  it('crosses to the other side only after exhausting the same side', () => {
    const cs = [
      cand('nearest-left', -0.1, false),
      cand('far-left', -0.6, false),
      cand('near-right', 0.2, true),
    ];
    expect(resolveTarget(cs, 0, null)).toBe('near-right');
  });

  it('keeps the previous target when nothing is reachable and it is still live', () => {
    const cs = [cand('a', -0.2, false), cand('b', 0.4, false)];
    expect(resolveTarget(cs, 0, 'a')).toBe('a');
  });

  it('returns null when nothing is reachable and the previous target is gone', () => {
    const cs = [cand('a', -0.2, false), cand('b', 0.4, false)];
    expect(resolveTarget(cs, 0, 'old')).toBe(null);
  });

  it('returns null for an empty candidate set', () => {
    expect(resolveTarget([], 0, 'a')).toBe(null);
    expect(resolveTarget([], null, null)).toBe(null);
  });
});

describe('update (target plumbing)', () => {
  it('stores and returns the resolved target id', () => {
    const s = new Steering(makeConfig());
    expect(s.update([cand('a', -0.9), cand('b', 0.1)], ['a', 'b'], 0.0)).toBe('b');
    expect(s.targetDoorwayId).toBe('b');
  });

  it('clears the target when no candidate is reachable and the previous is gone', () => {
    const s = new Steering(makeConfig());
    s.update([cand('a', 0.1)], ['a'], 0.0);
    expect(s.targetDoorwayId).toBe('a');
    expect(s.update([cand('x', 0.2, false)], ['x'], 0.1)).toBe(null);
    expect(s.targetDoorwayId).toBe(null);
  });
});

describe('step (jerk-limited kinematics)', () => {
  it('never exceeds the jerk limit between consecutive steps', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    const dt = cfg.simTimestep;
    let p = player();
    // Target 90 deg to the right so the heading error stays large for a while.
    const target: Vec2 = { x: 0, z: 10 };
    const omegas: number[] = [p.angularVelocity];
    for (let i = 0; i < 200; i++) {
      p = s.step(p, target, dt);
      omegas.push(p.angularVelocity);
    }
    const jerkLimit = cfg.jerkThreshold * dt;
    for (let i = 1; i < omegas.length; i++) {
      const dw = omegas[i]! - omegas[i - 1]!;
      expect(Math.abs(dw)).toBeLessThanOrEqual(jerkLimit + EPS);
    }
  });

  it('never exceeds the max turn rate', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    let p = player();
    const target: Vec2 = { x: 0, z: 10 };
    for (let i = 0; i < 200; i++) {
      p = s.step(p, target, cfg.simTimestep);
      expect(Math.abs(p.angularVelocity)).toBeLessThanOrEqual(cfg.maxTurnRate + EPS);
    }
  });

  it('moves exactly walkSpeed*dt per step (constant forward speed)', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    const p0 = player();
    const p1 = s.step(p0, { x: 10, z: 0 }, cfg.simTimestep);
    const moved = Math.hypot(p1.x - p0.x, p1.z - p0.z);
    expect(moved).toBeCloseTo(cfg.walkSpeed * cfg.simTimestep, 12);
  });

  it('keeps straight ahead when target is null (heading unchanged, omega decays)', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    let p = player({ heading: 0.7, angularVelocity: 0.3 });
    const before = p.heading;
    p = s.step(p, null, cfg.simTimestep);
    expect(p.heading).toBeCloseTo(before, 12);
    expect(Math.abs(p.angularVelocity)).toBeLessThan(0.3);
  });
});

describe('continuity across retargeting', () => {
  it('produces no heading jump larger than maxTurnRate*dt on retarget', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    const dt = cfg.simTimestep;
    let p = player();
    for (let i = 0; i < 30; i++) p = s.step(p, { x: 0, z: -10 }, dt); // turn left
    const before = p;
    const after = s.step(before, { x: 0, z: 10 }, dt); // retarget to the right
    const dHeading = Math.abs(angleDiff(before.heading, after.heading));
    expect(dHeading).toBeLessThanOrEqual(cfg.maxTurnRate * dt + EPS);
    const moved = Math.hypot(after.x - before.x, after.z - before.z);
    expect(moved).toBeCloseTo(cfg.walkSpeed * dt, 12);
  });

  it('keeps omega continuous (change bounded by jerk limit) across retarget', () => {
    const cfg = makeConfig();
    const s = new Steering(cfg);
    const dt = cfg.simTimestep;
    let p = player();
    for (let i = 0; i < 30; i++) p = s.step(p, { x: 0, z: -10 }, dt);
    const before = p.angularVelocity;
    const after = s.step(p, { x: 0, z: 10 }, dt).angularVelocity;
    expect(Math.abs(after - before)).toBeLessThanOrEqual(cfg.jerkThreshold * dt + EPS);
  });
});

describe('isReachable', () => {
  const s = new Steering(makeConfig());

  it('rejects null / NaN / Infinity entries', () => {
    expect(s.isReachable(player(), null as unknown as Vec2)).toBe(false);
    expect(s.isReachable(player(), { x: NaN, z: 0 })).toBe(false);
    expect(s.isReachable(player(), { x: 0, z: NaN })).toBe(false);
    expect(s.isReachable(player(), { x: Infinity, z: 0 })).toBe(false);
  });

  it('rejects points behind the player', () => {
    // heading 0 faces +X, so -X is directly behind.
    expect(s.isReachable(player(), { x: -10, z: 0 })).toBe(false);
  });

  it('rejects points outside the reachability angle', () => {
    // 90 deg to the side exceeds the 75 deg default front cone.
    expect(s.isReachable(player(), { x: 0, z: 10 })).toBe(false);
  });

  it('rejects points too close to complete the required turn', () => {
    // 45 deg ahead but only ~0.7 units away: not enough room to finish the turn.
    expect(s.isReachable(player(), { x: 0.5, z: 0.5 })).toBe(false);
  });

  it('accepts a point straight ahead with plenty of distance', () => {
    expect(s.isReachable(player(), { x: 10, z: 0 })).toBe(true);
  });

  it('accepts a point within angle and with enough distance', () => {
    expect(s.isReachable(player(), { x: 10, z: 10 })).toBe(true);
  });
});
