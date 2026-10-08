import type { Vec2 } from './types';

/** Wrap an angle to (-PI, PI]. */
export function wrapAngle(a: number): number {
  let x = a % (Math.PI * 2);
  if (x > Math.PI) x -= Math.PI * 2;
  else if (x <= -Math.PI) x += Math.PI * 2;
  return x;
}

/** Signed difference b - a, wrapped to (-PI, PI]. */
export function angleDiff(a: number, b: number): number {
  return wrapAngle(b - a);
}

export function vecAdd(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, z: a.z + b.z };
}

export function vecSub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, z: a.z - b.z };
}

export function vecScale(a: Vec2, s: number): Vec2 {
  return { x: a.x * s, z: a.z * s };
}

export function vecLen(a: Vec2): number {
  return Math.hypot(a.x, a.z);
}

export function vecDist(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

export function vecNormalize(a: Vec2): Vec2 {
  const l = Math.hypot(a.x, a.z);
  return l === 0 ? { x: 0, z: 0 } : { x: a.x / l, z: a.z / l };
}

/** Rotate a Vec2 by +90 degrees (counter-clockwise in XZ seen from +Y). */
export function vecRotate90(a: Vec2): Vec2 {
  return { x: -a.z, z: a.x };
}

/** Rotate a Vec2 by -90 degrees (clockwise). */
export function vecRotateNeg90(a: Vec2): Vec2 {
  return { x: a.z, z: -a.x };
}

/** Linear interpolation. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Clamp x into [min, max]. */
export function clamp(x: number, min: number, max: number): number {
  return x < min ? min : x > max ? max : x;
}
