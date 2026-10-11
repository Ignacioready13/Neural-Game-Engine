/**
 * VIEW-SAFETY MEASUREMENT (Stage 2 follow-up, viewUnsafe label)
 * ============================================================
 * The player is a floor POINT with no collision radius (`resolveWallCollision`),
 * so the camera can end up arbitrarily close to a wall. Two things go wrong when
 * it does, and both are pure rendering artifacts (no simulation involvement):
 *
 *   1. NEAR-PLANE CLIP — a wall closer than the camera's near plane is clipped
 *      away entirely, leaving a hole you can see through.
 *   2. SINGLE-SIDED SEE-THROUGH — walls are FrontSide, so a wall whose front
 *      face points away from the camera is culled; if the camera has crossed
 *      that wall's plane, you look straight through it.
 *
 * This module measures, per frame, the clearance from the camera to every
 * rendered triangle of the active entity and its maintained neighbours, and
 * flags the frame when either condition holds. The result is recorded in the
 * manifest as `viewUnsafe`. It is a LABEL ONLY: it never aborts, trims or
 * otherwise changes an episode. Data loaders should drop any training window
 * that touches a flagged frame.
 *
 * Nothing here touches steering, timing, RNG or the renderer's output.
 */
import * as THREE from 'three';
import type { PlayerState } from '../core/types';

export interface ViewSafety {
  /** Min distance from the camera to any triangle considered (Infinity when
   * there is nothing to measure). */
  minClearance: number;
  /** True when the nearest triangle is back-facing to the camera AND lies in
   * front of it — the single-sided see-through case. */
  seeThrough: boolean;
  /** The manifest label. */
  unsafe: boolean;
}

/** A triangle in world space, with its unit normal. */
interface Tri {
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  cx: number; cy: number; cz: number;
  nx: number; ny: number; nz: number;
}

/** An entity group to measure against. `group` is typed `unknown` because the
 * shared core `Entity` (core/types.ts) keeps three.js out of its module graph;
 * the probe casts it to a real Group, which the world generator guarantees. */
export interface MeasurableEntity {
  descriptor: { id: string };
  group: unknown;
}

// Scratch vectors — the probe runs every frame, so it must not allocate.
const _ab = new THREE.Vector3();
const _ac = new THREE.Vector3();
const _ap = new THREE.Vector3();
const _bp = new THREE.Vector3();
const _cp = new THREE.Vector3();
const _e = new THREE.Vector3();
const _closest = new THREE.Vector3();

/**
 * Closest point on triangle (a,b,c) to point p, written into `_closest`.
 * Returns the squared distance (Ericson, Real-Time Collision Detection §5.1.5).
 */
function closestPointOnTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const p = _ap.set(px, py, pz);
  const a = _e.set(ax, ay, az);
  const b = _bp.set(bx, by, bz);
  const c = _cp.set(cx, cy, cz);

  _ab.subVectors(b, a);
  _ac.subVectors(c, a);
  const apx = p.x - a.x, apy = p.y - a.y, apz = p.z - a.z;
  const d1 = _ab.x * apx + _ab.y * apy + _ab.z * apz;
  const d2 = _ac.x * apx + _ac.y * apy + _ac.z * apz;
  if (d1 <= 0 && d2 <= 0) { _closest.copy(a); return p.distanceToSquared(a); }

  const bpx = p.x - b.x, bpy = p.y - b.y, bpz = p.z - b.z;
  const d3 = _ab.x * bpx + _ab.y * bpy + _ab.z * bpz;
  const d4 = _ac.x * bpx + _ac.y * bpy + _ac.z * bpz;
  if (d3 >= 0 && d4 <= d3) { _closest.copy(b); return p.distanceToSquared(b); }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    _closest.set(a.x + _ab.x * v, a.y + _ab.y * v, a.z + _ab.z * v);
    return p.distanceToSquared(_closest);
  }

  const cpx = p.x - c.x, cpy = p.y - c.y, cpz = p.z - c.z;
  const d5 = _ab.x * cpx + _ab.y * cpy + _ab.z * cpz;
  const d6 = _ac.x * cpx + _ac.y * cpy + _ac.z * cpz;
  if (d6 >= 0 && d5 <= d6) { _closest.copy(c); return p.distanceToSquared(c); }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    _closest.set(a.x + _ac.x * w, a.y + _ac.y * w, a.z + _ac.z * w);
    return p.distanceToSquared(_closest);
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    _closest.set(b.x + (c.x - b.x) * w, b.y + (c.y - b.y) * w, b.z + (c.z - b.z) * w);
    return p.distanceToSquared(_closest);
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  _closest.set(a.x + _ab.x * v + _ac.x * w, a.y + _ab.y * v + _ac.y * w, a.z + _ab.z * v + _ac.z * w);
  return p.distanceToSquared(_closest);
}

/** Extract a group's triangles into world space (once per entity). */
function extractTriangles(group: THREE.Object3D): Tri[] {
  const out: Tri[] = [];
  group.updateMatrixWorld(true);
  group.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!(mesh as unknown as { isMesh?: boolean }).isMesh || !mesh.geometry) return;
    const geo = mesh.geometry as THREE.BufferGeometry;
    const pos = geo.getAttribute('position') as THREE.BufferAttribute | undefined;
    if (!pos) return;
    const m = mesh.matrixWorld;
    const vx = new THREE.Vector3();
    const vy = new THREE.Vector3();
    const vz = new THREE.Vector3();
    const emit = (i0: number, i1: number, i2: number): void => {
      vx.fromBufferAttribute(pos, i0).applyMatrix4(m);
      vy.fromBufferAttribute(pos, i1).applyMatrix4(m);
      vz.fromBufferAttribute(pos, i2).applyMatrix4(m);
      const ux = vy.x - vx.x, uy = vy.y - vx.y, uz = vy.z - vx.z;
      const wx = vz.x - vx.x, wy = vz.y - vx.y, wz = vz.z - vx.z;
      let nx = uy * wz - uz * wy;
      let ny = uz * wx - ux * wz;
      let nz = ux * wy - uy * wx;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      out.push({
        ax: vx.x, ay: vx.y, az: vx.z,
        bx: vy.x, by: vy.y, bz: vy.z,
        cx: vz.x, cy: vz.y, cz: vz.z,
        nx, ny, nz,
      });
    };
    const idx = geo.getIndex();
    if (idx) {
      for (let i = 0; i < idx.count; i += 3) emit(idx.getX(i), idx.getX(i + 1), idx.getX(i + 2));
    } else {
      for (let i = 0; i < pos.count; i += 3) emit(i, i + 1, i + 2);
    }
  });
  return out;
}

/**
 * Per-engine triangle cache. Entity ids repeat across episodes but never within
 * one Engine instance, and an entity's geometry never changes, so keying by id
 * is safe; `clear()` on world sync disposes of entities that were removed.
 */
export class ViewSafetyProbe {
  private cache = new Map<string, Tri[]>();

  /** Drop cached geometry (call when the maintained entity set changes). */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Measure clearance from the camera to every rendered triangle of `entities`.
   *
   * @param near   the render camera's near plane (read from the camera)
   * @param margin the `viewUnsafe` margin (config.viewUnsafeMargin)
   */
  measure(
    player: PlayerState,
    eyeHeight: number,
    near: number,
    margin: number,
    entities: ReadonlyArray<MeasurableEntity>,
  ): ViewSafety {
    const camX = player.x;
    const camY = eyeHeight;
    const camZ = player.z;
    const fwdX = Math.cos(player.heading);
    const fwdZ = Math.sin(player.heading);
    const threshold = near + margin;

    let minD2 = Infinity;
    let nearest: Tri | null = null;
    for (const ent of entities) {
      if (!ent.group) continue;
      let tris = this.cache.get(ent.descriptor.id);
      if (!tris) {
        tris = extractTriangles(ent.group as THREE.Object3D);
        this.cache.set(ent.descriptor.id, tris);
      }
      for (const t of tris) {
        const d2 = closestPointOnTriangle(
          camX, camY, camZ,
          t.ax, t.ay, t.az, t.bx, t.by, t.bz, t.cx, t.cy, t.cz,
        );
        if (d2 < minD2) { minD2 = d2; nearest = t; }
      }
    }

    if (!nearest) return { minClearance: Infinity, seeThrough: false, unsafe: false };

    const minClearance = Math.sqrt(minD2);
    // Back-facing (camera on the back side of the triangle's plane) AND in front
    // of the camera — the see-through case.
    const side = nearest.nx * (camX - nearest.ax) + nearest.ny * (camY - nearest.ay) + nearest.nz * (camZ - nearest.az);
    const inFront = (nearest.ax - camX) * fwdX + (nearest.az - camZ) * fwdZ > 0 ||
      (nearest.bx - camX) * fwdX + (nearest.bz - camZ) * fwdZ > 0 ||
      (nearest.cx - camX) * fwdX + (nearest.cz - camZ) * fwdZ > 0;
    const seeThrough = side < 0 && inFront;

    return { minClearance, seeThrough, unsafe: minClearance < threshold || seeThrough };
  }
}
