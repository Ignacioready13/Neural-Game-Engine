/**
 * Mesh construction for world entities. Builds a THREE.Group from a pure
 * EntityDescriptor (no three.js objects inside descriptors, so rule tests stay
 * WebGL-free).
 *
 * Walls are built as axis-aligned quads split around doorway openings
 * (full-height pieces + lintels above regular doorways; corner openings are
 * full height). Floor/ceiling use world-space UVs (uv = world / TILE_WORLD_SIZE)
 * so the seamless canvas textures tile without stretching.
 */
import * as THREE from 'three';
import type { Cardinal, EntityDescriptor, Vec2 } from '../core/types';
import { CARDINAL_VEC, OPPOSITE } from '../core/types';
import type { EngineConfig } from '../../config/config';
import { vecAdd, vecScale, vecRotate90, vecRotateNeg90 } from '../core/math';
import { createMaterials, TILE_WORLD_SIZE, WorldMaterials } from './materials';
import { leftOf, rightOf } from './layout';

export interface BuildOptions {
  /** Skip the hallway child's start wall (coplanar with a parent wall). Rooms
   *  always build their own entry wall (see buildRoom). */
  skipEntryWall?: boolean;
  /** Parent's travel direction — bend children need it to place the corner. */
  parentDir?: Cardinal;
}

interface Rect {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
}

interface Opening {
  center: number;
  width: number;
  height: number;
  full: boolean;
}

interface WallSeg {
  constX: boolean;
  fixed: number;
  spanMin: number;
  spanMax: number;
  height: number;
  normal: Vec2;
  openings: Opening[];
}

function wallCoord(v: Vec2, constX: boolean): number {
  return constX ? v.x : v.z;
}

function spanCoord(v: Vec2, constX: boolean): number {
  return constX ? v.z : v.x;
}

function vecEq(a: Vec2, b: Vec2): boolean {
  return a.x === b.x && a.z === b.z;
}

function addQuad(
  group: THREE.Group,
  mat: THREE.Material,
  corners: THREE.Vector3[],
  normal: THREE.Vector3,
  uv: THREE.Vector2[],
): void {
  const positions = new Float32Array(corners.length * 3);
  const normals = new Float32Array(corners.length * 3);
  const uvs = new Float32Array(corners.length * 2);
  for (let i = 0; i < corners.length; i++) {
    const v = corners[i]!;
    positions[i * 3] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;
    normals[i * 3] = normal.x;
    normals[i * 3 + 1] = normal.y;
    normals[i * 3 + 2] = normal.z;
    const u = uv[i]!;
    uvs[i * 2] = u.x;
    uvs[i * 2 + 1] = u.y;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  // Normalize winding against the intended normal: the front face (visible
  // under THREE.FrontSide) must wind so its geometric normal points along
  // `normal`. Callers pass corners in an arbitrary but consistent order — some
  // walls (and the floor/ceiling) were historically wound opposite their normal
  // and only looked right because the material was DoubleSide — so flip the
  // triangles here when the winding opposes `normal`.
  const e1 = new THREE.Vector3().subVectors(corners[1]!, corners[0]!);
  const e2 = new THREE.Vector3().subVectors(corners[2]!, corners[0]!);
  const winding = new THREE.Vector3().crossVectors(e1, e2);
  geo.setIndex(winding.dot(normal) >= 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]);
  group.add(new THREE.Mesh(geo, mat));
}

function floorCeiling(group: THREE.Group, mat: THREE.Material, rect: Rect, y: number, up: boolean): void {
  const { x0, z0, x1, z1 } = rect;
  const corners = up
    ? [new THREE.Vector3(x0, y, z0), new THREE.Vector3(x1, y, z0), new THREE.Vector3(x1, y, z1), new THREE.Vector3(x0, y, z1)]
    : [new THREE.Vector3(x0, y, z0), new THREE.Vector3(x0, y, z1), new THREE.Vector3(x1, y, z1), new THREE.Vector3(x1, y, z0)];
  const normal = new THREE.Vector3(0, up ? 1 : -1, 0);
  const uv = corners.map((c) => new THREE.Vector2(c.x / TILE_WORLD_SIZE, c.z / TILE_WORLD_SIZE));
  addQuad(group, mat, corners, normal, uv);
}

function wallPiece(
  group: THREE.Group,
  mat: THREE.Material,
  seg: WallSeg,
  a: number,
  b: number,
  y0: number,
  y1: number,
): void {
  const corners = seg.constX
    ? [
        new THREE.Vector3(seg.fixed, y0, a),
        new THREE.Vector3(seg.fixed, y0, b),
        new THREE.Vector3(seg.fixed, y1, b),
        new THREE.Vector3(seg.fixed, y1, a),
      ]
    : [
        new THREE.Vector3(a, y0, seg.fixed),
        new THREE.Vector3(b, y0, seg.fixed),
        new THREE.Vector3(b, y1, seg.fixed),
        new THREE.Vector3(a, y1, seg.fixed),
      ];
  const normal = new THREE.Vector3(seg.normal.x, 0, seg.normal.z);
  const uv = [
    new THREE.Vector2(a / TILE_WORLD_SIZE, y0 / TILE_WORLD_SIZE),
    new THREE.Vector2(b / TILE_WORLD_SIZE, y0 / TILE_WORLD_SIZE),
    new THREE.Vector2(b / TILE_WORLD_SIZE, y1 / TILE_WORLD_SIZE),
    new THREE.Vector2(a / TILE_WORLD_SIZE, y1 / TILE_WORLD_SIZE),
  ];
  addQuad(group, mat, corners, normal, uv);
}

/** World-space point on a wall, given its (span, y, fixed) coordinates. */
function wallPoint(seg: WallSeg, span: number, y: number, fixed: number): THREE.Vector3 {
  return seg.constX
    ? new THREE.Vector3(fixed, y, span)
    : new THREE.Vector3(span, y, fixed);
}

/** The wall's fixed-axis coordinate moves OUTWARD (opposite `seg.normal`, into
 * the void) by `thickness`. Returns [inner, outer] fixed coordinates. */
function wallThicknessRange(seg: WallSeg, thickness: number): [number, number] {
  const normalFixed = seg.constX ? seg.normal.x : seg.normal.z;
  const inner = seg.fixed;
  const outer = seg.fixed - normalFixed * thickness;
  return [inner, outer];
}

/** Vertical jamb face at `span`, spanning [y0, y1], facing `dir` along the span
 * axis (into the opening). This is what makes a doorway read as having wall
 * thickness when the player walks through. */
function addJamb(
  group: THREE.Group,
  mat: THREE.Material,
  seg: WallSeg,
  span: number,
  y0: number,
  y1: number,
  thickness: number,
  dir: 1 | -1,
): void {
  const [inner, outer] = wallThicknessRange(seg, thickness);
  const corners = [
    wallPoint(seg, span, y0, inner),
    wallPoint(seg, span, y0, outer),
    wallPoint(seg, span, y1, outer),
    wallPoint(seg, span, y1, inner),
  ];
  const normal = seg.constX ? new THREE.Vector3(0, 0, dir) : new THREE.Vector3(dir, 0, 0);
  const uv = corners.map((c) =>
    new THREE.Vector2((seg.constX ? c.x : c.z) / TILE_WORLD_SIZE, c.y / TILE_WORLD_SIZE),
  );
  addQuad(group, mat, corners, normal, uv);
}

/** Underside of a lintel (the top of a doorway opening), facing down into the
 * opening. */
function addLintel(
  group: THREE.Group,
  mat: THREE.Material,
  seg: WallSeg,
  spanA: number,
  spanB: number,
  y: number,
  thickness: number,
): void {
  const [inner, outer] = wallThicknessRange(seg, thickness);
  const corners = [
    wallPoint(seg, spanA, y, inner),
    wallPoint(seg, spanB, y, inner),
    wallPoint(seg, spanB, y, outer),
    wallPoint(seg, spanA, y, outer),
  ];
  const normal = new THREE.Vector3(0, -1, 0);
  const uv = corners.map((c) =>
    new THREE.Vector2(
      (seg.constX ? c.z : c.x) / TILE_WORLD_SIZE,
      (seg.constX ? c.x : c.z) / TILE_WORLD_SIZE,
    ),
  );
  addQuad(group, mat, corners, normal, uv);
}

function buildWall(group: THREE.Group, mat: THREE.Material, seg: WallSeg, thickness: number): void {
  const sorted = seg.openings.slice().sort((a, b) => a.center - b.center);
  let cursor = seg.spanMin;
  for (const o of sorted) {
    const l = Math.max(o.center - o.width / 2, seg.spanMin);
    const r = Math.min(o.center + o.width / 2, seg.spanMax);
    if (l > cursor) wallPiece(group, mat, seg, cursor, l, 0, seg.height);
    if (r > l) {
      // Wall thickness: the two jambs (opening sides) and, for a normal-height
      // doorway, the lintel underside. Extruded outward (into the void) so they
      // never narrow the opening or change floor-based collision.
      const jambTop = o.full ? seg.height : o.height;
      addJamb(group, mat, seg, l, 0, jambTop, thickness, 1); // left jamb, faces +span
      addJamb(group, mat, seg, r, 0, jambTop, thickness, -1); // right jamb, faces -span
      if (!o.full) {
        addLintel(group, mat, seg, l, r, o.height, thickness); // lintel underside
        wallPiece(group, mat, seg, l, r, o.height, seg.height);
      }
    }
    cursor = Math.max(cursor, r);
  }
  if (cursor < seg.spanMax) wallPiece(group, mat, seg, cursor, seg.spanMax, 0, seg.height);
}

function sideWallSeg(s: Vec2, e: Vec2, sideVec: Vec2, W: number, H: number, openings: Opening[]): WallSeg {
  const constX = sideVec.x !== 0;
  const fixedPt = vecAdd(s, vecScale(sideVec, W / 2));
  const c0 = spanCoord(s, constX);
  const c1 = spanCoord(e, constX);
  return {
    constX,
    fixed: wallCoord(fixedPt, constX),
    spanMin: Math.min(c0, c1),
    spanMax: Math.max(c0, c1),
    height: H,
    normal: { x: -sideVec.x, z: -sideVec.z },
    openings,
  };
}

function crossWallSeg(anchor: Vec2, outward: Vec2, W: number, H: number, openings: Opening[]): WallSeg {
  const constX = outward.x !== 0;
  const p = spanCoord(anchor, constX);
  return {
    constX,
    fixed: wallCoord(anchor, constX),
    spanMin: p - W / 2,
    spanMax: p + W / 2,
    height: H,
    normal: { x: -outward.x, z: -outward.z },
    openings,
  };
}

function corridorRect(s: Vec2, e: Vec2, axis: 'x' | 'z', W: number): Rect {
  if (axis === 'x') {
    return { x0: Math.min(s.x, e.x), x1: Math.max(s.x, e.x), z0: s.z - W / 2, z1: s.z + W / 2 };
  }
  return { x0: s.x - W / 2, x1: s.x + W / 2, z0: Math.min(s.z, e.z), z1: Math.max(s.z, e.z) };
}

function localRect(
  origin: Vec2,
  aVec: Vec2,
  pVec: Vec2,
  a0: number,
  a1: number,
  p0: number,
  p1: number,
): Rect {
  const pts = [
    vecAdd(vecAdd(origin, vecScale(aVec, a0)), vecScale(pVec, p0)),
    vecAdd(vecAdd(origin, vecScale(aVec, a1)), vecScale(pVec, p0)),
    vecAdd(vecAdd(origin, vecScale(aVec, a1)), vecScale(pVec, p1)),
    vecAdd(vecAdd(origin, vecScale(aVec, a0)), vecScale(pVec, p1)),
  ];
  const xs = pts.map((p) => p.x);
  const zs = pts.map((p) => p.z);
  return { x0: Math.min(...xs), x1: Math.max(...xs), z0: Math.min(...zs), z1: Math.max(...zs) };
}

/** Bend child floor = corridor minus the shared corner square (two rects). */
function bendChildFloorRects(s: Vec2, dirVec: Vec2, parentDirVec: Vec2, W: number, L: number): Rect[] {
  return [
    localRect(s, dirVec, parentDirVec, 0, L, 0, W / 2),
    localRect(s, dirVec, parentDirVec, W / 2, L, -W / 2, 0),
  ];
}

function nodeEdgeSeg(center: Vec2, W: number, H: number, dirOut: Cardinal): WallSeg {
  const constX = dirOut === 'east' || dirOut === 'west';
  const sign = dirOut === 'east' || dirOut === 'south' ? 1 : -1;
  const inward = CARDINAL_VEC[OPPOSITE[dirOut]];
  return {
    constX,
    fixed: constX ? center.x + sign * (W / 2) : center.z + sign * (W / 2),
    spanMin: constX ? center.z - W / 2 : center.x - W / 2,
    spanMax: constX ? center.z + W / 2 : center.x + W / 2,
    height: H,
    normal: { x: inward.x, z: inward.z },
    openings: [],
  };
}

function buildRoom(
  group: THREE.Group,
  mats: WorldMaterials,
  desc: EntityDescriptor,
  config: EngineConfig,
): void {
  const c = desc.center!;
  const w = desc.width!;
  const d = desc.depth!;
  const H = desc.ceilingHeight!;
  const rect = { x0: c.x - w / 2, z0: c.z - d / 2, x1: c.x + w / 2, z1: c.z + d / 2 };
  floorCeiling(group, mats.floor, rect, 0, true);
  floorCeiling(group, mats.ceiling, rect, H, false);

  const walls: Array<{ side: Cardinal; seg: WallSeg }> = [
    { side: 'east', seg: crossWallSeg({ x: c.x + w / 2, z: c.z }, { x: 1, z: 0 }, d, H, []) },
    { side: 'west', seg: crossWallSeg({ x: c.x - w / 2, z: c.z }, { x: -1, z: 0 }, d, H, []) },
    { side: 'north', seg: crossWallSeg({ x: c.x, z: c.z - d / 2 }, { x: 0, z: -1 }, w, H, []) },
    { side: 'south', seg: crossWallSeg({ x: c.x, z: c.z + d / 2 }, { x: 0, z: 1 }, w, H, []) },
  ];

  // Every doorway (entry + additional) becomes an opening in its wall. The entry
  // wall is built even though it is coplanar with the parent's wall: the two are
  // back-to-back with opposite normals (no z-fighting), and the room's entry
  // wall must cover the full room footprint, which often extends past the
  // parent's wall/doorway — skipping it left a see-through gap.
  for (const dw of desc.doorways) {
    const side = dw.side as Cardinal;
    const wall = walls.find((wl) => wl.side === side);
    if (!wall) continue;
    const center = side === 'north' || side === 'south' ? dw.x : dw.z;
    wall.seg.openings.push({ center, width: dw.width, height: dw.height, full: false });
  }

  for (const wall of walls) {
    buildWall(group, mats.wall, wall.seg, config.wallThickness);
  }
}

function buildJunctionNode(
  group: THREE.Group,
  mats: WorldMaterials,
  desc: EntityDescriptor,
  config: EngineConfig,
): void {
  const W = desc.corridorWidth ?? config.hallwayWidth;
  const H = config.hallwayHeight;
  const dir = desc.travelDirection as Cardinal;
  const dirVec = CARDINAL_VEC[dir];
  const e = desc.end!;
  const center = vecAdd(e, vecScale(dirVec, W / 2));
  const rect = { x0: center.x - W / 2, x1: center.x + W / 2, z0: center.z - W / 2, z1: center.z + W / 2 };
  floorCeiling(group, mats.floor, rect, 0, true);
  floorCeiling(group, mats.ceiling, rect, H, false);

  const endDways = desc.doorways.filter((dw) => dw.side === 'end');
  const frontDir = dir;
  const leftDir = leftOf(dir);
  const rightDir = rightOf(dir);
  const front = nodeEdgeSeg(center, W, H, frontDir);
  const leftSeg = nodeEdgeSeg(center, W, H, leftDir);
  const rightSeg = nodeEdgeSeg(center, W, H, rightDir);

  for (const dw of endDways) {
    const seg = dw.facing === frontDir ? front : dw.facing === leftDir ? leftSeg : rightSeg;
    seg.openings.push({ center: spanCoord({ x: dw.x, z: dw.z }, seg.constX), width: dw.width, height: dw.height, full: false });
  }
  buildWall(group, mats.wall, front, config.wallThickness);
  buildWall(group, mats.wall, leftSeg, config.wallThickness);
  buildWall(group, mats.wall, rightSeg, config.wallThickness);
}

function buildHallway(
  group: THREE.Group,
  mats: WorldMaterials,
  desc: EntityDescriptor,
  config: EngineConfig,
  opts: BuildOptions,
): void {
  const W = desc.corridorWidth ?? config.hallwayWidth;
  const H = config.hallwayHeight;
  const dir = desc.travelDirection as Cardinal;
  const dirVec = CARDINAL_VEC[dir];
  const leftVec = vecRotateNeg90(dirVec);
  const rightVec = vecRotate90(dirVec);
  const s = desc.start!;
  const e = desc.end!;
  const axis = desc.axis!;
  const length = Math.hypot(e.x - s.x, e.z - s.z);

  const entryDway = desc.doorways.find((dw) => dw.side === 'start');
  const endDways = desc.doorways.filter((dw) => dw.side === 'end');
  const leftDways = desc.doorways.filter((dw) => dw.side === 'left');
  const rightDways = desc.doorways.filter((dw) => dw.side === 'right');

  // Floors / ceilings.
  if (opts.parentDir) {
    const parentDirVec = CARDINAL_VEC[opts.parentDir];
    for (const r of bendChildFloorRects(s, dirVec, parentDirVec, W, length)) {
      floorCeiling(group, mats.floor, r, 0, true);
      floorCeiling(group, mats.ceiling, r, H, false);
    }
  } else {
    const rect = corridorRect(s, e, axis, W);
    floorCeiling(group, mats.floor, rect, 0, true);
    floorCeiling(group, mats.ceiling, rect, H, false);
  }

  // Side walls with side-doorway openings.
  const leftOpen: Opening[] = leftDways.map((dw) => ({
    center: spanCoord({ x: dw.x, z: dw.z }, leftVec.x !== 0),
    width: dw.width,
    height: dw.height,
    full: false,
  }));
  const rightOpen: Opening[] = rightDways.map((dw) => ({
    center: spanCoord({ x: dw.x, z: dw.z }, rightVec.x !== 0),
    width: dw.width,
    height: dw.height,
    full: false,
  }));

  // Bend parent: full-height corner opening on the turn-side wall.
  if (desc.termination === 'bend' && endDways.length > 0) {
    const bend = endDways[0]!;
    const isLeft = bend.facing === leftOf(dir);
    const constX = isLeft ? leftVec.x !== 0 : rightVec.x !== 0;
    (isLeft ? leftOpen : rightOpen).push({
      center: spanCoord({ x: bend.x, z: bend.z }, constX),
      width: W / 2,
      height: H,
      full: true,
    });
  }
  // Bend child: full-height corner opening on the wall toward the parent.
  if (opts.parentDir && entryDway) {
    const negParent = vecScale(CARDINAL_VEC[opts.parentDir], -1);
    const isLeft = vecEq(negParent, leftVec);
    const constX = isLeft ? leftVec.x !== 0 : rightVec.x !== 0;
    (isLeft ? leftOpen : rightOpen).push({
      center: spanCoord(vecAdd(s, vecScale(dirVec, W / 4)), constX),
      width: W / 2,
      height: H,
      full: true,
    });
  }

  buildWall(group, mats.wall, sideWallSeg(s, e, leftVec, W, H, leftOpen), config.wallThickness);
  buildWall(group, mats.wall, sideWallSeg(s, e, rightVec, W, H, rightOpen), config.wallThickness);

  // Start wall (skipped when coplanar with a parent wall).
  if (!opts.skipEntryWall) {
    const openings: Opening[] = [];
    if (entryDway) {
      const full = opts.parentDir !== undefined;
      openings.push({
        center: spanCoord({ x: entryDway.x, z: entryDway.z }, dirVec.x !== 0),
        width: full ? W / 2 : entryDway.width,
        height: full ? H : entryDway.height,
        full,
      });
    }
    buildWall(group, mats.wall, crossWallSeg(s, vecScale(dirVec, -1), W, H, openings), config.wallThickness);
  }

  // Termination.
  if (desc.termination === 'room') {
    const endDway = endDways[0]!;
    buildWall(
      group,
      mats.wall,
      crossWallSeg(e, dirVec, W, H, [
        { center: spanCoord({ x: endDway.x, z: endDway.z }, dirVec.x !== 0), width: endDway.width, height: endDway.height, full: false },
      ]),
      config.wallThickness,
    );
  } else if (desc.termination === 'bend') {
    const bend = endDways[0]!;
    const turnVec = CARDINAL_VEC[bend.facing];
    const constX = dirVec.x !== 0;
    const p = spanCoord(e, constX);
    const turnSpan = constX ? turnVec.z : turnVec.x;
    const spanMin = turnSpan > 0 ? p - W / 2 : p;
    const spanMax = turnSpan > 0 ? p : p + W / 2;
    buildWall(group, mats.wall, {
      constX,
      fixed: wallCoord(e, constX),
      spanMin,
      spanMax,
      height: H,
      normal: { x: -dirVec.x, z: -dirVec.z },
      openings: [],
    }, config.wallThickness);
  } else {
    buildJunctionNode(group, mats, desc, config);
  }
}

/** Build the three.js Group for a descriptor (created once at generation). */
export function buildEntityGroup(
  descriptor: EntityDescriptor,
  config: EngineConfig,
  opts: BuildOptions = {},
): THREE.Group {
  const group = new THREE.Group();
  const mats = createMaterials();
  if (descriptor.kind === 'room') {
    buildRoom(group, mats, descriptor, config);
  } else {
    buildHallway(group, mats, descriptor, config, opts);
  }
  return group;
}
