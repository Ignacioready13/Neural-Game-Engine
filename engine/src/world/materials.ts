/**
 * Procedural seamless tiling materials (floor / wall / ceiling), generated
 * locally as canvas textures — no internet fetches, no external assets.
 *
 * The meshes built by geometry.ts sample UVs in world units
 * (uv = world / TILE_WORLD_SIZE), so each texture repeats exactly once per
 * TILE_WORLD_SIZE world units and tiles seamlessly without stretching.
 *
 * Textures are deterministic (seeded per-cell). Materials are lit
 * (MeshStandardMaterial) so the renderer's point/ambient lights illuminate them.
 *
 * In plain-Node environments (rule tests) there is no `document`; the world
 * generator still builds its three.js Group, so materials fall back to flat
 * colors with no canvas textures.
 */
import * as THREE from 'three';
import type { EntityDescriptor } from '../core/types';

/** One texture repeat spans this many world units (one tile = 1 world unit). */
export const TILE_WORLD_SIZE = 2;

/** World units per visible ceiling tile: the ceiling texture draws a 2x2 grid
 * of grout squares per TILE_WORLD_SIZE repeat, so a single square tile is half
 * the repeat. */
export const CEILING_TILE = TILE_WORLD_SIZE / 2;

const TEX = 256;

const HAS_DOM = typeof document !== 'undefined' && typeof document.createElement === 'function';

function makeCanvas(): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas');
  canvas.width = TEX;
  canvas.height = TEX;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  return [canvas, ctx];
}

/** Deterministic per-cell hash -> PRNG-ish float sequence for noise/speckles. */
function cellRng(i: number, j: number): () => number {
  let a = ((i * 73856093) ^ (j * 19349663)) >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shade(r: number, g: number, b: number, f: number): string {
  const rr = Math.round(Math.min(255, Math.max(0, r * f)));
  const gg = Math.round(Math.min(255, Math.max(0, g * f)));
  const bb = Math.round(Math.min(255, Math.max(0, b * f)));
  return `rgb(${rr},${gg},${bb})`;
}

/** Draw a 2x2 grid of tiles that wraps seamlessly (grout at x/z = 0 and TEX/2). */
function drawGrid(
  ctx: CanvasRenderingContext2D,
  base: [number, number, number],
  variation: number,
  grout: string,
  speckles: number,
  accent: (ctx: CanvasRenderingContext2D, x0: number, y0: number, rng: () => number) => void,
): void {
  const cells = 2;
  const cell = TEX / cells;
  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      const rng = cellRng(i, j);
      const f = 1 - variation / 2 + variation * rng();
      ctx.fillStyle = shade(base[0], base[1], base[2], f);
      ctx.fillRect(i * cell, j * cell, cell, cell);
      for (let k = 0; k < speckles; k++) {
        const x = i * cell + rng() * cell;
        const y = j * cell + rng() * cell;
        ctx.fillStyle = shade(base[0], base[1], base[2], f + (rng() - 0.5) * variation);
        ctx.fillRect(x, y, 1.4, 1.4);
      }
      accent(ctx, i * cell, j * cell, rng);
    }
  }
  // Grout lines on interior boundaries (TEX/2) plus the wrapping edges (0 = TEX).
  ctx.fillStyle = grout;
  for (let i = 0; i < cells; i++) {
    const x = i * cell;
    ctx.fillRect(x, 0, 2, TEX);
    const y = i * cell;
    ctx.fillRect(0, y, TEX, 2);
  }
}

function drawFloor(ctx: CanvasRenderingContext2D): void {
  drawGrid(ctx, [58, 57, 52], 0.14, 'rgb(24,24,22)', 320, () => {});
}

function drawWall(ctx: CanvasRenderingContext2D): void {
  // Yellowing wallpaper: a dull warm-yellow base with very faint vertical
  // striping and mottled speckling, plus a dark baseboard at the floor. No grid
  // lines — the texture is seamless by construction (periodic stripes + speckles
  // drawn per wrapped cell).
  const base: [number, number, number] = [196, 186, 130];
  const T = TEX;

  ctx.fillStyle = shade(base[0], base[1], base[2], 1);
  ctx.fillRect(0, 0, T, T);

  // Very subtle vertical wallpaper stripes (periodic, so they tile seamlessly).
  const stripes = 8;
  const sw = T / stripes;
  for (let i = 0; i < stripes; i++) {
    const rng = cellRng(i, 7);
    ctx.fillStyle = shade(base[0], base[1], base[2], 0.965 + 0.06 * rng());
    ctx.fillRect(Math.round(i * sw), 0, Math.ceil(sw) + 1, T);
  }

  // Fine mottling / faint stains, drawn per wrapped cell so it tiles seamlessly.
  const cells = 4;
  const cell = T / cells;
  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      const rng = cellRng(i, j);
      for (let k = 0; k < 26; k++) {
        const x = i * cell + rng() * cell;
        const y = j * cell + rng() * cell;
        ctx.fillStyle = shade(base[0], base[1], base[2], 1 + (rng() - 0.5) * 0.1);
        ctx.fillRect(x, y, 1.8, 1.8);
      }
    }
  }

  // Baseboard along the floor (canvas bottom maps to world y=0 via flipY).
  ctx.fillStyle = shade(base[0], base[1], base[2], 0.56);
  ctx.fillRect(0, T - 26, T, 26);
  ctx.fillStyle = shade(base[0], base[1], base[2], 0.74);
  ctx.fillRect(0, T - 28, T, 2);
}

function drawCeiling(ctx: CanvasRenderingContext2D): void {
  drawGrid(ctx, [205, 203, 194], 0.05, 'rgb(150,148,140)', 500, (c, x0, y0, rng) => {
    c.fillStyle = 'rgb(120,118,112)';
    for (let k = 0; k < 6; k++) {
      c.fillRect(x0 + 8 + rng() * (TEX / 2 - 16), y0 + 8 + rng() * (TEX / 2 - 16), 2, 2);
    }
  });
}

function makeTexture(draw: (ctx: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const [canvas, ctx] = makeCanvas();
  draw(ctx);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  return tex;
}

/**
 * Ceiling-tile grid helpers. The ceiling texture is drawn as a 2x2 grid of
 * grout squares repeated every TILE_WORLD_SIZE world units, so its tile centers
 * sit at CEILING_TILE/2 + k*CEILING_TILE in the *unshifted* texture. To make a
 * hallway's center line (or a room's center) run through a tile center — so the
 * lit tile reads as centered — the ceiling texture is shifted per entity. Both
 * the texture offset (geometry.ts) and the fixture snapping (scene.ts) derive
 * from the same anchor so they can never disagree.
 */

/** Nearest tile center to a world coordinate in the unshifted tile grid. */
export function nearestTileCenter(v: number): number {
  const half = CEILING_TILE / 2;
  return Math.round(v - half) + half;
}

/** UV offset that shifts the ceiling texture so a tile center lands exactly on
 * world coordinate `v` (tile centers then sit at v + k*CEILING_TILE). */
export function tileCenterUvOffset(v: number): number {
  return (nearestTileCenter(v) - v) / TILE_WORLD_SIZE;
}

/** Snap a world coordinate to the nearest tile center of an entity's ceiling
 * grid, given that grid's anchor coordinate (tile centers at anchor + k*CEILING_TILE). */
export function snapToCeilingTile(v: number, anchor: number): number {
  return Math.round((v - anchor) / CEILING_TILE) * CEILING_TILE + anchor;
}

/** Anchor world coordinates whose ±integer offsets are the ceiling-tile centers
 * for one entity. Hallways center a tile column on their center line and keep the
 * along-corridor grid unshifted; rooms center the grid on their footprint center. */
export function ceilingAnchors(desc: EntityDescriptor): { ax: number; az: number } {
  if (desc.kind === 'room') {
    return { ax: desc.center!.x, az: desc.center!.z };
  }
  const half = CEILING_TILE / 2;
  if (desc.axis === 'x') return { ax: half, az: desc.start!.z };
  return { ax: desc.start!.x, az: half };
}

export interface WorldMaterials {
  floor: THREE.MeshStandardMaterial;
  wall: THREE.MeshStandardMaterial;
  ceiling: THREE.MeshStandardMaterial;
}

/** Build the three lit surface materials (shared by every entity group). */
export function createMaterials(): WorldMaterials {
  const floor = new THREE.MeshStandardMaterial({
    roughness: 0.9,
    metalness: 0.0,
    side: THREE.FrontSide,
  });
  const wall = new THREE.MeshStandardMaterial({
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.FrontSide,
  });
  const ceiling = new THREE.MeshStandardMaterial({
    roughness: 0.85,
    metalness: 0.0,
    side: THREE.FrontSide,
  });

  if (HAS_DOM) {
    floor.map = makeTexture(drawFloor);
    wall.map = makeTexture(drawWall);
    ceiling.map = makeTexture(drawCeiling);
  } else {
    floor.color = new THREE.Color(0x3a3934);
    wall.color = new THREE.Color(0xb2aa84);
    ceiling.color = new THREE.Color(0xcdcbc2);
  }
  return { floor, wall, ceiling };
}
