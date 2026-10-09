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

/** One texture repeat spans this many world units (one tile = 1 world unit). */
export const TILE_WORLD_SIZE = 2;

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
  const base: [number, number, number] = [178, 170, 132];
  drawGrid(ctx, base, 0.06, 'rgb(120,114,88)', 140, (c, x0, y0) => {
    // baseboard along the bottom of every cell
    c.fillStyle = shade(base[0], base[1], base[2], 0.62);
    c.fillRect(x0, y0 + TEX / 2 - 18, TEX / 2, 18);
    // chair rail ~1/3 up
    c.fillStyle = shade(base[0], base[1], base[2], 0.82);
    c.fillRect(x0, y0 + Math.floor((TEX / 2) * 0.34), TEX / 2, 5);
  });
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
