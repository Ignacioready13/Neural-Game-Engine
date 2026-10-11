/**
 * STAGE 2, A3b — clipping-definition measurement (scratch).
 * Renderer-free replica of Engine.step (same logic as tests/regression.test.ts),
 * driven by the real placeholder bot, over ~100 episodes. Measures, per frame:
 *   - wallOverlap (L1 hard clip)
 *   - near-clip: distance from the player to the nearest wall the camera FACES
 *   - slide backstop engaged (the collision callback)
 * and reports, for L1 (overlap + near-clip), L2 (sustained slide runs >= N), and
 * L3 (brief grazes < N), the share of frames and episodes each would flag.
 */
import * as THREE from 'three';
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import { Steering } from '../src/control/steering';
import { resolveWallCollision } from '../src/sim/collision';
import { doorwayApproachPoint, doorwayPhase, doorwayTargetPoint } from '../src/sim/navigation';
import type { DoorwayPhase } from '../src/sim/navigation';
import { verticalFovDeg } from '../src/render/scene';
import { CARDINAL_VEC } from '../src/core/types';
import type { CandidateDoorway, Doorway, PlayerState, Vec2 } from '../src/core/types';
import { makePlaceholderBot } from '../src/headless/bot';

const EPISODES = Number(process.argv[2] ?? 100);
const FRAMES = Number(process.argv[3] ?? 600);
const NEAR_CLIP = 0.02;

/** Axis-aligned box bounds of an entity (matching layout.containsPoint). */
function bounds(desc: { kind: string; center?: Vec2; width?: number; depth?: number; axis?: string; start?: Vec2; end?: Vec2; corridorWidth?: number; termination?: string; travelDirection?: string }, cfg = CONFIG): { x0: number; x1: number; z0: number; z1: number } {
  if (desc.kind === 'room') {
    const c = desc.center!;
    return { x0: c.x - desc.width! / 2, x1: c.x + desc.width! / 2, z0: c.z - desc.depth! / 2, z1: c.z + desc.depth! / 2 };
  }
  const axis = desc.axis!;
  const s = desc.start!;
  const e = desc.end!;
  const W = desc.corridorWidth ?? cfg.hallwayWidth;
  const along = (v: Vec2) => (axis === 'x' ? v.x : v.z);
  const perp = (v: Vec2) => (axis === 'x' ? v.z : v.x);
  let a0 = Math.min(along(s), along(e));
  let a1 = Math.max(along(s), along(e));
  if (desc.termination === 'junction' || desc.termination === 'bend') {
    const dirVec = CARDINAL_VEC[desc.travelDirection as 'east' | 'west' | 'north' | 'south'];
    const alongDir = axis === 'x' ? dirVec.x : dirVec.z;
    const reach = desc.termination === 'junction' ? W : W / 2;
    if (alongDir > 0) a1 += reach;
    else a0 -= reach;
  }
  const p0 = perp(s) - W / 2;
  const p1 = perp(s) + W / 2;
  const x0 = axis === 'x' ? a0 : p0;
  const x1 = axis === 'x' ? a1 : p1;
  const z0 = axis === 'z' ? a0 : p0;
  const z1 = axis === 'z' ? a1 : p1;
  return { x0, x1, z0, z1 };
}

/** Min distance to a wall whose OUTWARD normal the camera faces (forward . n > 0). */
function frontWallDist(player: PlayerState, b: { x0: number; x1: number; z0: number; z1: number }): number {
  const fx = Math.cos(player.heading);
  const fz = Math.sin(player.heading);
  let best = Infinity;
  // Each wall's outward normal, and signed distance from player to wall plane.
  const walls: Array<{ n: [number, number]; d: number }> = [
    { n: [-1, 0], d: player.x - b.x0 }, // -x wall
    { n: [1, 0], d: b.x1 - player.x }, // +x wall
    { n: [0, -1], d: player.z - b.z0 }, // -z wall
    { n: [0, 1], d: b.z1 - player.z }, // +z wall
  ];
  for (const w of walls) {
    if (fx * w.n[0] + fz * w.n[1] <= 0) continue; // camera faces away from this wall
    if (w.d < best) best = w.d;
  }
  return best;
}

/** Doorway-aware variant: a front-facing wall is "open" (not a clip) where the
 * camera's lateral position lines up with a doorway opening. Returns the min
 * distance to a front-facing SOLID wall. */
function frontSolidWallDist(player: PlayerState, desc: { x0: number; x1: number; z0: number; z1: number; doorways: Array<{ x: number; z: number; width: number; facing: string }> }): number {
  const fx = Math.cos(player.heading);
  const fz = Math.sin(player.heading);
  let best = Infinity;
  const walls: Array<{ n: [number, number]; d: number; lateral: number; facing: string }> = [
    { n: [-1, 0], d: player.x - desc.x0, lateral: player.z, facing: 'west' },
    { n: [1, 0], d: desc.x1 - player.x, lateral: player.z, facing: 'east' },
    { n: [0, -1], d: player.z - desc.z0, lateral: player.x, facing: 'north' },
    { n: [0, 1], d: desc.z1 - player.z, lateral: player.x, facing: 'south' },
  ];
  for (const w of walls) {
    if (fx * w.n[0] + fz * w.n[1] <= 0) continue; // faces away
    // Is the camera laterally within a doorway opening on this wall?
    let open = false;
    for (const d of desc.doorways) {
      if (d.facing !== w.facing) continue;
      const dLat = w.facing === 'east' || w.facing === 'west' ? d.z : d.x;
      if (Math.abs(w.lateral - dLat) <= d.width / 2 + 0.01) { open = true; break; }
    }
    if (open) continue; // legit crossing through an opening
    if (w.d < best) best = w.d;
  }
  return best;
}

interface Run {
  frames: number;
  episodes: number;
}

const overall = {
  frames: 0,
  episodes: 0,
  slideFrames: 0,
  overlapFrames: 0,
  nearClipFrames: 0, // frontWallDist < NEAR_CLIP + 0.08 (margin)
  nearClipFrames2: 0, // frontWallDist < NEAR_CLIP + 0.03
  solidClipFrames: 0, // doorway-aware front solid wall < NEAR_CLIP + 0.08
  l1Frames: 0, // overlap OR solidClip
  frontDistHist: new Map<number, number>(),
  runLenHist: new Map<number, number>(),
};
const l1Ep = { episodes: 0 };
const l2Ep: Record<number, Run> = { 5: { frames: 0, episodes: 0 }, 10: { frames: 0, episodes: 0 }, 20: { frames: 0, episodes: 0 } };
const l3Ep: Record<number, Run> = { 5: { frames: 0, episodes: 0 }, 10: { frames: 0, episodes: 0 }, 20: { frames: 0, episodes: 0 } };

const bin = (v: number) => Math.floor(v * 100) / 100;

for (let ep = 0; ep < EPISODES; ep++) {
  const seed = 4000 + ep;
  const world = new World(seed, CONFIG, { buildGeometry: false });
  const steering = new Steering(CONFIG);
  const aspect = CONFIG.renderWidth / CONFIG.renderHeight;
  const camera = new THREE.PerspectiveCamera(verticalFovDeg(CONFIG.fovHorizontalDeg, aspect), aspect, 0.02, 200);

  const start = world.init();
  let activeId = start.descriptor.id;
  world.maintain(activeId);
  const dir = start.descriptor.travelDirection ? CARDINAL_VEC[start.descriptor.travelDirection] : { x: 1, z: 0 };
  const pt = start.descriptor.start ?? { x: 0, z: 0 };
  const inset = Math.min(1.5, CONFIG.hallwayLengthMin / 2);
  let player: PlayerState = { x: pt.x + dir.x * inset, z: pt.z + dir.z * inset, heading: Math.atan2(dir.z, dir.x), angularVelocity: 0 };

  let targetDoorwayId: string | null = null;
  let phase: DoorwayPhase = 'approach';
  let currentTargetPoint: Vec2 | null = null;
  const bot = makePlaceholderBot(seed);

  const project = (wx: number, wz: number, wy: number): number | null => {
    camera.position.set(player.x, CONFIG.eyeHeight, player.z);
    const fx = Math.cos(player.heading);
    const fz = Math.sin(player.heading);
    camera.lookAt(player.x + fx, CONFIG.eyeHeight, player.z + fz);
    camera.updateMatrixWorld(true);
    const v = new THREE.Vector3(wx, wy, wz);
    v.applyMatrix4(camera.matrixWorldInverse);
    if (v.z >= -1e-9) return null;
    v.applyMatrix4(camera.projectionMatrix);
    return v.x;
  };

  let slideRun = 0; // current consecutive slide frames
  let epSlideFrames = 0;
  let epOverlap = 0;
  let epNearClip = 0;
  let epSolidClip = 0;
  const runs: number[] = [];

  for (let i = 0; i < FRAMES; i++) {
    const dt = CONFIG.simTimestep;
    const clickX = bot(i);
    if (clickX !== null) steering.handleClick(clickX, i * dt);

    const active = world.get(activeId);
    const candidates: CandidateDoorway[] = [];
    if (active) {
      const reference = steering.lastClickedX ?? 0;
      for (const d of active.descriptor.doorways) {
        const approach = doorwayApproachPoint(d);
        const sx = project(approach.x, approach.z, CONFIG.eyeHeight);
        if (sx === null) continue;
        candidates.push({ doorwayId: d.id, screenX: sx, reachable: steering.isReachable(player, approach), screenDistance: Math.abs(sx - reference) });
      }
    }
    const liveIds = active ? active.descriptor.doorways.map((d) => d.id) : [];
    const targetId = steering.update(candidates, liveIds, i * dt);
    if (targetId !== targetDoorwayId) { targetDoorwayId = targetId; phase = 'approach'; }
    const targetDoorway: Doorway | null = active && targetId !== null ? active.descriptor.doorways.find((d) => d.id === targetId) ?? null : null;
    if (targetDoorway) { phase = doorwayPhase(player, targetDoorway, CONFIG, phase); currentTargetPoint = doorwayTargetPoint(targetDoorway, phase); }
    else { currentTargetPoint = null; phase = 'approach'; }

    let slid = false;
    const stepped = steering.step(player, currentTargetPoint, dt);
    player = resolveWallCollision(world, player, stepped, () => { slid = true; });

    // L1 overlap: player outside all floors after collision.
    const overlap = !world.containsFloorPoint(player.x, player.z);
    // near-clip: distance to the nearest front-facing wall of the entity that
    // actually CONTAINS the player (post-movement, post-switch).
    const here = world.entityAt(player.x, player.z);
    const fwd = here ? frontWallDist(player, bounds(here)) : Infinity;
    const solid = here ? frontSolidWallDist(player, { ...bounds(here), doorways: here.doorways }) : Infinity;
    const nearClip = fwd < NEAR_CLIP + 0.08;
    const nearClipTight = fwd < NEAR_CLIP + 0.03;
    const solidClip = solid < NEAR_CLIP + 0.08;

    overall.frames += 1;
    overall.frontDistHist.set(bin(Math.min(fwd, 2)), (overall.frontDistHist.get(bin(Math.min(fwd, 2))) ?? 0) + 1);
    if (slid) { overall.slideFrames += 1; epSlideFrames += 1; slideRun += 1; }
    else if (slideRun > 0) { runs.push(slideRun); slideRun = 0; }
    if (overlap) { overall.overlapFrames += 1; epOverlap += 1; }
    if (nearClip) { overall.nearClipFrames += 1; epNearClip += 1; }
    if (nearClipTight) overall.nearClipFrames2 += 1;
    if (solidClip) { overall.solidClipFrames += 1; epSolidClip += 1; }
    if (overlap || solidClip) overall.l1Frames += 1;

    // entity tracking (hysteresis)
    if (!world.containsEntity(activeId, player.x, player.z)) {
      const here = world.entityAt(player.x, player.z);
      if (here && here.id !== activeId) { activeId = here.id; world.maintain(activeId); }
    }
  }
  if (slideRun > 0) runs.push(slideRun);
  for (const r of runs) overall.runLenHist.set(r, (overall.runLenHist.get(r) ?? 0) + 1);
  if (epSlideFrames > 0) {
    console.log(`[clip] seed ${seed}: slideFrames=${epSlideFrames} overlap=${epOverlap} solidClip=${epSolidClip} runs=${JSON.stringify(runs)}`);
  }

  overall.episodes += 1;
  if (epOverlap > 0 || epSolidClip > 0) l1Ep.episodes += 1;
  for (const [n, acc] of Object.entries(l2Ep)) {
    const N = Number(n);
    const hasLong = runs.some((r) => r >= N);
    if (hasLong) acc.episodes += 1;
    // frames inside a >=N run
    acc.frames += runs.filter((r) => r >= N).reduce((s, r) => s + r, 0);
  }
  for (const [n, acc] of Object.entries(l3Ep)) {
    const N = Number(n);
    acc.frames += runs.filter((r) => r < N).reduce((s, r) => s + r, 0);
  }
}

const pct = (num: number, den: number) => ((num / den) * 100).toFixed(3) + '%';

console.log(`=== ${overall.episodes} episodes x ${FRAMES} frames = ${overall.frames} frames (placeholder bot) ===`);
console.log(`\nL1 hard clip:`);
console.log(`  wallOverlap frames: ${overall.overlapFrames} (${pct(overall.overlapFrames, overall.frames)})`);
console.log(`  near-clip (< ${NEAR_CLIP}+0.08 = 0.10 front-wall) frames: ${overall.nearClipFrames} (${pct(overall.nearClipFrames, overall.frames)})`);
console.log(`  near-clip tight (< ${NEAR_CLIP}+0.03 = 0.05) frames: ${overall.nearClipFrames2} (${pct(overall.nearClipFrames2, overall.frames)})`);
console.log(`  near-clip doorway-aware SOLID (< 0.10) frames: ${overall.solidClipFrames} (${pct(overall.solidClipFrames, overall.frames)})`);
console.log(`  L1 (overlap OR doorway-aware solid near-clip) frames: ${overall.l1Frames} (${pct(overall.l1Frames, overall.frames)})`);
console.log(`  L1 episodes flagged: ${l1Ep.episodes} (${pct(l1Ep.episodes, overall.episodes)})`);

console.log(`\nL2 sustained slide runs (>= N consecutive frames):`);
for (const n of [5, 10, 20]) {
  const acc = l2Ep[n]!;
  console.log(`  N=${n}: frames ${acc.frames} (${pct(acc.frames, overall.frames)}), episodes ${acc.episodes} (${pct(acc.episodes, overall.episodes)})`);
}
console.log(`\nL3 brief grazes (slide runs < N):`);
for (const n of [5, 10, 20]) {
  const acc = l3Ep[n]!;
  console.log(`  N=${n}: frames ${acc.frames} (${pct(acc.frames, overall.frames)})`);
}
console.log(`\nSlide (backstop engaged) frames total: ${overall.slideFrames} (${pct(overall.slideFrames, overall.frames)})`);
console.log(`Slide run-length histogram (length -> count):`);
const rl = [...overall.runLenHist.entries()].sort((a, b) => a[0] - b[0]);
console.log('  ' + (rl.length ? rl.map(([k, v]) => `${k}:${v}`).join(' ') : '(none recorded)'));
console.log(`\nfront-wall distance histogram (0.01 bins, capped at 2.0):`);
const fh = [...overall.frontDistHist.entries()].sort((a, b) => a[0] - b[0]);
console.log('  ' + fh.map(([k, v]) => `${k.toFixed(2)}:${v}`).join(' '));
