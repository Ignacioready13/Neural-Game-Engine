/**
 * CLIPPING DIAGNOSIS (scratch, read-only) — Stage 2 follow-up.
 * Replays pilot episodes renderer-free (World + Steering + navigation +
 * collision, mirroring Engine.step), validates the replay against the stored
 * manifest poses, then measures per frame:
 *   - min distance from the camera to the nearest wall/jamb/floor/ceiling
 *     triangle of the active entity (and generated neighbors),
 *   - whether that min distance is below the camera near plane (0.02),
 *   - whether the camera is inside a wall's thickness band (behind the inner face).
 * No engine behavior is changed.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import * as THREE from 'three';
import { CONFIG } from '../../config/config';
import { World } from '../../src/world/World';
import { Steering } from '../../src/control/steering';
import { resolveWallCollision } from '../../src/sim/collision';
import { doorwayApproachPoint, doorwayPhase, doorwayTargetPoint } from '../../src/sim/navigation';
import type { DoorwayPhase } from '../../src/sim/navigation';
import { verticalFovDeg } from '../../src/render/scene';
import { CARDINAL_VEC } from '../../src/core/types';
import type { CandidateDoorway, Doorway, EpisodeManifest, PlayerState, Vec2 } from '../../src/core/types';

const ROOT = '/home/ignac/NeuralGameEngine/engine/data/pilot_v1';
const NEAR = 0.02;
const MAX_EPISODES = Number(process.argv[2] ?? 20);
const FOCUS_SEED = process.argv[3] ?? '0'; // seed to dump detail for
const FOCUS_FRAME = Number(process.argv[4] ?? -1); // frame to dump detail at

/** Point-to-triangle distance (Ericson, Real-Time Collision Detection). */
const _ab = new THREE.Vector3();
const _ac = new THREE.Vector3();
const _ap = new THREE.Vector3();
const _bp = new THREE.Vector3();
const _cp = new THREE.Vector3();
const _e0 = new THREE.Vector3();
const _e1 = new THREE.Vector3();
const _d = new THREE.Vector3();
const _q = new THREE.Vector3();

function pointTriDist(p: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3): number {
  _ab.subVectors(b, a);
  _ac.subVectors(c, a);
  _ap.subVectors(p, a);
  const d1 = _ab.dot(_ap);
  const d2 = _ac.dot(_ap);
  if (d1 <= 0 && d2 <= 0) return _ap.length();

  _bp.subVectors(p, b);
  const d3 = _ab.dot(_bp);
  const d4 = _ac.dot(_bp);
  if (d3 >= 0 && d4 <= d3) return _bp.length();

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return _d.subVectors(p, _ab.multiplyScalar(v).add(a)).length();
  }

  _cp.subVectors(p, c);
  const d5 = _ab.dot(_cp);
  const d6 = _ac.dot(_cp);
  if (d6 >= 0 && d5 <= d6) return _cp.length();

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return _d.subVectors(p, _ac.multiplyScalar(w).add(a)).length();
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    return _d.subVectors(p, _e1.subVectors(c, b).multiplyScalar(w).add(b)).length();
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  _q.copy(_ab).multiplyScalar(v).addScaledVector(_ac, w).add(a);
  return _d.subVectors(p, _q).length();
}

interface Tri {
  a: THREE.Vector3;
  b: THREE.Vector3;
  c: THREE.Vector3;
  normal: THREE.Vector3;
}

const triCache = new Map<string, Tri[]>();

// Near-wall histogram across all scanned frames (filled in runEpisode).
const clauses = { a: 0, b: 0, bOnly: 0, either: 0 };

const buckets = {
  lt002: 0, lt005: 0, lt01: 0, lt025: 0,
  bf005: 0, bf01: 0, bf025: 0, bfAny: 0,
};

function triangles(group: unknown): Tri[] {
  const out: Tri[] = [];
  const g = group as THREE.Object3D;
  g.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!(mesh as unknown as { isMesh?: boolean }).isMesh || !mesh.geometry) return;
    const geo = mesh.geometry as THREE.BufferGeometry;
    const pos = geo.getAttribute('position') as THREE.BufferAttribute | undefined;
    if (!pos) return;
    mesh.updateMatrixWorld(true);
    const idx = geo.getIndex();
    const get = (i: number, t: THREE.Vector3): THREE.Vector3 => {
      t.fromBufferAttribute(pos as THREE.BufferAttribute, i).applyMatrix4(mesh.matrixWorld);
      return t;
    };
    const emit = (i0: number, i1: number, i2: number): void => {
      const a = get(i0, new THREE.Vector3());
      const b = get(i1, new THREE.Vector3());
      const c = get(i2, new THREE.Vector3());
      const normal = new THREE.Vector3().crossVectors(
        new THREE.Vector3().subVectors(b, a),
        new THREE.Vector3().subVectors(c, a),
      );
      if (normal.lengthSq() > 0) normal.normalize();
      out.push({ a, b, c, normal });
    };
    if (idx) {
      for (let i = 0; i < idx.count; i += 3) emit(idx.getX(i), idx.getX(i + 1), idx.getX(i + 2));
    } else {
      for (let i = 0; i < pos.count; i += 3) emit(i, i + 1, i + 2);
    }
  });
  return out;
}

interface EpResult {
  seed: number;
  split: string;
  frames: number;
  maxPoseErr: number;
  unsafeFrames: number[]; // frame indices
  minDist: number;
  minDistFrame: number;
  divergedAt: number; // first frame whose replay pose left the manifest (or -1)
}

async function runEpisode(seed: number, split: string, rel: string, detail: boolean): Promise<EpResult> {
  const manifest = JSON.parse(
    await readFile(join(ROOT, rel, 'manifest.json'), 'utf8'),
  ) as EpisodeManifest;
  triCache.clear(); // entity ids repeat across episodes; geometry does not
  const world = new World(seed, CONFIG);
  const steering = new Steering(CONFIG);
  const aspect = CONFIG.renderWidth / CONFIG.renderHeight;
  const camera = new THREE.PerspectiveCamera(verticalFovDeg(CONFIG.fovHorizontalDeg, aspect), aspect, NEAR, 200);

  const start = world.init();
  let activeId = start.descriptor.id;
  world.maintain(activeId);
  const dir = start.descriptor.travelDirection ? CARDINAL_VEC[start.descriptor.travelDirection] : { x: 1, z: 0 };
  const pt = start.descriptor.start ?? { x: 0, z: 0 };
  const inset = Math.min(1.5, CONFIG.hallwayLengthMin / 2);
  let player: PlayerState = {
    x: pt.x + dir.x * inset,
    z: pt.z + dir.z * inset,
    heading: Math.atan2(dir.z, dir.x),
    angularVelocity: 0,
  };

  let targetDoorwayId: string | null = null;
  let phase: DoorwayPhase = 'approach';
  let currentTargetPoint: Vec2 | null = null;
  let simTime = 0; // Engine accumulates simulatedTime += dt; match it exactly.
  const camPos = new THREE.Vector3();

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

  const n = manifest.frames.length;
  let maxPoseErr = 0;
  let firstDiv = -1;
  const unsafeFrames: number[] = [];
  let minDist = Infinity;
  let minDistFrame = -1;
  const dt = CONFIG.simTimestep;

  for (let i = 0; i < n; i++) {
    // ---- Validate replay pose against the stored manifest ----
    const mf = manifest.frames[i]!;
    const err = Math.max(
      Math.abs(player.x - mf.playerX),
      Math.abs(player.z - mf.playerZ),
      Math.abs(player.heading - mf.playerHeading),
    );
    maxPoseErr = Math.max(maxPoseErr, err);
    if (firstDiv < 0 && err > 0.02) {
      firstDiv = i;
      console.log(
        `    [diverge] seed ${seed} f${i}: replay=(${player.x.toFixed(3)},${player.z.toFixed(3)},h${player.heading.toFixed(3)}) ` +
          `manifest=(${mf.playerX.toFixed(3)},${mf.playerZ.toFixed(3)},h${mf.playerHeading.toFixed(3)}) ` +
          `active=${activeId} mEnt=${mf.currentEntityId} click=${mf.clickX}`,
      );
    }

    // ---- Measure geometry distance to the camera ----
    camPos.set(player.x, CONFIG.eyeHeight, player.z);
    const detailNow =
      detail && FOCUS_FRAME >= 0 && Math.abs(i - FOCUS_FRAME) <= 1;
    const nearby: Array<{ d: number; id: string; c: THREE.Vector3; n: THREE.Vector3 }> = [];
    let d = Infinity;
    let nearest: Tri | null = null;
    for (const id of world.all().map((e) => e.descriptor.id)) {
      const ent = world.get(id);
      if (!ent || !ent.group) continue;
      let tris = triCache.get(id);
      if (!tris) {
        tris = triangles(ent.group);
        triCache.set(id, tris);
      }
      for (const t of tris) {
        const dd = pointTriDist(camPos, t.a, t.b, t.c);
        if (dd < d) {
          d = dd;
          nearest = t;
        }
        if (dd < 1.5) {
          nearby.push({
            d: dd,
            id,
            c: new THREE.Vector3().addVectors(t.a, t.b).add(t.c).multiplyScalar(1 / 3),
            n: t.normal.clone(),
          });
        }
      }
    }
    if (detailNow || d < NEAR) {
      // Screen coverage of geometry closer than the near plane = the see-through
      // hole. Project every triangle within 1.5*near and sum its pixel area.
      camera.position.set(player.x, CONFIG.eyeHeight, player.z);
      const fx2 = Math.cos(player.heading);
      const fz2 = Math.sin(player.heading);
      camera.lookAt(player.x + fx2, CONFIG.eyeHeight, player.z + fz2);
      camera.updateMatrixWorld(true);
      let holePx = 0;
      let clippedNear = 0; // triangles the near plane actually removes
      const PX = (CONFIG.outputWidth / 2) * (CONFIG.outputHeight / 2);

      // Clip a view-space triangle against depth z <= -plane, project, clamp the
      // polygon to the NDC viewport, and return its screen area in pixels.
      const screenArea = (tri: THREE.Vector3[], plane: number): number => {
        let poly: THREE.Vector3[] = [];
        for (let k = 0; k < tri.length; k++) {
          const cur = tri[k]!;
          const nxt = tri[(k + 1) % tri.length]!;
          const ci = cur.z <= -plane;
          const ni = nxt.z <= -plane;
          if (ci) poly.push(cur.clone());
          if (ci !== ni) {
            const tt = (-plane - cur.z) / (nxt.z - cur.z);
            poly.push(new THREE.Vector3().lerpVectors(cur, nxt, tt));
          }
        }
        if (poly.length < 3) return 0;
        let ndc = poly.map((v) => {
          const p = v.clone().applyMatrix4(camera.projectionMatrix);
          return new THREE.Vector3(p.x, p.y, 0);
        });
        // Clamp to the viewport: x in [-1,1], y in [-1,1].
        const planes: Array<{ inside: (p: THREE.Vector3) => number }> = [
          { inside: (p) => p.x + 1 }, // x >= -1
          { inside: (p) => 1 - p.x }, // x <= 1
          { inside: (p) => p.y + 1 }, // y >= -1
          { inside: (p) => 1 - p.y }, // y <= 1
        ];
        for (const pl of planes) {
          const inp = ndc;
          const out: THREE.Vector3[] = [];
          for (let k = 0; k < inp.length; k++) {
            const cur = inp[k]!;
            const nxt = inp[(k + 1) % inp.length]!;
            const dc = pl.inside(cur);
            const dn = pl.inside(nxt);
            if (dc >= 0) out.push(cur.clone());
            if (dc >= 0 !== dn >= 0) {
              const tt = dc / (dc - dn);
              out.push(new THREE.Vector3().lerpVectors(cur, nxt, tt));
            }
          }
          ndc = out;
          if (ndc.length < 3) return 0;
        }
        let a2 = 0;
        for (let k = 0; k < ndc.length; k++) {
          const p = ndc[k]!;
          const q = ndc[(k + 1) % ndc.length]!;
          a2 += p.x * q.y - q.x * p.y;
        }
        return (Math.abs(a2) / 2) * PX;
      };

      for (const id of world.all().map((e) => e.descriptor.id)) {
        const ent = world.get(id);
        if (!ent || !ent.group) continue;
        const tris = triCache.get(id) ?? [];
        for (const t of tris) {
          if (pointTriDist(camPos, t.a, t.b, t.c) > NEAR * 1.5) continue;
          const view = [t.a, t.b, t.c].map((v) => v.clone().applyMatrix4(camera.matrixWorldInverse));
          const zs = view.map((v) => v.z);
          // Near-clipped: the triangle has area in front of the camera but
          // closer than the near plane (zone (-NEAR, 0)).
          const inFront = zs.filter((z) => z < 0).length;
          const beyondNear = zs.filter((z) => z <= -NEAR).length;
          if (inFront > 0 && beyondNear < 3) clippedNear++;
          // "Would-be" area uses a 1 mm virtual near to stay finite.
          holePx += Math.max(0, screenArea(view, 0.001) - screenArea(view, NEAR));
        }
      }
      console.log(`      near-plane hole: ${holePx.toFixed(0)} px of ${CONFIG.outputWidth * CONFIG.outputHeight} (${((holePx / (CONFIG.outputWidth * CONFIG.outputHeight)) * 100).toFixed(1)}%), ${clippedNear} triangle(s) clipped by near`);
      if (d < NEAR) {
        console.log(`UNSAFE\t${seed}\t${i}\t${((holePx / (CONFIG.outputWidth * CONFIG.outputHeight)) * 100).toFixed(1)}\t${d.toFixed(4)}`);
      }
      nearby.sort((p, q) => p.d - q.d);
      console.log(`  --- seed ${seed} f${i} camera=(${player.x.toFixed(3)},${CONFIG.eyeHeight},${player.z.toFixed(3)}) active=${activeId}`);
      for (const nb of nearby.slice(0, 6)) {
        console.log(
          `      ${nb.id} d=${nb.d.toFixed(4)} centroid=(${nb.c.x.toFixed(2)},${nb.c.y.toFixed(2)},${nb.c.z.toFixed(2)}) normal=(${nb.n.x.toFixed(2)},${nb.n.y.toFixed(2)},${nb.n.z.toFixed(2)})`,
        );
      }
      const act = world.get(activeId);
      if (act) {
        for (const dr of act.descriptor.doorways) {
          console.log(
            `      doorway ${dr.id} -> ${dr.targetEntityId} center=(${dr.x.toFixed(2)},${dr.z.toFixed(2)}) w=${dr.width} facing=${dr.facing} side=${dr.side} turn=${!!dr.turn} entry=${!!dr.entry}`,
          );
        }
      }
      if (d < NEAR && nearest) {
        const backFacing = nearest.normal.dot(camPos.clone().sub(nearest.a)) < 0;
        const here = world.entityAt(player.x, player.z);
        const act0 = world.get(activeId);
        let dwDist = Infinity;
        let dwId = '-';
        if (act0) {
          for (const dr of act0.descriptor.doorways) {
            const dd = Math.hypot(player.x - dr.x, player.z - dr.z);
            if (dd < dwDist) { dwDist = dd; dwId = dr.id; }
          }
        }
        const nextEnt = manifest.frames[i + 1]?.currentEntityId ?? null;
        console.log(
          `      CLASSIFY: nearestTri backFacing=${backFacing}; ` +
            `containsActive=${world.containsEntity(activeId, player.x, player.z)}; ` +
            `entityAtPlayer=${here?.id ?? 'none'}; ` +
            `nearestDoorway=${dwId} @ ${dwDist.toFixed(2)}u; nextEntity=${nextEnt} (transition=${nextEnt !== null && nextEnt !== activeId})`,
        );
      }
    }
    if (d < minDist && firstDiv < 0) {
      minDist = d;
      minDistFrame = i;
    }
    if (nearest && firstDiv < 0) {
      const bf = nearest.normal.dot(camPos.clone().sub(nearest.a)) < 0;
      {
        const fwdX = Math.cos(player.heading);
        const fwdZ = Math.sin(player.heading);
        let inFront = false;
        for (const v of [nearest.a, nearest.b, nearest.c]) {
          if ((v.x - camPos.x) * fwdX + (v.z - camPos.z) * fwdZ > 0) { inFront = true; break; }
        }
        const A = d < 0.12;
        const B = bf && inFront;
        if (A) clauses.a++;
        if (B) clauses.b++;
        if (B && !A) clauses.bOnly++;
        if (A || B) clauses.either++;
      }
      if (d < 0.02) buckets.lt002++;
      if (d < 0.05) { buckets.lt005++; if (bf) buckets.bf005++; }
      if (d < 0.1) { buckets.lt01++; if (bf) buckets.bf01++; }
      if (d < 0.25) { buckets.lt025++; if (bf) buckets.bf025++; }
      if (bf) buckets.bfAny++;
    }
    if (d < NEAR && firstDiv < 0) unsafeFrames.push(i);
    if (detail && i === FOCUS_FRAME) {
      const backFacing = nearest ? nearest.normal.dot(camPos.clone().sub(nearest.a)) < 0 : false;
      console.log(
        `    seed ${seed} f${i}: pose=(${player.x.toFixed(3)}, ${player.z.toFixed(3)}) h=${player.heading.toFixed(3)} ` +
          `active=${activeId} nearest=${d.toFixed(4)} backFacing=${backFacing}`,
      );
    }

    // ---- One engine step (mirrors Engine.step) ----
    if (i + 1 >= n) break;
    const clickX = mf.clickX;
    if (clickX !== null) steering.handleClick(clickX, simTime);

    const active = world.get(activeId);
    const candidates: CandidateDoorway[] = [];
    if (active) {
      const reference = steering.lastClickedX ?? 0;
      for (const dr of active.descriptor.doorways) {
        const approach = doorwayApproachPoint(dr);
        const sx = project(approach.x, approach.z, CONFIG.eyeHeight);
        if (sx === null) continue;
        candidates.push({
          doorwayId: dr.id,
          screenX: sx,
          reachable: steering.isReachable(player, approach),
          screenDistance: Math.abs(sx - reference),
        });
      }
    }
    const liveIds = active ? active.descriptor.doorways.map((x) => x.id) : [];
    const targetId = steering.update(candidates, liveIds, simTime);
    if (targetId !== targetDoorwayId) {
      targetDoorwayId = targetId;
      phase = 'approach';
    }
    const targetDoorway: Doorway | null =
      active && targetId !== null ? active.descriptor.doorways.find((x) => x.id === targetId) ?? null : null;
    if (targetDoorway) {
      phase = doorwayPhase(player, targetDoorway, CONFIG, phase);
      currentTargetPoint = doorwayTargetPoint(targetDoorway, phase);
    } else {
      currentTargetPoint = null;
      phase = 'approach';
    }
    const stepped = steering.step(player, currentTargetPoint, dt);
    player = resolveWallCollision(world, player, stepped, () => {});
    if (!world.containsEntity(activeId, player.x, player.z)) {
      const here = world.entityAt(player.x, player.z);
      if (here && here.id !== activeId) {
        activeId = here.id;
        world.maintain(activeId);
      }
    }
    simTime += dt;
  }

  return { seed, split, frames: n, maxPoseErr, unsafeFrames, minDist, minDistFrame, divergedAt: firstDiv };
}

async function main(): Promise<void> {
  const results: EpResult[] = [];
  let done = 0;
  for (const split of ['train', 'val', 'test'] as const) {
    const dirs = await (async () => {
      try {
        const { readdir } = await import('node:fs/promises');
        const e = await readdir(join(ROOT, split), { withFileTypes: true });
        return e.filter((x) => x.isDirectory()).map((x) => x.name);
      } catch {
        return [];
      }
    })();
    for (const seedDir of dirs) {
      if (done >= MAX_EPISODES) break;
      const rel = `${split}/${seedDir}`;
      let manifest: EpisodeManifest;
      try {
        manifest = JSON.parse(await readFile(join(ROOT, rel, 'manifest.json'), 'utf8')) as EpisodeManifest;
      } catch {
        continue;
      }
      const focus = String(manifest.seed) === FOCUS_SEED;
      if (focus) console.log(`  -- detail for seed ${manifest.seed} --`);
      results.push(await runEpisode(manifest.seed, split, rel, focus));
      done++;
    }
    if (done >= MAX_EPISODES) break;
  }

  let totalFrames = 0;
  let unsafeTotal = 0;
  let episodesWithUnsafe = 0;
  let maxErr = 0;
  let globalMin = Infinity;
  let globalMinEp = -1;
  let globalMinFrame = -1;
  const diverged: string[] = [];
  for (const r of results) {
    totalFrames += r.frames;
    unsafeTotal += r.unsafeFrames.length;
    if (r.unsafeFrames.length > 0) episodesWithUnsafe++;
    maxErr = Math.max(maxErr, r.maxPoseErr);
    if (r.divergedAt >= 0) diverged.push(`seed ${r.seed} @ f${r.divergedAt}`);
    if (r.minDist < globalMin) {
      globalMin = r.minDist;
      globalMinEp = r.seed;
      globalMinFrame = r.minDistFrame;
    }
  }
  console.log('\n=== CLIP DIAGNOSIS ===');
  console.log(`episodes: ${results.length}, frames: ${totalFrames}, near plane: ${NEAR}`);
  console.log(`replay max pose error vs manifest: ${maxErr.toExponential(2)} (should be ~0)`);
  console.log(`frames with camera-to-geometry < near (${NEAR}): ${unsafeTotal} (${((unsafeTotal / totalFrames) * 100).toFixed(3)}%)`);
  console.log(`episodes with >=1 such frame: ${episodesWithUnsafe} / ${results.length}`);
  console.log(`global min camera-to-geometry distance: ${globalMin.toFixed(4)} (seed ${globalMinEp} f${globalMinFrame})`);
  console.log(`episodes whose replay DIVERGED from the manifest (excluded from stats): ${diverged.length}${diverged.length ? ' -> ' + diverged.join(', ') : ''}`);
  const pct = (x: number): string => `${x} (${((x / totalFrames) * 100).toFixed(3)}%)`;
  console.log('--- nearest-geometry histogram (all scanned frames) ---');
  console.log(`min dist < 0.02 (near plane): ${pct(buckets.lt002)}`);
  console.log(`min dist < 0.05: ${pct(buckets.lt005)}   of which nearest tri BACK-FACING: ${buckets.bf005}`);
  console.log(`min dist < 0.10: ${pct(buckets.lt01)}   of which nearest tri BACK-FACING: ${buckets.bf01}`);
  console.log(`min dist < 0.25: ${pct(buckets.lt025)}   of which nearest tri BACK-FACING: ${buckets.bf025}`);
  console.log(`any frame whose nearest tri is BACK-FACING: ${pct(buckets.bfAny)}`);
  console.log('--- viewUnsafe predicate at margin 0.10 (near+margin = 0.12) ---');
  console.log(`clause A (min clearance < 0.12):        ${pct(clauses.a)}`);
  console.log(`clause B (nearest tri back-facing & in front): ${pct(clauses.b)}`);
  console.log(`  B but not A:                          ${pct(clauses.bOnly)}`);
  console.log(`A or B (the flag):                      ${pct(clauses.either)}`);
}

main().catch((e) => {
  console.error('[clip] fatal:', e);
  process.exit(1);
});
