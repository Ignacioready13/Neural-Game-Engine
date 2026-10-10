import { CONFIG } from '/home/ignac/NeuralGameEngine/engine/config/config';
import { World } from '/home/ignac/NeuralGameEngine/engine/src/world/World';
import { containsPoint } from '/home/ignac/NeuralGameEngine/engine/src/world/layout';
import * as THREE from 'three';
import * as fs from 'node:fs';

const manifest = JSON.parse(fs.readFileSync('/home/ignac/NeuralGameEngine/engine/data/playtest_2026-10-09T02-53-19-924Z/manifest.json','utf8'));
const world = new World(manifest.seed, CONFIG, { buildGeometry: true, });
world.init();

interface Quad { ent: string; idx: number; n: THREE.Vector3; pts: THREE.Vector3[]; }

function quadsOf(id: string): Quad[] {
  const e = world.get(id)!;
  const g = e.group as THREE.Group;
  return g.children.map((obj, idx) => {
    const m = obj as THREE.Mesh;
    const pos = m.geometry.getAttribute('position');
    const nor = m.geometry.getAttribute('normal');
    const pts: THREE.Vector3[] = [];
    for (let k = 0; k < pos.count; k++) pts.push(new THREE.Vector3(pos.getX(k), pos.getY(k), pos.getZ(k)));
    return { ent: id, idx, n: new THREE.Vector3(nor.getX(0), nor.getY(0), nor.getZ(0)), pts };
  });
}

function insideAnyFloor(x: number, z: number): string | null {
  for (const e of world.all()) {
    if (containsPoint(e.descriptor, x, z, CONFIG)) return e.descriptor.id;
  }
  return null;
}

const EPS = 0.02;
const reported = new Set<string>();

function checkQuad(q: Quad, frame: number) {
  if (Math.abs(q.n.y) > 0.5) return; // floor/ceiling
  // quad is axis-aligned vertical rect: fixed coord along normal axis
  const axis = Math.abs(q.n.x) > 0.5 ? 'x' : 'z';
  const fixed = q.pts[0]![axis];
  const ys = q.pts.map(p => p.y);
  const other = axis === 'x' ? 'z' : 'x';
  const ss = q.pts.map(p => p[other]);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s0 = Math.min(...ss), s1 = Math.max(...ss);
  if (y1 - y0 < 1e-6 || s1 - s0 < 1e-6) return;
  // sample at eye height
  const y = Math.max(y0, Math.min(y1, 1.6));
  if (y <= y0 + 1e-6 || y >= y1 - 1e-6) return;
  const smid = (s0 + s1) / 2;
  const p1 = { x: axis === 'x' ? fixed - EPS : smid, z: axis === 'x' ? smid : fixed - EPS };
  const p2 = { x: axis === 'x' ? fixed + EPS : smid, z: axis === 'x' ? smid : fixed + EPS };
  const a = insideAnyFloor(p1.x, p1.z);
  const b = insideAnyFloor(p2.x, p2.z);
  if (a && b) {
    const key = `${q.ent}:${q.idx}`;
    if (!reported.has(key)) {
      reported.add(key);
      console.log(`SPURIOUS-WALL frame=${frame} ent=${q.ent} mesh#${q.idx} n=(${q.n.x},${q.n.y},${q.n.z}) fixed${axis}=${fixed.toFixed(2)} span=[${s0.toFixed(2)},${s1.toFixed(2)}] y=[${y0.toFixed(2)},${y1.toFixed(2)}] floors=${a},${b}`);
    }
  }
}

// duplicate detection per frame: identical planes+normals overlapping
function checkDuplicates(frame: number) {
  const all: Quad[] = [];
  for (const e of world.all()) all.push(...quadsOf(e.descriptor.id));
  for (let i = 0; i < all.length; i++) {
    for (let j = i + 1; j < all.length; j++) {
      const A = all[i]!, B = all[j]!;
      if (A.ent === B.ent) continue;
      if (A.n.dot(B.n) < 0.99) continue;
      const axis = Math.abs(A.n.x) > 0.5 ? 'x' : Math.abs(A.n.z) > 0.5 ? 'z' : 'y';
      if (A.pts[0]![axis] !== B.pts[0]![axis]) continue;
      const other = axis === 'x' ? 'z' : 'x';
      const ys = (q: Quad) => [Math.min(...q.pts.map(p=>p.y)), Math.max(...q.pts.map(p=>p.y))];
      const ss = (q: Quad) => [Math.min(...q.pts.map(p=>p[other])), Math.max(...q.pts.map(p=>p[other]))];
      const ov = (a:[number,number], b:[number,number]) => Math.min(a[1],b[1]) - Math.max(a[0],b[0]);
      const oy = ov(ys(A), ys(B)); const os = ov(ss(A), ss(B));
      if (oy > 0.01 && os > 0.01) {
        const key = `DUP:${A.ent}#${A.idx}|${B.ent}#${B.idx}`;
        if (!reported.has(key)) {
          reported.add(key);
          console.log(`DUPLICATE frame=${frame} ${A.ent}#${A.idx} & ${B.ent}#${B.idx} axis=${axis} fixed=${A.pts[0]![axis].toFixed(2)} overlapSpan=${os.toFixed(2)} overlapY=${oy.toFixed(2)} n=(${A.n.x},${A.n.y},${A.n.z})`);
        }
      }
    }
  }
}

for (let i = 0; i < manifest.frames.length; i++) {
  const f = manifest.frames[i];
  world.maintain(f.currentEntityId);
  const all: Quad[] = [];
  for (const e of world.all()) all.push(...quadsOf(e.descriptor.id));
  for (const q of all) checkQuad(q, f.frameIndex);
  checkDuplicates(f.frameIndex);
}
console.log('done');
