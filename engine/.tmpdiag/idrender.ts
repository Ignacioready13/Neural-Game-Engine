/** Raycast "ID render": color each pixel by the (entity, mesh#) it sees. */
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import * as THREE from 'three';
import * as fs from 'node:fs';

const MANIFEST = '/home/ignac/NeuralGameEngine/engine/data/playtest_2026-10-09T02-53-19-924Z/manifest.json';
const frameNo = Number(process.argv[2] ?? 500);
const outPath = process.argv[3] ?? '/tmp/diag/id500.png';
const highlight = process.argv[4]; // "ent#idx,ent#idx"
const highlights = new Set((highlight ?? '').split(',').filter(Boolean));

const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
const world = new World(manifest.seed, CONFIG, { buildGeometry: true });
world.init();
let target: any = null;
for (const f of manifest.frames) {
  world.maintain(f.currentEntityId);
  if (f.frameIndex === frameNo) { target = f; break; }
}
const f = target;
const W = 320, H = 240;
const aspect = 320/240;
const hfov = CONFIG.fovHorizontalDeg * Math.PI/180;
const vfov = 2 * Math.atan(Math.tan(hfov/2)/aspect);
const cam = new THREE.PerspectiveCamera(vfov*180/Math.PI, aspect, 0.02, 200);
const fx = Math.cos(f.playerHeading), fz = Math.sin(f.playerHeading);
cam.position.set(f.playerX, CONFIG.eyeHeight, f.playerZ);
cam.lookAt(f.playerX+fx, CONFIG.eyeHeight, f.playerZ+fz);
cam.updateMatrixWorld(true);

const meshes: {ent:string, idx:number, mesh:THREE.Mesh}[] = [];
for (const e of world.all()) {
  const g = e.group as THREE.Group;
  g.children.forEach((o, i) => meshes.push({ent: e.descriptor.id, idx: i, mesh: o as THREE.Mesh}));
  g.updateMatrixWorld(true);
}
const ray = new THREE.Raycaster();
ray.near = 0.02; ray.far = 200;
const buf = Buffer.alloc(W*H*3);
const counts: Record<string, number> = {};
const ndc = new THREE.Vector2();
for (let py = 0; py < H; py++) {
  for (let px = 0; px < W; px++) {
    ndc.x = (px + 0.5) / W * 2 - 1;
    ndc.y = -((py + 0.5) / H * 2 - 1);
    ray.setFromCamera(ndc, cam);
    const hits = ray.intersectObjects(meshes.map(m => m.mesh), false);
    const o = (py*W+px)*3;
    if (hits.length === 0) { buf[o]=0; buf[o+1]=0; buf[o+2]=0; continue; }
    const hit = hits[0]!;
    const info = meshes.find(m => m.mesh === hit.object)!;
    const key = `${info.ent}#${info.idx}`;
    counts[key] = (counts[key] ?? 0) + 1;
    const isHl = highlights.has(key);
    // deterministic color from key
    let h = 0; for (const c of key) h = (h*31 + c.charCodeAt(0)) >>> 0;
    const r = isHl ? 255 : 60 + (h % 180);
    const g = isHl ? 0 : 60 + ((h>>8) % 180);
    const b = isHl ? 0 : 60 + ((h>>16) % 180);
    buf[o]=r; buf[o+1]=g; buf[o+2]=b;
  }
}
fs.writeFileSync(outPath, buf);
const top = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
console.log(`frame ${frameNo} visible quads (${top.length}):`);
for (const [k,v] of top) console.log(`  ${k}: ${(v/(W*H)*100).toFixed(2)}%${highlights.has(k)?'  <== HIGHLIGHT':''}`);
