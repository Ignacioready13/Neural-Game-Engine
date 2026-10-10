import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import * as THREE from 'three';
import * as fs from 'node:fs';
const MANIFEST = '/home/ignac/NeuralGameEngine/engine/data/playtest_2026-10-09T02-53-19-924Z/manifest.json';
const frameNo = Number(process.argv[2]);
const ents = process.argv[3] ? process.argv[3].split(',') : null;
const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
const world = new World(manifest.seed, CONFIG, { buildGeometry: true });
world.init();
for (const f of manifest.frames) { world.maintain(f.currentEntityId); if (f.frameIndex === frameNo) break; }
for (const e of world.all()) {
  if (ents && !ents.includes(e.descriptor.id)) continue;
  const g = e.group as THREE.Group;
  const d: any = e.descriptor;
  console.log(`### ${d.id} ${d.kind} ${d.kind==='hallway'?`axis=${d.axis} s=(${d.start.x.toFixed(2)},${d.start.z.toFixed(2)}) e=(${d.end.x.toFixed(2)},${d.end.z.toFixed(2)}) dir=${d.travelDirection} term=${d.termination}`:`c=(${d.center.x.toFixed(2)},${d.center.z.toFixed(2)}) ${d.width.toFixed(2)}x${d.depth.toFixed(2)} h=${d.ceilingHeight.toFixed(2)}`}`);
  console.log('   doorways: ' + d.doorways.map((w:any)=>`${w.id}[${w.side}/${w.facing}@(${w.x.toFixed(2)},${w.z.toFixed(2)}) w${w.width}]`).join(' '));
  g.children.forEach((obj, i) => {
    const m = obj as THREE.Mesh; const pos = m.geometry.getAttribute('position'); const nor = m.geometry.getAttribute('normal');
    const pts: string[] = [];
    for (let k = 0; k < pos.count; k++) pts.push(`(${pos.getX(k).toFixed(2)},${pos.getY(k).toFixed(2)},${pos.getZ(k).toFixed(2)})`);
    console.log(`  [${i}] n=(${nor.getX(0)},${nor.getY(0)},${nor.getZ(0)}) ${pts.join(' ')}`);
  });
}
