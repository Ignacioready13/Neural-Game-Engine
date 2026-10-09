import { CONFIG } from './config/config';
import { World } from './src/world/World';

function dump(seed: number, steps: number) {
  const world = new World(seed, CONFIG, { buildGeometry: false });
  let active = world.init();
  const rng = (() => { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
  for (let s = 0; s <= steps; s++) {
    world.maintain(active.descriptor.id);
    const all = world.all();
    for (const e of all) {
      const d = e.descriptor;
      if (d.kind === 'room') {
        console.log(`seed=${seed} step=${s} ROOM ${d.id} center=(${d.center!.x.toFixed(2)},${d.center!.z.toFixed(2)}) w=${d.width!.toFixed(2)} d=${d.depth!.toFixed(2)} rect=[${(d.center!.x-d.width!/2).toFixed(2)}..${(d.center!.x+d.width!/2).toFixed(2)}]x[${(d.center!.z-d.depth!/2).toFixed(2)}..${(d.center!.z+d.depth!/2).toFixed(2)}]`);
      } else {
        const W = d.corridorWidth!;
        console.log(`seed=${seed} step=${s} HALL ${d.id} axis=${d.axis} start=(${d.start!.x.toFixed(2)},${d.start!.z.toFixed(2)}) end=(${d.end!.x.toFixed(2)},${d.end!.z.toFixed(2)}) term=${d.termination}`);
      }
    }
    const exits = active.descriptor.doorways.filter((d) => d.targetGenerated && d.targetEntityId !== '');
    if (exits.length === 0) break;
    active = world.get(exits[Math.floor(rng() * exits.length)]!.targetEntityId)!;
    if (!active) break;
  }
}
console.log('===== seed 15851 ====='); dump(15851, 3);
console.log('===== seed 95041 ====='); dump(95041, 34);
