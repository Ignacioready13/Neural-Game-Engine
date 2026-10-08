/**
 * HUMAN INTERACTIVE MODE (Section 9.1)
 * ====================================
 * Fixed-timestep game loop (accumulate real elapsed time, step the simulation
 * in fixed increments, render once after catching up) so the simulation logic
 * is identical to headless mode. Input: the real mouse. Only the click's X is
 * used (Section 6.1).
 */
import { CONFIG } from '../../config/config';
import { Engine } from '../sim/Engine';
import { clamp } from '../core/math';

const canvas = document.querySelector<HTMLCanvasElement>('#app-canvas')!;
canvas.width = CONFIG.outputWidth;
canvas.height = CONFIG.outputHeight;

const engine = new Engine(CONFIG, 1, canvas);

/** Normalize a client X to screen space -1..1 using the canvas bounding rect. */
function clientXToScreen(clientX: number): number {
  const rect = canvas.getBoundingClientRect();
  const t = (clientX - rect.left) / Math.max(1, rect.width);
  return clamp(t * 2 - 1, -1, 1);
}

canvas.addEventListener('click', (ev: MouseEvent) => {
  engine.submitClick(clientXToScreen(ev.clientX));
});

// Fixed-timestep accumulator loop.
const dt = CONFIG.simTimestep;
let last = performance.now();
let accumulator = 0;

function frame(now: number): void {
  accumulator += (now - last) / 1000;
  last = now;

  // Clamp to avoid a spiral-of-death after a long tab switch.
  if (accumulator > 0.25) accumulator = 0.25;

  while (accumulator >= dt) {
    engine.step();
    accumulator -= dt;
  }

  engine.render();
  requestAnimationFrame(frame);
}

engine.render();
requestAnimationFrame(frame);
