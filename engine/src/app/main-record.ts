/**
 * PLAYTEST RECORDING MODE
 * =======================
 * Same shared Engine + fixed-timestep loop as human-interactive mode, but every
 * simulated frame is recorded (full state + PNG) into an in-page buffer that the
 * Node CLI (src/headless/record.ts) drains over the Puppeteer bridge and writes
 * to disk. This produces a bot-like, fully analyzable playthrough from a REAL
 * human mouse — exactly what the bot data pipeline produces, plus richer debug
 * fields (target doorway, approach phase, candidates, collision events, overlap).
 *
 * Input is the real mouse (same as main-interactive.ts). Pressing Q or Escape
 * sets `window.__recordDone` so the CLI can finish and save cleanly; closing the
 * window also works (the CLI drains whatever it last polled).
 */
import { CONFIG } from '../../config/config';
import { Engine } from '../sim/Engine';
import { clamp } from '../core/math';

const canvas = document.querySelector<HTMLCanvasElement>('#app-canvas')!;
canvas.width = CONFIG.outputWidth;
canvas.height = CONFIG.outputHeight;

function engineSeed(): number {
  return typeof window.__BACKROOMS_SEED__ === 'number' ? window.__BACKROOMS_SEED__ : 1;
}

const engine = new Engine(CONFIG, engineSeed(), canvas);

/** Normalize a client X to screen space -1..1 using the canvas bounding rect. */
function clientXToScreen(clientX: number): number {
  const rect = canvas.getBoundingClientRect();
  const t = (clientX - rect.left) / Math.max(1, rect.width);
  return clamp(t * 2 - 1, -1, 1);
}

// --- click input (recorded per frame) --------------------------------------
let pendingClickX: number | null = null;

canvas.addEventListener('click', (ev: MouseEvent) => {
  pendingClickX = clientXToScreen(ev.clientX);
});

// --- recording buffer ------------------------------------------------------
interface RecordFrame {
  frameIndex: number;
  simulatedTime: number;
  imagePath: string;
  controlSignalX: number;
  playerX: number;
  playerZ: number;
  playerHeading: number;
  playerOmega: number;
  currentEntityId: string;
  lastClickedX: number | null;
  targetDoorwayId: string | null;
  targetPointX: number | null;
  targetPointZ: number | null;
  targetPhase: string;
  collisionEvents: number;
  wallOverlap: boolean;
  candidates: Array<{ doorwayId: string; screenX: number; reachable: boolean; screenDistance: number }>;
  /** The NEW click applied this frame (null when none). */
  clickedX: number | null;
  dataUrl: string;
}

const buffer: RecordFrame[] = [];

// --- fixed-timestep loop (same pattern as main-interactive.ts) --------------
const dt = CONFIG.simTimestep;
let last = performance.now();
let accumulator = 0;

function stepAndRecord(): void {
  const clickX = pendingClickX;
  if (clickX !== null) engine.submitClick(clickX);
  engine.step();

  const state = engine.getState();
  const capture = engine.captureFrame();
  buffer.push({
    frameIndex: state.frameIndex,
    simulatedTime: state.simulatedTime,
    imagePath: `frames/${String(state.frameIndex).padStart(6, '0')}.png`,
    controlSignalX: state.controlSignalX,
    playerX: state.player.x,
    playerZ: state.player.z,
    playerHeading: state.player.heading,
    playerOmega: state.player.angularVelocity,
    currentEntityId: state.activeEntityId,
    lastClickedX: state.lastClickedX ?? null,
    targetDoorwayId: state.targetDoorwayId ?? null,
    targetPointX: state.targetPoint?.x ?? null,
    targetPointZ: state.targetPoint?.z ?? null,
    targetPhase: state.targetPhase ?? 'approach',
    collisionEvents: state.collisionEvents ?? 0,
    wallOverlap: state.wallOverlap ?? false,
    candidates: state.candidates ?? [],
    clickedX: clickX,
    dataUrl: capture.dataUrl,
  });
  pendingClickX = null;
}

function frame(now: number): void {
  accumulator += (now - last) / 1000;
  last = now;
  if (accumulator > 0.25) accumulator = 0.25;
  while (accumulator >= dt) {
    stepAndRecord();
    accumulator -= dt;
  }
  requestAnimationFrame(frame);
}

// --- bridge for the Node CLI ------------------------------------------------
declare global {
  interface Window {
    __BACKROOMS_SEED__?: number;
    __recordDone?: boolean;
    __recordDrain(): RecordFrame[];
    __recordInfo(): { seed: number; frameCount: number };
  }
}

window.__recordDone = false;
window.__recordDrain = () => buffer.splice(0, buffer.length);
window.__recordInfo = () => ({ seed: engineSeed(), frameCount: engine.getState().frameIndex });

window.addEventListener('keydown', (ev) => {
  if (ev.key === 'q' || ev.key === 'Q' || ev.key === 'Escape') {
    window.__recordDone = true;
  }
});

engine.render();
requestAnimationFrame(frame);
