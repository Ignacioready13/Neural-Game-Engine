/**
 * HEADLESS HARNESS ENTRY (Section 9.2)
 * ====================================
 * Builds the `window.__backrooms` global over the shared Engine, driven by the
 * Puppeteer pipeline (src/headless/driver.ts) instead of a real mouse.
 *
 * This is the SAME shared core as main-interactive.ts (Section 9.3): only the
 * input source (bot-fed clicks vs. real mouse) and output sink (captureFrame()
 * -> disk vs. a visible canvas) differ.
 */
import { CONFIG } from '../../config/config';
import { Engine } from '../sim/Engine';
import type { BackroomsApi, BackroomsState, FrameCapture, GpuInfo } from '../headless/api';

let engine: Engine | null = null;
let bootPromise: Promise<void> | null = null;

/** The episode seed the driver injected via window.__BACKROOMS_SEED__. */
function engineSeed(): number {
  return typeof window.__BACKROOMS_SEED__ === 'number' ? window.__BACKROOMS_SEED__ : 1;
}

function requireCanvas(): HTMLCanvasElement {
  const canvas = document.querySelector<HTMLCanvasElement>('#app-canvas');
  if (!canvas) {
    throw new Error('headless harness requires a <canvas id="app-canvas"> element');
  }
  canvas.width = CONFIG.outputWidth;
  canvas.height = CONFIG.outputHeight;
  return canvas;
}

function ensureEngine(): Engine {
  if (!engine) {
    engine = new Engine(CONFIG, engineSeed(), requireCanvas());
  }
  return engine;
}

/** Create the engine + render once at DOMContentLoaded (Section A). Idempotent. */
function boot(): Promise<void> {
  if (!bootPromise) {
    bootPromise = (async () => {
      if (document.readyState === 'loading') {
        await new Promise<void>((resolve) =>
          document.addEventListener('DOMContentLoaded', () => resolve(), { once: true }),
        );
      }
      ensureEngine().render();
    })();
  }
  return bootPromise;
}

/** Read UNMASKED_RENDERER/VENDOR from a canvas's WebGL context. */
function readGpuInfo(canvas: HTMLCanvasElement): GpuInfo {
  const gl = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as
    | WebGLRenderingContext
    | null;
  if (!gl) return { renderer: 'NO WEBGL CONTEXT', vendor: '' };
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  if (ext) {
    return {
      renderer: String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)),
      vendor: String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)),
    };
  }
  return {
    renderer: String(gl.getParameter(gl.RENDERER)),
    vendor: String(gl.getParameter(gl.VENDOR)),
  };
}

/** Queryable BEFORE init: probes a fresh canvas, falling back to the app canvas. */
function gpuInfo(): GpuInfo {
  const probe = document.createElement('canvas');
  const fromProbe = readGpuInfo(probe);
  if (fromProbe.renderer !== 'NO WEBGL CONTEXT') return fromProbe;

  const app = document.querySelector<HTMLCanvasElement>('#app-canvas');
  if (app) {
    const fromApp = readGpuInfo(app);
    if (fromApp.renderer !== 'NO WEBGL CONTEXT') return fromApp;
  }
  return fromProbe;
}

const api: BackroomsApi = {
  async init(): Promise<void> {
    await boot();
  },

  step(clickX: number | null): void {
    const e = ensureEngine();
    if (clickX !== null) e.submitClick(clickX);
    e.step();
  },

  captureFrame(): FrameCapture {
    // Engine.captureFrame() renders the latest state then grabs the PNG, so the
    // driver only needs step() + captureFrame() per frame (no separate render).
    return ensureEngine().captureFrame();
  },

  getState(): BackroomsState {
    return ensureEngine().getState();
  },

  gpuInfo,
};

window.__backrooms = api;

// Create the engine and render once at DOMContentLoaded (Section A).
void boot();
