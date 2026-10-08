/**
 * HEADLESS EPISODE DRIVER (Sections 9.2 / 10 / 11.1)
 * ==================================================
 * Launches one headless Chromium instance, verifies REAL GPU acceleration is
 * active (failing loudly on any software rasterizer), then drives the shared
 * Engine through the `window.__backrooms` API: per frame, ask the bot for a
 * click, advance exactly one fixed simulated timestep, capture the PNG, and
 * record a FrameRecord. Finally writes manifest.json and closes the browser.
 *
 * IMPORTANT: simulation advances exactly one fixed timestep per frame and is
 * never tied to wall-clock time (Section 7).
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import puppeteer from 'puppeteer';
import type { Page } from 'puppeteer';
import type { EngineConfig } from '../../config/config';
import type { EpisodeManifest, FrameRecord } from '../core/types';
import type { GpuInfo } from './api';
import { makePlaceholderBot } from './bot';
import { writeManifest, writePng } from './manifest';

export interface EpisodeResult {
  episodeId: string;
  frameCount: number;
  dir: string;
}

export interface RunEpisodeOptions {
  serverUrl: string;
  seed: number;
  outputDir: string;
  episodeFrames: number;
  config: EngineConfig;
  /** Caller-provided episode id. The worker pool passes the same uuid it used
   * for the output directory name so they stay consistent. */
  episodeId?: string;
}

/**
 * Software-rasterizer markers (case-insensitive substring). A hardware renderer
 * string such as "ANGLE (NVIDIA, Vulkan 1.4 ... NVIDIA GeForce ...)" contains
 * NONE of these. "mesa" is deliberately NOT a marker: Mesa Intel / Mesa NVIDIA
 * are legitimate hardware drivers, not software. With no GPU flags headless
 * Chrome reports SwiftShader here, which the first marker catches.
 */
const SOFTWARE_RENDERER_MARKERS = [
  'swiftshader',
  'llvmpipe',
  'softpipe',
  'software',
  'basic render driver',
  'swiftshader device',
] as const;

/**
 * Headless GPU launch flags (Section 11.1), tuned for real Linux hardware:
 * - `--use-gl=angle --use-angle=vulkan` routes WebGL through ANGLE-on-Vulkan,
 *   which reaches the NVIDIA dGPU on this machine (`--use-angle=gl` only hit
 *   the Intel iGPU during live probes).
 * - `--enable-gpu-rasterization` / `--ignore-gpu-blocklist` / `--enable-webgl`
 *   force real GPU usage instead of a software fallback.
 * - `--enable-unsafe-swiftshader` is deliberately NOT passed: without it Chrome
 *   refuses the SwiftShader fallback and the self-check below fails loudly.
 * - `--no-sandbox` is required in this containerized environment.
 * - `headless: true` makes Puppeteer emit `--headless=new` (new headless mode).
 */
export const GPU_LAUNCH_ARGS = [
  '--use-gl=angle',
  '--use-angle=vulkan',
  '--enable-gpu-rasterization',
  '--ignore-gpu-blocklist',
  '--enable-webgl',
  '--no-sandbox',
];

/**
 * Hybrid-graphics routing: this is an Intel iGPU + NVIDIA dGPU laptop. These
 * env vars tell Chromium to use the NVIDIA GPU via PRIME render offload. On a
 * non-hybrid machine they are harmless no-ops.
 */
export const GPU_ENV: Record<string, string> = {
  __NV_PRIME_RENDER_OFFLOAD: '1',
  __GLX_VENDOR_LIBRARY_NAME: 'nvidia',
};

/**
 * Launch args for a HEADFUL (interactive, windowed) Chromium — the playtest
 * recorder (`npm run record`). These deliberately differ from `GPU_LAUNCH_ARGS`:
 * the Vulkan/NVIDIA-offload path that accelerates the headless data pipeline has
 * no usable surface in a real X11/Wayland window here, so `--use-angle=vulkan`
 * yields NO WebGL context and a blank window. ANGLE-over-GL reaches the Intel
 * iGPU through Mesa and renders fine for a human playtest (speed is not the
 * point — visibility is). No NVIDIA offload env vars are set, so windowing
 * stays on the default GPU.
 */
export const HEADFUL_LAUNCH_ARGS = [
  '--use-gl=angle',
  '--use-angle=gl',
  '--ignore-gpu-blocklist',
  '--enable-webgl',
  '--no-sandbox',
];

/** System Chrome executable (per build spec / coordinator's probes). */
export const CHROME_PATH = '/usr/bin/google-chrome';

/**
 * GPU self-check (Section 11.1). Evaluate in the page to obtain the WebGL
 * renderer string; PASS only if it indicates real GPU acceleration. FAILS
 * LOUDLY (throws with the actual renderer string) on any software marker.
 */
export async function assertGpuAcceleration(page: Page): Promise<string> {
  const info = await page.evaluate((): GpuInfo => window.__backrooms.gpuInfo());
  const renderer = info.renderer;

  if (!renderer || renderer === 'NO WEBGL CONTEXT') {
    throw new Error(
      'GPU acceleration check FAILED: no WebGL context could be created in the page. ' +
        'This usually means Chrome was denied a real GPU and fell back to software rendering. ' +
        'Check the launch flags and hybrid-GPU env vars in src/headless/driver.ts.',
    );
  }

  const lower = renderer.toLowerCase();
  const hit = SOFTWARE_RENDERER_MARKERS.find((marker) => lower.includes(marker));
  if (hit) {
    throw new Error(
      `GPU acceleration check FAILED: WebGL renderer "${renderer}" matches software-rasterizer ` +
        `marker "${hit}". Refusing to generate data on a software rasterizer (far too slow). ` +
        'Check the GPU launch flags and hybrid-GPU env vars in src/headless/driver.ts.',
    );
  }

  return renderer;
}

export async function runEpisode(opts: RunEpisodeOptions): Promise<EpisodeResult> {
  const { serverUrl, seed, outputDir, episodeFrames, config } = opts;
  const episodeId = opts.episodeId ?? randomUUID();
  const framesDir = join(outputDir, 'frames');
  await mkdir(framesDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: GPU_LAUNCH_ARGS,
    env: { ...process.env, ...GPU_ENV },
    defaultViewport: { width: config.outputWidth, height: config.outputHeight },
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: config.outputWidth, height: config.outputHeight });

    // Inject the seed before any page script runs, so main-headless.ts reads
    // window.__BACKROOMS_SEED__ when it creates the Engine at DOMContentLoaded.
    await page.evaluateOnNewDocument((s: number) => {
      window.__BACKROOMS_SEED__ = s;
    }, seed);

    await page.goto(`${serverUrl}/headless.html`, { waitUntil: 'load' });

    // GPU self-check BEFORE building the renderer / starting the episode loop.
    const renderer = await assertGpuAcceleration(page);
    console.log(`[driver] episode ${episodeId} GPU renderer: ${renderer}`);

    await page.evaluate(() => window.__backrooms.init());

    const bot = makePlaceholderBot(seed);
    const frames: FrameRecord[] = [];

    // Capture-before-step: frame i is the state AFTER i simulated steps (t =
    // i*dt), and its recorded control signal is the value held at that instant
    // (Section 6.1). The following step() then applies any new click to produce
    // frame i+1, so every (image, timestamp, control) triple is self-consistent.
    for (let frameIndex = 0; frameIndex < episodeFrames; frameIndex++) {
      const clickX = bot(frameIndex);

      // Engine.captureFrame() renders the latest state then grabs the PNG.
      const capture = await page.evaluate(() => window.__backrooms.captureFrame());

      // getState() reflects the state just captured (before the step below).
      const state = await page.evaluate(() => window.__backrooms.getState());

      const imagePath = `frames/${String(frameIndex).padStart(6, '0')}.png`;
      await writePng(join(outputDir, imagePath), capture.dataUrl);

      frames.push({
        frameIndex: state.frameIndex,
        simulatedTime: state.simulatedTime,
        imagePath,
        controlSignalX: state.controlSignalX,
        playerX: state.player.x,
        playerZ: state.player.z,
        playerHeading: state.player.heading,
        currentEntityId: state.activeEntityId,
      });

      // Advance exactly one fixed simulated timestep (never wall-clock).
      await page.evaluate((x: number | null) => window.__backrooms.step(x), clickX);
    }

    const manifest: EpisodeManifest = {
      episodeId,
      renderWidth: config.renderWidth,
      renderHeight: config.renderHeight,
      outputWidth: config.outputWidth,
      outputHeight: config.outputHeight,
      simTimestep: config.simTimestep,
      seed,
      config: {
        jerkThreshold: config.jerkThreshold,
        hallwayWidth: config.hallwayWidth,
        hallwayLengthMin: config.hallwayLengthMin,
        hallwayLengthMax: config.hallwayLengthMax,
        roomWidthMin: config.roomWidthMin,
        roomWidthMax: config.roomWidthMax,
        roomDepthMin: config.roomDepthMin,
        roomDepthMax: config.roomDepthMax,
        roomHeightMin: config.roomHeightMin,
        roomHeightMax: config.roomHeightMax,
        doorwayWidth: config.doorwayWidth,
        doorwaySpacingFactor: config.doorwaySpacingFactor,
        debounceSeconds: config.debounceSeconds,
        walkSpeed: config.walkSpeed,
        fovHorizontalDeg: config.fovHorizontalDeg,
      },
      frames,
    };

    await writeManifest(outputDir, manifest);

    return { episodeId, frameCount: frames.length, dir: outputDir };
  } finally {
    await browser.close();
  }
}
