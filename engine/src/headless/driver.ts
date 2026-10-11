/**
 * HEADLESS EPISODE DRIVER (Sections 9.2 / 10 / 11.1 + Stage 2 4.1-4.4, 4.11)
 * ==========================================================================
 * Launches one headless Chromium instance, verifies REAL GPU acceleration, then
 * drives the shared Engine through `window.__backrooms`: per frame, read the
 * state, ask the bot for a click, capture the PNG, advance exactly one fixed
 * simulated timestep, and record the frame (training fields + separable debug
 * fields). Implements the clipping failsafe (L1 wallOverlap / L2 sustained slide)
 * that trims or deletes bad episodes. Writes a schema-v2 manifest with
 * provenance. Browsers are closed in a finally block.
 */
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import puppeteer from 'puppeteer';
import type { Page } from 'puppeteer';
import type { EngineConfig } from '../../config/config';
import type { AbortRecord, DebugFrame, EpisodeManifest, FrameRecord } from '../core/types';
import type { BackroomsState, GpuInfo } from './api';
import { makeBot, type Bot, type BotState, type ProfileName } from './bot';
import { appendAbortedLine, writeManifest, writePng } from './manifest';

export interface Provenance {
  configHash: string;
  engineCommit: string;
  dirty: boolean;
  engineSourceHash: string;
}

export interface EpisodeResult {
  episodeId: string;
  status: 'complete' | 'aborted';
  /** Frames in the written manifest (0 when the whole dir was deleted). */
  frameCount: number;
  dir: string;
  abort?: AbortRecord;
  /** Kept frames where the slide backstop engaged (for L3 / collision reporting). */
  slideFrames: number;
  /** Kept frames inside slide runs shorter than clipL2RunLength (brief grazes). */
  grazeFrames: number;
}

export interface RunEpisodeOptions {
  serverUrl: string;
  seed: number;
  split: string;
  profile: ProfileName;
  outputDir: string;
  planDir: string;
  episodeFrames: number;
  config: EngineConfig;
  episodeId?: string;
  provenance: Provenance;
  /** Test/verification hook: replace the mixture bot with a scripted one. */
  botOverride?: Bot;
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

/** Convert the bridge's state snapshot into the bot's read-only view. */
function toBotState(state: BackroomsState): BotState {
  return {
    frameIndex: state.frameIndex,
    activeEntityId: state.activeEntityId,
    player: state.player,
    candidates: state.candidates ?? [],
    controlSignalX: state.controlSignalX,
  };
}

/** Count kept frames where the slide engaged, and those inside brief (<N) runs. */
function slideStats(debugFrames: DebugFrame[], n: number): { slideFrames: number; grazeFrames: number } {
  let slideFrames = 0;
  let grazeFrames = 0;
  let run = 0;
  for (const d of debugFrames) {
    if (d.slid) {
      run += 1;
      slideFrames += 1;
    } else {
      if (run > 0 && run < n) grazeFrames += run;
      run = 0;
    }
  }
  if (run > 0 && run < n) grazeFrames += run;
  return { slideFrames, grazeFrames };
}

export async function runEpisode(opts: RunEpisodeOptions): Promise<EpisodeResult> {
  const { serverUrl, seed, split, profile, outputDir, planDir, episodeFrames, config } = opts;
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

    const bot = opts.botOverride ?? makeBot(seed, profile, config);
    const frames: FrameRecord[] = [];
    const debugFrames: DebugFrame[] = [];
    const clickSequence: Array<{ frameIndex: number; clickX: number }> = [];
    const stateDump: Array<{
      frameIndex: number;
      x: number;
      z: number;
      heading: number;
      entityId: string;
      targetDoorwayId: string | null;
    }> = [];

    const trimFrames = Math.round(config.clipTrimSeconds / config.simTimestep);
    let clip: { level: 'L1' | 'L2'; crashFrameIndex: number; persona: string } | null = null;
    let runStart = -1;
    let runLength = 0;

    for (let frameIndex = 0; frameIndex < episodeFrames; frameIndex++) {
      const stateBefore = await page.evaluate(() => window.__backrooms.getState());
      const isLast = frameIndex === episodeFrames - 1;

      // The bot decides the click for the step i -> i+1. The final frame has no
      // following step, so its action/click fields stay null.
      const decision = isLast
        ? { clickX: null as number | null, persona: 'idle' as const, intentDoorwayId: null }
        : bot(frameIndex, toBotState(stateBefore));

      // Engine.captureFrame() renders the latest state then grabs the PNG.
      const capture = await page.evaluate(() => window.__backrooms.captureFrame());

      const imagePath = `frames/${String(frameIndex).padStart(6, '0')}.png`;
      await writePng(join(outputDir, imagePath), capture.dataUrl);

      const frame: FrameRecord = {
        frameIndex: stateBefore.frameIndex,
        simulatedTime: stateBefore.simulatedTime,
        imagePath,
        controlSignalX: stateBefore.controlSignalX,
        actionX: null,
        actionHasClicked: null,
        clickX: null,
        clickAccepted: null,
        playerX: stateBefore.player.x,
        playerZ: stateBefore.player.z,
        playerHeading: stateBefore.player.heading,
        currentEntityId: stateBefore.activeEntityId,
      };
      const debug: DebugFrame = {
        frameIndex: stateBefore.frameIndex,
        playerOmega: stateBefore.player.angularVelocity,
        lastClickedX: stateBefore.lastClickedX ?? null,
        targetDoorwayId: stateBefore.targetDoorwayId ?? null,
        targetPointX: stateBefore.targetPoint?.x ?? null,
        targetPointZ: stateBefore.targetPoint?.z ?? null,
        targetPhase: stateBefore.targetPhase ?? 'approach',
        collisionEvents: stateBefore.collisionEvents ?? 0,
        wallOverlap: stateBefore.wallOverlap ?? false,
        candidates: stateBefore.candidates ?? [],
        persona: decision.persona,
        intentDoorwayId: decision.intentDoorwayId,
        slid: false,
      };

      if (isLast) {
        frames.push(frame);
        debugFrames.push(debug);
        break;
      }

      // Advance exactly one fixed simulated timestep (never wall-clock).
      await page.evaluate((x: number | null) => window.__backrooms.step(x), decision.clickX);
      const stateAfter = await page.evaluate(() => window.__backrooms.getState());

      // Action-aligned fields (the step from this frame to the next).
      frame.actionX = stateAfter.controlSignalX;
      frame.actionHasClicked = stateAfter.hasAcceptedClick ?? false;
      frame.clickX = decision.clickX;
      frame.clickAccepted = stateAfter.clickAccepted ?? null;
      debug.slid = stateAfter.slidThisStep ?? false;

      if (decision.clickX !== null) clickSequence.push({ frameIndex, clickX: decision.clickX });
      stateDump.push({
        frameIndex: stateAfter.frameIndex,
        x: stateAfter.player.x,
        z: stateAfter.player.z,
        heading: stateAfter.player.heading,
        entityId: stateAfter.activeEntityId,
        targetDoorwayId: stateAfter.targetDoorwayId ?? null,
      });
      if (stateDump.length > 20) stateDump.shift();

      frames.push(frame);
      debugFrames.push(debug);

      // Clipping failsafe (4.11). The offending state is frame index i+1.
      const crashFrameIndex = stateAfter.frameIndex;
      if (stateAfter.wallOverlap) {
        clip = { level: 'L1', crashFrameIndex, persona: decision.persona };
        break;
      }
      if (stateAfter.slidThisStep) {
        if (runLength === 0) runStart = crashFrameIndex;
        runLength += 1;
        if (runLength >= config.clipL2RunLength) {
          clip = { level: 'L2', crashFrameIndex: runStart, persona: decision.persona };
          break;
        }
      } else {
        runStart = -1;
        runLength = 0;
      }
    }

    // --- Clipping failsafe: trim (or delete) the episode. ---
    if (clip) {
      const keepCount = clip.crashFrameIndex - trimFrames;
      const reason =
        clip.level === 'L1'
          ? 'wallOverlap'
          : `slide backstop >= ${config.clipL2RunLength} consecutive frames`;

      if (keepCount < config.minKeptFrames) {
        // Too little remains: delete the whole episode directory.
        await rm(outputDir, { recursive: true, force: true });
        const abort: AbortRecord = {
          episodeId,
          seed,
          split,
          persona: clip.persona,
          level: clip.level,
          reason,
          crashFrameIndex: clip.crashFrameIndex,
          framesKept: 0,
          clickSequence,
          stateDump,
        };
        await appendAbortedLine(planDir, abort);
        return { episodeId, status: 'aborted', frameCount: 0, dir: outputDir, abort, slideFrames: 0, grazeFrames: 0 };
      }

      // Trim: physically delete the trimmed PNGs, truncate records, and null out
      // the action/click fields of the new final frame.
      const trimmed = frames.length - keepCount;
      for (let i = keepCount; i < frames.length; i++) {
        await rm(join(outputDir, frames[i]!.imagePath), { force: true });
      }
      frames.length = keepCount;
      debugFrames.length = keepCount;
      const last = frames[frames.length - 1]!;
      last.actionX = null;
      last.actionHasClicked = null;
      last.clickX = null;
      last.clickAccepted = null;

      const manifest = buildManifest(opts, episodeId, config, frames, debugFrames, {
        truncated: true,
        truncationReason: `${clip.level}: ${reason}`,
        crashFrameIndex: clip.crashFrameIndex,
        trimmedFrames: trimmed,
      });
      await writeManifest(outputDir, manifest);

      const abort: AbortRecord = {
        episodeId,
        seed,
        split,
        persona: clip.persona,
        level: clip.level,
        reason,
        crashFrameIndex: clip.crashFrameIndex,
        framesKept: keepCount,
        clickSequence,
        stateDump,
      };
      await appendAbortedLine(planDir, abort);
      const st = slideStats(debugFrames, config.clipL2RunLength);
      return { episodeId, status: 'aborted', frameCount: keepCount, dir: outputDir, abort, ...st };
    }

    // --- Clean completion. ---
    const manifest = buildManifest(opts, episodeId, config, frames, debugFrames, {
      truncated: false,
      truncationReason: null,
      crashFrameIndex: null,
      trimmedFrames: 0,
    });
    await writeManifest(outputDir, manifest);

    const st = slideStats(debugFrames, config.clipL2RunLength);
    return { episodeId, status: 'complete', frameCount: frames.length, dir: outputDir, ...st };
  } finally {
    await browser.close();
  }
}

function buildManifest(
  opts: RunEpisodeOptions,
  episodeId: string,
  config: EngineConfig,
  frames: FrameRecord[],
  debugFrames: DebugFrame[],
  clip: { truncated: boolean; truncationReason: string | null; crashFrameIndex: number | null; trimmedFrames: number },
): EpisodeManifest {
  return {
    schemaVersion: 2,
    episodeId,
    seed: opts.seed,
    split: opts.split,
    profile: opts.profile,
    renderWidth: config.renderWidth,
    renderHeight: config.renderHeight,
    outputWidth: config.outputWidth,
    outputHeight: config.outputHeight,
    simTimestep: config.simTimestep,
    configHash: opts.provenance.configHash,
    engineCommit: opts.provenance.engineCommit,
    dirty: opts.provenance.dirty,
    engineSourceHash: opts.provenance.engineSourceHash,
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
    debugFrames,
    truncated: clip.truncated,
    truncationReason: clip.truncationReason,
    crashFrameIndex: clip.crashFrameIndex,
    trimmedFrames: clip.trimmedFrames,
    complete: true,
  };
}
