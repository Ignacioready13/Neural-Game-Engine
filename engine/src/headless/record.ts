/**
 * PLAYTEST RECORDING CLI (`npm run record`)
 * =========================================
 * Launches a HEADFUL Chromium window pointed at record.html (the same shared
 * Engine, driven by a real mouse). While the human plays, the page buffers
 * every simulated frame (full state + PNG); this CLI drains that buffer over
 * the Puppeteer bridge and writes frames/ + manifest.json to disk. The result
 * is a bot-like, fully analyzable playthrough from a real human — the same
 * shape as the headless training episodes, plus richer debug fields.
 *
 * The human presses Q (or Escape) to finish and save, or just closes the
 * window (the CLI keeps whatever it last drained). Ctrl+C also finishes.
 */
import { spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import type { Page } from 'puppeteer';
import { CONFIG } from '../../config/config';
import { randomSeed } from '../core/rng';
import { serveDir } from './staticServer';
import { writePng } from './manifest';
import { CHROME_PATH, HEADFUL_LAUNCH_ARGS } from './driver';
import { needsRebuild } from './stale';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const DIST_DIR = join(ROOT, 'dist');
const DEFAULT_PORT = 4588;
const POLL_MS = 400;

interface RecordArgs {
  output: string;
  port: number;
  seed: number | null;
  build: boolean;
  help: boolean;
}

interface RawRecord {
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
  clickedX: number | null;
  dataUrl: string;
}

function parseArgs(argv: string[]): RecordArgs {
  const args: RecordArgs = {
    output: resolve(ROOT, 'data', `playtest_${timestamp()}`),
    port: DEFAULT_PORT,
    seed: null,
    build: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]!;
    const next = (): string => {
      i += 1;
      const v = argv[i];
      if (v === undefined) {
        console.error(`[record] missing value for ${flag}`);
        process.exit(1);
      }
      return v;
    };
    switch (flag) {
      case '--output':
        args.output = resolve(next());
        break;
      case '--port':
        args.port = parseIntArg(next(), flag);
        break;
      case '--seed':
        args.seed = parseIntArg(next(), flag);
        break;
      case '--build':
        args.build = true;
        break;
      case '--help':
      case '-h':
        args.help = true;
        break;
      default:
        console.warn(`[record] ignoring unknown argument: ${flag}`);
    }
  }
  return args;
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function parseIntArg(raw: string, flag: string): number {
  const n = Number(raw);
  if (!Number.isInteger(n)) {
    console.error(`[record] invalid integer for ${flag}: "${raw}"`);
    process.exit(1);
  }
  return n;
}

function printUsage(): void {
  console.log(
    [
      'Usage: npm run record -- [options]',
      '  --output <dir>  output directory (default: <repo>/data/playtest_<timestamp>)',
      '  --seed <n>      world RNG seed (default: random, printed on start)',
      '  --port <n>      static server port (default: 4588)',
      '  --build         force a vite build even if dist/ already exists',
      '  --help          show this help',
      '',
      'Play in the opened window. Press Q (or Escape) to finish and save, or',
      'close the window. Output: <dir>/frames/*.png + <dir>/manifest.json.',
    ].join('\n'),
  );
}

function buildDist(): boolean {
  const viteJs = join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  console.log('[record] building dist via vite...');
  const result = spawnSync(process.execPath, [viteJs, 'build'], { cwd: ROOT, stdio: 'inherit' });
  return result.status === 0;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }

  const needsBuild = args.build || needsRebuild(ROOT, 'record.html');
  if (needsBuild && !buildDist()) {
    console.error('[record] vite build failed; aborting');
    process.exitCode = 1;
    return;
  }

  const server = await serveDir(DIST_DIR, args.port);
  const seed = args.seed ?? randomSeed();
  const framesDir = join(args.output, 'frames');
  await mkdir(framesDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: false,
    args: HEADFUL_LAUNCH_ARGS,
    defaultViewport: { width: 1280, height: 720 },
  });

  const frames: Omit<RawRecord, 'dataUrl'>[] = [];
  let closed = false;
  let aborted = false;

  const drain = async (page: Page): Promise<RawRecord[]> => {
    try {
      return await page.evaluate(() => (window.__recordDrain as () => RawRecord[])());
    } catch {
      return [];
    }
  };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });

    await page.evaluateOnNewDocument((s: number) => {
      window.__BACKROOMS_SEED__ = s;
    }, seed);

    page.on('close', () => {
      closed = true;
    });
    browser.on('disconnected', () => {
      closed = true;
    });
    const onSigint = () => {
      aborted = true;
    };
    process.on('SIGINT', onSigint);

    await page.goto(`${server.url}/record.html`, { waitUntil: 'load' });

    // Wait for the recording bridge to exist.
    for (let i = 0; i < 50; i++) {
      const ready = await page
        .evaluate(() => typeof (window as unknown as { __recordDrain?: unknown }).__recordDrain === 'function')
        .catch(() => false);
      if (ready) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    const info = await page.evaluate(() => (window.__recordInfo as () => { seed: number; frameCount: number })());
    console.log(`[record] recording seed ${info.seed} — play in the window, press Q to finish.`);
    console.log(`[record] output -> ${args.output}`);

    // Poll-drain loop until the human finishes or closes the window.
    while (!closed && !aborted) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      let done = false;
      let records: RawRecord[] = [];
      try {
        const res = await page.evaluate(() => ({
          done: (window as unknown as { __recordDone?: boolean }).__recordDone === true,
          recs: (window.__recordDrain as () => RawRecord[])(),
        }));
        done = res.done;
        records = res.recs ?? [];
      } catch {
        if (closed) break;
        continue;
      }
      for (const rec of records) {
        await writePng(join(args.output, rec.imagePath), rec.dataUrl);
        const { dataUrl: _dataUrl, ...meta } = rec;
        frames.push(meta);
      }
      if (done) break;
    }

    // Final drain (covers the frames between the last poll and finish/close).
    if (!closed) {
      const records = await drain(page);
      for (const rec of records) {
        await writePng(join(args.output, rec.imagePath), rec.dataUrl);
        const { dataUrl: _dataUrl, ...meta } = rec;
        frames.push(meta);
      }
    }

    const manifest = {
      kind: 'playtest-record',
      recordedAt: new Date().toISOString(),
      seed,
      outputWidth: CONFIG.outputWidth,
      outputHeight: CONFIG.outputHeight,
      renderWidth: CONFIG.renderWidth,
      renderHeight: CONFIG.renderHeight,
      simTimestep: CONFIG.simTimestep,
      config: {
        walkSpeed: CONFIG.walkSpeed,
        jerkThreshold: CONFIG.jerkThreshold,
        maxTurnRate: CONFIG.maxTurnRate,
        turnRateGain: CONFIG.turnRateGain,
        debounceSeconds: CONFIG.debounceSeconds,
        reachabilityMaxAngleDeg: CONFIG.reachabilityMaxAngleDeg,
        doorwayGateRadius: CONFIG.doorwayGateRadius,
        doorwayWidth: CONFIG.doorwayWidth,
        hallwayWidth: CONFIG.hallwayWidth,
        fovHorizontalDeg: CONFIG.fovHorizontalDeg,
      },
      frames,
    };
    await writeFile(join(args.output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

    console.log(`\n[record] saved ${frames.length} frames -> ${args.output}`);
  } finally {
    await browser.close().catch(() => {});
    await server.close();
  }
}

main().catch((err) => {
  console.error('[record] fatal:', err);
  process.exitCode = 1;
});
