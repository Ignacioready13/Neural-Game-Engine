/**
 * HEADLESS GENERATION CLI (the `npm run generate` entry)
 * ======================================================
 * Arg parsing + orchestration: build dist/ if needed, start the static server,
 * generate seeds, run the worker pool, print a summary, close the server.
 * Exits non-zero on total failure.
 */
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CONFIG } from '../../config/config';
import { randomSeed } from '../core/rng';
import { serveDir } from './staticServer';
import { runPool } from './workerPool';
import { needsRebuild } from './stale';

/** Engine repo root, resolved from this file so the CLI is cwd-independent. */
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const DIST_DIR = join(ROOT, 'dist');
const DEFAULT_SERVE_PORT = 4587;

interface CliArgs {
  concurrency: number;
  episodes: number;
  frames: number;
  seed: number | null;
  output: string;
  servePort: number;
  build: boolean;
  help: boolean;
}

function parseIntArg(raw: string, flag: string): number {
  const n = Number(raw);
  if (!Number.isInteger(n)) {
    console.error(`[cli] invalid integer for ${flag}: "${raw}"`);
    process.exit(1);
  }
  return n;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    concurrency: CONFIG.workerConcurrency,
    episodes: 1,
    frames: CONFIG.episodeFrames,
    seed: null,
    output: resolve(ROOT, CONFIG.dataOutputRoot),
    servePort: DEFAULT_SERVE_PORT,
    build: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]!;
    const next = (): string => {
      i += 1;
      const v = argv[i];
      if (v === undefined) {
        console.error(`[cli] missing value for ${flag}`);
        process.exit(1);
      }
      return v;
    };

    switch (flag) {
      case '--concurrency':
        args.concurrency = parseIntArg(next(), flag);
        break;
      case '--episodes':
        args.episodes = parseIntArg(next(), flag);
        break;
      case '--frames':
        args.frames = parseIntArg(next(), flag);
        break;
      case '--seed':
        args.seed = parseIntArg(next(), flag);
        break;
      case '--output':
        args.output = resolve(next());
        break;
      case '--serve-port':
        args.servePort = parseIntArg(next(), flag);
        break;
      case '--build':
        args.build = true;
        break;
      case '--help':
      case '-h':
        args.help = true;
        break;
      default:
        console.warn(`[cli] ignoring unknown argument: ${flag}`);
        break;
    }
  }

  return args;
}

function printUsage(): void {
  console.log(
    [
      'Usage: npm run generate -- [options]',
      '  --concurrency <n>   simultaneous headless browsers (default: config.workerConcurrency)',
      '  --episodes <n>      number of episodes to generate (default: 1)',
      '  --frames <n>        simulated frames per episode (default: config.episodeFrames)',
      '  --seed <n>          base RNG seed for the episodes (default: random)',
      '  --output <dir>      output root for episodes (default: <repo>/data)',
      '  --serve-port <n>    static server port (default: 4587)',
      '  --build             force a vite build even if dist/ already exists',
      '  --help              show this help',
    ].join('\n'),
  );
}

function buildDist(): boolean {
  // `npx` is broken in this environment (npm 12 vs Node 22 mismatch), so invoke
  // vite's JS entry directly through node instead.
  const viteJs = join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  console.log('[cli] building dist via vite...');
  const result = spawnSync(process.execPath, [viteJs, 'build'], {
    cwd: ROOT,
    stdio: 'inherit',
  });
  return result.status === 0;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }

  if (args.episodes < 1) {
    console.error('[cli] --episodes must be >= 1');
    process.exitCode = 1;
    return;
  }
  if (args.concurrency < 1) {
    console.error('[cli] --concurrency must be >= 1');
    process.exitCode = 1;
    return;
  }
  if (args.frames < 1) {
    console.error('[cli] --frames must be >= 1');
    process.exitCode = 1;
    return;
  }

  const needsBuild = args.build || needsRebuild(ROOT, 'headless.html');
  if (needsBuild && !buildDist()) {
    console.error('[cli] vite build failed; aborting');
    process.exitCode = 1;
    return;
  }

  const server = await serveDir(DIST_DIR, args.servePort);
  const startedAt = Date.now();
  try {
    const baseSeed = args.seed ?? randomSeed();
    const seeds = Array.from({ length: args.episodes }, (_, i) => (baseSeed + i) >>> 0);

    console.log(
      `[cli] generating ${args.episodes} episode(s) x ${args.frames} frames, ` +
        `concurrency ${args.concurrency}, output ${args.output}`,
    );
    console.log(`[cli] serving ${DIST_DIR} at ${server.url}`);

    const results = await runPool({
      serverUrl: server.url,
      seeds,
      outputRoot: args.output,
      episodeFrames: args.frames,
      concurrency: args.concurrency,
      config: CONFIG,
    });

    const seconds = ((Date.now() - startedAt) / 1000).toFixed(2);
    console.log('\n=== SUMMARY ===');
    console.log(`episodes: ${results.length}/${args.episodes} succeeded in ${seconds}s wall-clock`);
    for (const r of results) {
      console.log(`  ${r.episodeId}  frames=${r.frameCount}  dir=${r.dir}`);
    }
  } catch (err) {
    console.error('[cli] headless generation failed:', err);
    process.exitCode = 1;
  } finally {
    await server.close();
  }
}

main().catch((err) => {
  console.error('[cli] fatal:', err);
  process.exitCode = 1;
});
