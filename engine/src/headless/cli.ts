/**
 * HEADLESS GENERATION CLI (the `npm run generate` entry)
 * ======================================================
 * Two modes:
 *   --make-plan <name>   build a plan JSON file and exit (no generation).
 *   --plan <path>        load a plan JSON file and run it (resumable).
 * With neither, an implicit single-split plan is built from the legacy
 * --episodes/--seed/--frames/--output flags (backwards compatible).
 *
 * Orchestration: build dist/ if needed, start the static server, compute
 * provenance, run the plan through the worker pool, print a summary, close the
 * server. Exits non-zero on total failure or a high abort share (4.11).
 */
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CONFIG } from '../../config/config';
import { randomSeed } from '../core/rng';
import { serveDir } from './staticServer';
import { runPlan } from './workerPool';
import { needsRebuild } from './stale';
import { generatePlan, loadPlan, savePlan, type RunPlan } from './plan';
import { configHash, dirtyFlag, engineCommit, engineSourceHash } from './provenance';
import type { ProfileName } from './bot';

/** Engine repo root, resolved from this file so the CLI is cwd-independent. */
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const DIST_DIR = join(ROOT, 'dist');
const DEFAULT_SERVE_PORT = 4587;
const DEFAULT_PLAN_NAME = 'pilot';

interface CliArgs {
  concurrency: number;
  episodes: number;
  frames: number;
  seed: number | null;
  output: string;
  servePort: number;
  build: boolean;
  help: boolean;
  makePlan: string | null;
  plan: string | null;
  name: string;
  train: number;
  val: number;
  test: number;
  profile: ProfileName;
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
    makePlan: null,
    plan: null,
    name: DEFAULT_PLAN_NAME,
    train: 0,
    val: 0,
    test: 0,
    profile: 'mixture',
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
      case '--make-plan':
        args.makePlan = next();
        break;
      case '--plan':
        args.plan = resolve(next());
        break;
      case '--name':
        args.name = next();
        break;
      case '--train':
        args.train = parseIntArg(next(), flag);
        break;
      case '--val':
        args.val = parseIntArg(next(), flag);
        break;
      case '--test':
        args.test = parseIntArg(next(), flag);
        break;
      case '--profile':
        args.profile = next() as ProfileName;
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
      '  Plan mode (recommended):',
      '    --make-plan <name>   write a plan JSON at <output>/<name>.plan.json and exit',
      '    --plan <path>        load a plan JSON and run it (resumable)',
      '    --train/--val/--test <n>  per-split episode counts (for --make-plan)',
      `    --profile <name>     mixture profile: ${Object.keys(CONFIG.mixtureProfiles).join(' | ')}`,
      '    --name <name>        plan name (default: pilot)',
      '',
      '  Legacy single-split mode (no --make-plan/--plan):',
      '    --episodes <n>       number of episodes (train split) (default: 1)',
      '    --seed <n>           base RNG seed (default: random)',
      '',
      '  Common:',
      '    --frames <n>         simulated frames per episode (default: config.episodeFrames)',
      '    --concurrency <n>    simultaneous headless browsers (default: config.workerConcurrency)',
      '    --output <dir>       output root (default: <repo>/data)',
      '    --serve-port <n>     static server port (default: 4587)',
      '    --build              force a vite build even if dist/ exists',
      '    --help               show this help',
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

function validateProfile(profile: string): profile is ProfileName {
  return profile in CONFIG.mixtureProfiles;
}

/** Build the implicit (legacy) plan from --episodes/--seed/--frames/--output. */
function implicitPlan(args: CliArgs): RunPlan {
  const baseSeed = args.seed ?? randomSeed();
  return generatePlan({
    name: args.name,
    profile: args.profile,
    framesPerEpisode: args.frames,
    outputRoot: args.output,
    train: args.episodes,
    val: 0,
    test: 0,
    trainSeedBase: baseSeed,
  });
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }
  if (!validateProfile(args.profile)) {
    console.error(`[cli] unknown profile "${args.profile}" (expected one of ${Object.keys(CONFIG.mixtureProfiles).join(', ')})`);
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

  // --- --make-plan: write a plan JSON and exit (no generation). ---
  if (args.makePlan) {
    const train = args.train > 0 || args.val > 0 || args.test > 0 ? args.train : args.episodes;
    const plan = generatePlan({
      name: args.makePlan,
      profile: args.profile,
      framesPerEpisode: args.frames,
      outputRoot: args.output,
      train,
      val: args.val,
      test: args.test,
    });
    const planPath = join(args.output, `${args.makePlan}.plan.json`);
    await savePlan(plan, planPath);
    console.log(`[cli] wrote plan ${planPath} (${plan.episodes.length} episodes: train ${train}, val ${args.val}, test ${args.test})`);
    return;
  }

  // --- Resolve the plan (explicit file, or implicit legacy). ---
  let plan: RunPlan;
  if (args.plan) {
    plan = await loadPlan(args.plan);
    console.log(`[cli] loaded plan ${args.plan} (${plan.episodes.length} episodes)`);
  } else {
    plan = implicitPlan(args);
    console.log(`[cli] implicit plan "${plan.name}": ${plan.episodes.length} train episode(s) x ${args.frames} frames`);
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
    console.log(`[cli] serving ${DIST_DIR} at ${server.url}`);

    const provenance = {
      configHash: configHash(CONFIG),
      engineCommit: engineCommit(ROOT),
      dirty: dirtyFlag(ROOT),
      engineSourceHash: engineSourceHash(ROOT),
    };
    console.log(
      `[cli] provenance: config=${provenance.configHash.slice(0, 8)}… commit=${provenance.engineCommit.slice(0, 8)} dirty=${provenance.dirty}`,
    );

    const summary = await runPlan({
      serverUrl: server.url,
      plan,
      concurrency: args.concurrency,
      config: CONFIG,
      provenance,
    });

    const seconds = ((Date.now() - startedAt) / 1000).toFixed(2);
    console.log('\n=== SUMMARY ===');
    console.log(
      `complete=${summary.complete} aborted=${summary.aborted} failed=${summary.failed} ` +
        `(total ${summary.total}) in ${seconds}s wall-clock`,
    );
    console.log(
      `kept frames=${summary.keptFrames}  slideFrames=${summary.slideFrames}  ` +
        `grazeFrames=${summary.grazeFrames}`,
    );
    if (summary.failed > 0) {
      for (const f of summary.failures) {
        console.log(`  FAILED ${f.episodeId} (seed ${f.seed}): ${f.error}`);
      }
      process.exitCode = 1;
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
