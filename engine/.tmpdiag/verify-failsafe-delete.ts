import { join } from 'node:path';
import { readdir, rm } from 'node:fs/promises';
import { CONFIG } from '../config/config';
import { serveDir } from '../src/headless/staticServer';
import { runEpisode } from '../src/headless/driver';
import { configHash, engineCommit, dirtyFlag, engineSourceHash } from '../src/headless/provenance';
import type { Bot } from '../src/headless/bot';

const DIST = join(process.cwd(), 'dist');
const ROOT = process.cwd();
const wallDriver: Bot = () => ({ clickX: 1, persona: 'spammer', intentDoorwayId: null });

async function main() {
  const server = await serveDir(DIST, 4600);
  const provenance = { configHash: configHash(CONFIG), engineCommit: engineCommit(ROOT), dirty: dirtyFlag(ROOT), engineSourceHash: engineSourceHash(ROOT) };
  const planDir = '/tmp/backrooms-failsafe-del';
  const outDir = '/tmp/backrooms-failsafe-del/seed_6';
  await rm('/tmp/backrooms-failsafe-del', { recursive: true, force: true });
  // Inject: any clip keeps < 100000 frames -> whole episode deleted.
  const cfg = { ...CONFIG, minKeptFrames: 100000 };
  try {
    const r = await runEpisode({
      serverUrl: server.url, seed: 6, split: 'train', profile: 'mixture',
      outputDir: outDir, planDir, episodeFrames: 400, config: cfg,
      episodeId: 'fail-del-6', provenance, botOverride: wallDriver,
    });
    console.log('status:', r.status, 'frameCount:', r.frameCount, 'level:', r.abort?.level, 'framesKept:', r.abort?.framesKept);
    const entries = await readdir(planDir).catch(() => []);
    console.log('episode dir still exists?', entries.includes('seed_6'));
    console.log('aborted.jsonl lines:', (await import('node:fs')).readFileSync(join(planDir, 'aborted.jsonl'), 'utf8').trim().split('\n').length);
  } finally { await server.close(); }
}
main().catch((e) => { console.error(e); process.exit(1); });
