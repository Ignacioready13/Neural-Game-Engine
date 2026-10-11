/**
 * SCRATCH VERIFICATION (Section 7 check 6): force a wall clip and confirm the
 * clipping failsafe stops, trims, truncates, writes aborted.jsonl, and (for a
 * too-early crash) deletes the whole episode. Not a committed test.
 */
import { join } from 'node:path';
import { CONFIG } from '../config/config';
import { serveDir } from '../src/headless/staticServer';
import { runEpisode } from '../src/headless/driver';
import { configHash, engineCommit, dirtyFlag, engineSourceHash } from '../src/headless/provenance';
import type { Bot } from '../src/headless/bot';

const DIST = join(process.cwd(), 'dist');
const ROOT = process.cwd();

// A bot that always clicks the far-right extreme, steering the player hard into
// the right wall until the slide backstop engages and (after 10 frames) clips.
const wallDriver: Bot = () => ({ clickX: 1, persona: 'spammer', intentDoorwayId: null });

async function main(): Promise<void> {
  const server = await serveDir(DIST, 4599);
  const provenance = {
    configHash: configHash(CONFIG),
    engineCommit: engineCommit(ROOT),
    dirty: dirtyFlag(ROOT),
    engineSourceHash: engineSourceHash(ROOT),
  };
  const planDir = '/tmp/backrooms-failsafe';
  try {
    for (const seed of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
      const r = await runEpisode({
        serverUrl: server.url,
        seed,
        split: 'train',
        profile: 'mixture',
        outputDir: `/tmp/backrooms-failsafe/seed_${seed}`,
        planDir,
        episodeFrames: 400,
        config: CONFIG,
        episodeId: `fail-${seed}`,
        provenance,
        botOverride: wallDriver,
      });
      console.log(
        `seed ${seed}: ${r.status} framesKept=${r.frameCount} ` +
          `level=${r.abort?.level ?? '-'} crash=${r.abort?.crashFrameIndex ?? '-'}`,
      );
    }
  } finally {
    await server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
