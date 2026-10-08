/**
 * WORKER POOL / CONCURRENCY (Section 11.2)
 * ========================================
 * Runs episodes in batches of `concurrency` simultaneous headless browsers.
 * Each episode writes to its own `outputRoot/episode_<uuid>/` directory.
 * Per-episode failures are logged and the remaining episodes continue; only if
 * EVERY episode fails does runPool throw.
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { EngineConfig } from '../../config/config';
import { runEpisode, type EpisodeResult } from './driver';

export interface PoolSpec {
  serverUrl: string;
  seeds: number[];
  outputRoot: string;
  episodeFrames: number;
  concurrency: number;
  config: EngineConfig;
}

export async function runPool(spec: PoolSpec): Promise<EpisodeResult[]> {
  const { serverUrl, seeds, outputRoot, episodeFrames, concurrency, config } = spec;
  await mkdir(outputRoot, { recursive: true });

  const results: EpisodeResult[] = [];
  const failures: Array<{ seed: number; episodeId: string; error: unknown }> = [];

  // Shared cursor over the seeds; each worker claims the next seed synchronously
  // before awaiting, so every seed is claimed exactly once.
  let cursor = 0;
  const workerCount = Math.max(1, Math.min(concurrency, seeds.length));

  async function worker(): Promise<void> {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= seeds.length) return;

      const seed = seeds[index]!;
      const episodeId = randomUUID();
      const outputDir = join(outputRoot, `episode_${episodeId}`);
      console.log(`[pool] starting episode ${episodeId} (seed ${seed}) -> ${outputDir}`);

      try {
        const result = await runEpisode({
          serverUrl,
          seed,
          outputDir,
          episodeFrames,
          config,
          episodeId,
        });
        results.push(result);
        console.log(`[pool] finished ${episodeId} (seed ${seed}): ${result.frameCount} frames`);
      } catch (error) {
        failures.push({ seed, episodeId, error });
        console.error(`[pool] episode ${episodeId} (seed ${seed}) FAILED:`, error);
      }
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  if (results.length === 0) {
    const first = failures[0];
    throw new Error(
      `All ${seeds.length} episode(s) failed.` +
        (first
          ? ` First failure (${first.episodeId}, seed ${first.seed}): ${String(first.error)}`
          : ''),
    );
  }

  if (failures.length > 0) {
    console.warn(
      `[pool] ${failures.length}/${seeds.length} episode(s) failed (continuing with the rest)`,
    );
  }

  return results;
}
