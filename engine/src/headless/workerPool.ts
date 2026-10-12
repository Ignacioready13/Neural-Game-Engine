/**
 * PLAN RUNNER / CONCURRENCY (Stage 2, Section 4.7)
 * ================================================
 * Runs a loaded RunPlan with resume: skips episodes already complete (manifest
 * `complete: true`) or already aborted (recorded in `aborted.jsonl`), deletes
 * and redoes partial ones, and retries infrastructure crashes once. Prints a
 * clear aborted-count + L3-graze summary and warns/stops on a high abort share
 * (Section 4.11).
 */
import { mkdir, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import type { EngineConfig } from '../../config/config';
import type { AbortRecord } from '../core/types';
import { runEpisode, type EpisodeResult, type Provenance } from './driver';
import { episodeDirName, episodeIdFor, type PlanEpisode, type RunPlan } from './plan';

export interface PlanRunSpec {
  serverUrl: string;
  plan: RunPlan;
  concurrency: number;
  config: EngineConfig;
  provenance: Provenance;
}

export interface PlanRunSummary {
  total: number;
  complete: number;
  aborted: number;
  failed: number;
  aborts: AbortRecord[];
  failures: Array<{ episodeId: string; seed: number; error: string }>;
  slideFrames: number;
  grazeFrames: number;
  keptFrames: number;
}

function planRoot(plan: RunPlan): string {
  return join(plan.outputRoot, plan.name);
}

async function loadAbortedIds(dir: string): Promise<Set<string>> {
  const set = new Set<string>();
  try {
    const content = await readFile(join(dir, 'aborted.jsonl'), 'utf8');
    for (const line of content.split('\n')) {
      if (!line.trim()) continue;
      try {
        const r = JSON.parse(line) as { episodeId?: string };
        if (r.episodeId) set.add(r.episodeId);
      } catch {
        /* ignore malformed line */
      }
    }
  } catch {
    /* no aborted file yet */
  }
  return set;
}

async function isComplete(dir: string): Promise<boolean> {
  try {
    const m = JSON.parse(await readFile(join(dir, 'manifest.json'), 'utf8')) as { complete?: boolean };
    return m.complete === true;
  } catch {
    return false;
  }
}

export async function runPlan(spec: PlanRunSpec): Promise<PlanRunSummary> {
  const { serverUrl, plan, concurrency, config, provenance } = spec;
  const root = planRoot(plan);
  await mkdir(root, { recursive: true });

  const abortedIds = await loadAbortedIds(root);

  // Build the job list: skip episodes already complete or already aborted.
  const jobs: PlanEpisode[] = [];
  let priorComplete = 0;
  for (const ep of plan.episodes) {
    const id = episodeIdFor(plan, ep);
    if (abortedIds.has(id)) continue;
    const dir = join(root, ep.split, episodeDirName(ep));
    if (await isComplete(dir)) {
      priorComplete += 1;
      continue;
    }
    jobs.push(ep);
  }

  const aborts: AbortRecord[] = [];
  const failures: Array<{ episodeId: string; seed: number; error: string }> = [];
  const results: EpisodeResult[] = [];

  // Minimum finished episodes before the abort-share early stop may fire. A
  // small early sample is noisy (a couple of aborts can look like a runaway
  // rate), so the stop needs a real denominator. The per-plan field is an
  // override; the fail-safe thresholds themselves (clipStopShare / the alert)
  // are unchanged. See Stage 4B.
  const clipStopMinFinished = Math.max(1, plan.clipStopMinEpisodes ?? 100);

  // Running counters for the early-stop check (4.11). Updated synchronously in
  // the worker loop; JavaScript's single thread makes this race-free.
  let finished = priorComplete;
  let abortCount = abortedIds.size;
  let stop = false;

  let cursor = 0;
  const workerCount = Math.max(1, Math.min(concurrency, Math.max(jobs.length, 1)));

  async function worker(): Promise<void> {
    while (!stop) {
      const index = cursor;
      cursor += 1;
      if (index >= jobs.length) return;

      const ep = jobs[index]!;
      const id = episodeIdFor(plan, ep);
      const dir = join(root, ep.split, episodeDirName(ep));
      console.log(`[plan] episode ${id} (seed ${ep.seed}, ${ep.split}) -> ${dir}`);

      // Delete any partial dir before (re)running.
      await rm(dir, { recursive: true, force: true });

      let result: EpisodeResult | null = null;
      let lastError: unknown = null;
      for (let attempt = 0; attempt < 2 && result === null; attempt++) {
        try {
          result = await runEpisode({
            serverUrl,
            seed: ep.seed,
            split: ep.split,
            profile: ep.profile,
            outputDir: dir,
            planDir: root,
            episodeFrames: plan.framesPerEpisode,
            config,
            episodeId: id,
            provenance,
          });
        } catch (error) {
          lastError = error;
          await rm(dir, { recursive: true, force: true });
        }
      }

      if (result === null) {
        failures.push({ episodeId: id, seed: ep.seed, error: String(lastError) });
        finished += 1;
        console.error(`[plan] episode ${id} FAILED (after retry):`, lastError);
      } else {
        results.push(result);
        finished += 1;
        if (result.abort) {
          aborts.push(result.abort);
          abortCount += 1;
        }
        console.log(`[plan] episode ${id}: ${result.status} (${result.frameCount} frames kept)`);

        // Early stop: after enough finished episodes, halt if the abort share is too high.
        if (finished >= clipStopMinFinished && abortCount / finished > config.clipStopShare) {
          stop = true;
          console.warn(
            `[plan] abort share ${((abortCount / finished) * 100).toFixed(1)}% exceeds ` +
              `CLIP_STOP_SHARE (${config.clipStopShare * 100}%) — stopping early`,
          );
        }
      }
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  const total = plan.episodes.length;
  const complete = priorComplete + results.filter((r) => r.status === 'complete').length;
  const aborted = abortCount;
  const failed = failures.length;
  const slideFrames = results.reduce((s, r) => s + r.slideFrames, 0);
  const grazeFrames = results.reduce((s, r) => s + r.grazeFrames, 0);
  const keptFrames = results.reduce((s, r) => s + r.frameCount, 0);

  if (finished >= 20) {
    const abortShare = aborted / finished;
    const grazeShare = keptFrames > 0 ? grazeFrames / keptFrames : 0;
    console.log(
      `\n[plan] ABORTED ${aborted}/${finished} finished (${(abortShare * 100).toFixed(1)}%), ` +
        `L3 graze ${(grazeShare * 100).toFixed(2)}% of kept frames`,
    );
    if (abortShare > config.clipAlertShare && abortShare <= config.clipStopShare) {
      console.warn(
        `[plan] WARNING: abort share ${(abortShare * 100).toFixed(1)}% exceeds ` +
          `CLIP_ALERT_SHARE (${config.clipAlertShare * 100}%)`,
      );
    }
  }

  return { total, complete, aborted, failed, aborts, failures, slideFrames, grazeFrames, keptFrames };
}
