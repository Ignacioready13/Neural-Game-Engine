/**
 * RUN PLANS (Stage 2, Section 4.7)
 * ================================
 * A plan names a dataset, its mixture profile, frames per episode, output root,
 * and an explicit list of episodes (seed + split + profile). Seeds for the three
 * splits come from DISJOINT ranges recorded in the plan. Plans are JSON files so
 * batches are resumable and findable.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ProfileName } from './bot';

export type Split = 'train' | 'val' | 'test';

export interface PlanEpisode {
  seed: number;
  split: Split;
  profile: ProfileName;
}

export interface SeedRange {
  base: number;
  count: number;
}

export interface RunPlan {
  name: string;
  profile: ProfileName;
  framesPerEpisode: number;
  outputRoot: string;
  /** Disjoint seed ranges per split, recorded for reproducibility. */
  seedRanges: Record<Split, SeedRange>;
  episodes: PlanEpisode[];
}

export interface GeneratePlanOptions {
  name: string;
  profile: ProfileName;
  framesPerEpisode: number;
  outputRoot: string;
  train: number;
  val: number;
  test: number;
  /** Disjoint seed bases (defaults keep the three splits far apart). */
  trainSeedBase?: number;
  valSeedBase?: number;
  testSeedBase?: number;
}

/** Build a plan from per-split counts + disjoint seed ranges. */
export function generatePlan(opts: GeneratePlanOptions): RunPlan {
  const trainBase = opts.trainSeedBase ?? 0;
  const valBase = opts.valSeedBase ?? 1_000_000;
  const testBase = opts.testSeedBase ?? 2_000_000;

  const episodes: PlanEpisode[] = [];
  const push = (count: number, base: number, split: Split): void => {
    for (let i = 0; i < count; i++) {
      episodes.push({ seed: (base + i) >>> 0, split, profile: opts.profile });
    }
  };
  push(opts.train, trainBase, 'train');
  push(opts.val, valBase, 'val');
  push(opts.test, testBase, 'test');

  return {
    name: opts.name,
    profile: opts.profile,
    framesPerEpisode: opts.framesPerEpisode,
    outputRoot: opts.outputRoot,
    seedRanges: {
      train: { base: trainBase, count: opts.train },
      val: { base: valBase, count: opts.val },
      test: { base: testBase, count: opts.test },
    },
    episodes,
  };
}

/** Deterministic episode directory name derived from the plan (split + seed). */
export function episodeDirName(episode: PlanEpisode): string {
  return `seed_${episode.seed}`;
}

/** Deterministic episode id: `<plan>-<split>-<seed>`. */
export function episodeIdFor(plan: RunPlan, episode: PlanEpisode): string {
  return `${plan.name}-${episode.split}-${episode.seed}`;
}

export async function savePlan(plan: RunPlan, filePath: string): Promise<void> {
  await mkdir(join(filePath, '..'), { recursive: true });
  await writeFile(filePath, JSON.stringify(plan, null, 2) + '\n');
}

export async function loadPlan(filePath: string): Promise<RunPlan> {
  return JSON.parse(await readFile(filePath, 'utf8')) as RunPlan;
}
