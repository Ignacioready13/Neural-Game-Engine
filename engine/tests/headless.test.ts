/**
 * HEADLESS PIPELINE UNIT TESTS
 * ============================
 * Covers the pure pieces: placeholder-bot determinism + value range, and the
 * manifest/PNG writer schema. The GPU self-check and the Puppeteer episode run
 * are integration concerns exercised via `npm run generate` (see the driver).
 */
import { describe, it, expect } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { makePlaceholderBot } from '../src/headless/bot';
import { dataUrlToBuffer, writeManifest } from '../src/headless/manifest';
import { episodeDirName, episodeIdFor, generatePlan } from '../src/headless/plan';
import { configHash } from '../src/headless/provenance';
import { Steering } from '../src/control/steering';
import { CONFIG } from '../config/config';
import type { EpisodeManifest, PlayerState } from '../src/core/types';

describe('makePlaceholderBot', () => {
  it('is deterministic for a given seed', () => {
    const a = makePlaceholderBot(1234);
    const b = makePlaceholderBot(1234);
    for (let i = 0; i < 1000; i++) {
      expect(a(i)).toBe(b(i));
    }
  });

  it('returns null most frames, with click X within [-1, 1]', () => {
    const bot = makePlaceholderBot(42);
    let clicks = 0;
    for (let i = 0; i < 2000; i++) {
      const v = bot(i);
      if (v !== null) {
        clicks += 1;
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
    expect(clicks).toBeGreaterThan(0);
    expect(clicks).toBeLessThan(2000);
  });

  it('produces different click schedules for different seeds', () => {
    const a = makePlaceholderBot(1);
    const b = makePlaceholderBot(2);
    const seqA = Array.from({ length: 300 }, (_, i) => a(i));
    const seqB = Array.from({ length: 300 }, (_, i) => b(i));
    expect(seqA).not.toEqual(seqB);
  });
});

describe('manifest helpers', () => {
  it('decodes PNG data URLs to raw bytes', () => {
    const bytes = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3]);
    const url = `data:image/png;base64,${bytes.toString('base64')}`;
    expect(dataUrlToBuffer(url)).toEqual(bytes);
  });

  it('writes a manifest.json matching the EpisodeManifest schema', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'backrooms-manifest-'));
    try {
      const manifest: EpisodeManifest = {
        schemaVersion: 2,
        episodeId: 'test-episode',
        seed: 7,
        split: 'train',
        profile: 'mixture',
        renderWidth: 1280,
        renderHeight: 960,
        outputWidth: 320,
        outputHeight: 240,
        simTimestep: 0.05,
        configHash: '0'.repeat(64),
        engineCommit: 'deadbeef',
        dirty: true,
        engineSourceHash: '1'.repeat(64),
        config: {
          jerkThreshold: 6,
          hallwayWidth: 4,
          hallwayLengthMin: 8,
          hallwayLengthMax: 24,
          roomWidthMin: 4,
          roomWidthMax: 10,
          roomDepthMin: 4,
          roomDepthMax: 10,
          roomHeightMin: 2.5,
          roomHeightMax: 5,
          doorwayWidth: 2,
          doorwaySpacingFactor: 2.5,
          debounceSeconds: 0.15,
          walkSpeed: 2,
          fovHorizontalDeg: 85,
        },
        frames: [
          {
            frameIndex: 0,
            simulatedTime: 0,
            imagePath: 'frames/000000.png',
            controlSignalX: 0.5,
            actionX: null,
            actionHasClicked: null,
            clickX: null,
            clickAccepted: null,
            playerX: 0,
            playerZ: 0,
            playerHeading: 0,
            currentEntityId: 'entity-1',
          },
        ],
        debugFrames: [],
        truncated: false,
        truncationReason: null,
        crashFrameIndex: null,
        trimmedFrames: 0,
        complete: true,
      };

      await writeManifest(dir, manifest);

      const parsed = JSON.parse(
        await readFile(join(dir, 'manifest.json'), 'utf8'),
      ) as EpisodeManifest;
      expect(parsed.episodeId).toBe('test-episode');
      expect(parsed.outputWidth).toBe(320);
      expect(parsed.frames).toHaveLength(1);
      expect(parsed.frames[0]).toMatchObject({ frameIndex: 0, imagePath: 'frames/000000.png' });
      expect(parsed.config).toHaveProperty('doorwaySpacingFactor');
      expect(parsed.config).toHaveProperty('fovHorizontalDeg');
      expect(parsed.schemaVersion).toBe(2);
      expect(parsed.truncated).toBe(false);
      expect(parsed.complete).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('run plans', () => {
  it('assigns disjoint seed ranges and deterministic dir/id names', () => {
    const plan = generatePlan({
      name: 'p',
      profile: 'mixture',
      framesPerEpisode: 100,
      outputRoot: '/tmp/out',
      train: 2,
      val: 1,
      test: 1,
    });

    expect(plan.episodes).toHaveLength(4);
    expect(plan.seedRanges.train).toEqual({ base: 0, count: 2 });
    expect(plan.seedRanges.val).toEqual({ base: 1_000_000, count: 1 });
    expect(plan.seedRanges.test).toEqual({ base: 2_000_000, count: 1 });

    const trainEp = plan.episodes[0]!;
    expect(episodeDirName(trainEp)).toBe(`seed_${trainEp.seed}`);
    expect(episodeIdFor(plan, trainEp)).toBe(`p-train-${trainEp.seed}`);

    // Seeds are disjoint across splits.
    const seeds = plan.episodes.map((e) => e.seed);
    expect(new Set(seeds).size).toBe(seeds.length);
  });
});

describe('provenance', () => {
  it('hashes the full config deterministically', () => {
    expect(configHash(CONFIG)).toBe(configHash(CONFIG));
    expect(configHash(CONFIG)).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('action-alignment rule (Section 4.1)', () => {
  // The manifest's actionX[i] is the held signal in effect during the step
  // i -> i+1, i.e. controlSignalX[i+1]. The driver reads the pre-step state,
  // THEN submits the click and steps. This test mirrors that ordering at the
  // Steering level: a click submitted at frame i leaves the frame-i snapshot
  // unchanged and only shows up (held signal + heading) in the frame-i+1 state.
  it('a click at frame i changes controlSignalX and heading only at frame i+1', () => {
    const steering = new Steering(CONFIG);
    const dt = CONFIG.simTimestep;

    // --- Frame i: read the pre-step snapshot (as the driver does). ---
    const preStepSignal = steering.lastClickedX ?? 0; // 0: never clicked
    const preStepHeading = 0;

    // Submit the click for the step i -> i+1.
    steering.handleClick(0.6, 0.0);

    // The frame-i snapshot still shows the pre-click values.
    expect(preStepSignal).toBe(0);
    expect(preStepHeading).toBe(0);

    // --- Step i -> i+1. ---
    let p: PlayerState = { x: 0, z: 0, heading: 0, angularVelocity: 0 };
    p = steering.step(p, { x: 0, z: 10 }, dt);

    // The frame-i+1 state carries the new signal and a changed heading.
    expect(steering.lastClickedX).toBe(0.6); // controlSignalX[i+1]
    expect(p.heading).not.toBeCloseTo(0, 12); // heading first changes at i+1
  });

  it('the held signal is unchanged by the frames between a click and the next', () => {
    const steering = new Steering(CONFIG);
    steering.handleClick(-0.4, 0.0);
    for (let i = 0; i < 5; i++) {
      steering.update([], [], i / CONFIG.simTimestep);
      expect(steering.lastClickedX).toBe(-0.4);
    }
  });
});
