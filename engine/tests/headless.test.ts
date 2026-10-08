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
import type { EpisodeManifest } from '../src/core/types';

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
        episodeId: 'test-episode',
        renderWidth: 1280,
        renderHeight: 960,
        outputWidth: 320,
        outputHeight: 240,
        simTimestep: 0.05,
        seed: 7,
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
            playerX: 0,
            playerZ: 0,
            playerHeading: 0,
            currentEntityId: 'entity-1',
          },
        ],
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
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
