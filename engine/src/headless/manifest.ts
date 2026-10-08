/**
 * MANIFEST / PNG WRITERS (Section 10)
 * ===================================
 * Frames are saved as lossless PNG (never JPEG) and each episode's manifest.json
 * is self-describing so a future Python/PyTorch consumer needs no coordination.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import type { EpisodeManifest } from '../core/types';

/** Decode a `data:image/png;base64,...` URL into raw PNG bytes. */
export function dataUrlToBuffer(dataUrl: string): Buffer {
  const comma = dataUrl.indexOf(',');
  if (comma === -1) {
    throw new Error('dataUrlToBuffer: malformed data URL (missing comma)');
  }
  return Buffer.from(dataUrl.slice(comma + 1), 'base64');
}

/** Write one PNG frame (from a data URL) to `filePath`, creating parent dirs. */
export async function writePng(filePath: string, dataUrl: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, dataUrlToBuffer(dataUrl));
}

/** Write `manifest.json` into `dir` as pretty-printed JSON. */
export async function writeManifest(dir: string, manifest: EpisodeManifest): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
}
