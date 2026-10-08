/**
 * STALENESS DETECTION FOR THE BUILT BUNDLE
 * ========================================
 * Both CLIs (`record.ts` and `cli.ts`) only rebuilt `dist/` when the target
 * HTML was missing or `--build` was passed. That meant editing any source or
 * config file left a STALE vite bundle on disk, and the human-play recorder /
 * headless generator silently ran old code — producing training data (or
 * playthroughs) from a build that no longer matches the source. Since the whole
 * point of this engine is clean, correct training signal, that silent staleness
 * is worse than a loud failure.
 *
 * `needsRebuild` returns true when `dist/<targetHtml>` is missing OR older than
 * any source/config/build file, so a rebuild is triggered automatically whenever
 * the source has changed since the last build.
 */
import { existsSync, readdirSync, statSync } from 'node:fs';
import type { Dirent } from 'node:fs';
import { join } from 'node:path';

/** Collect the newest mtime under `dir` (recursively), skipping heavy/excluded
 * trees that cannot affect the bundle. */
function collectNewest(dir: string, into: number[]): void {
  let entries: Dirent[];
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    if (e.name === 'node_modules' || e.name === 'dist' || e.name === 'data' || e.name.startsWith('.')) {
      continue;
    }
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      collectNewest(p, into);
    } else {
      try {
        into.push(statSync(p).mtimeMs);
      } catch {
        /* ignore unreadable files */
      }
    }
  }
}

/**
 * True if `dist/<targetHtml>` is missing or older than any file that feeds the
 * vite build: `config/`, `src/`, the entry HTMLs, and the build config files.
 */
export function needsRebuild(root: string, targetHtml: string): boolean {
  const target = join(root, 'dist', targetHtml);
  if (!existsSync(target)) return true;
  let targetTime: number;
  try {
    targetTime = statSync(target).mtimeMs;
  } catch {
    return true;
  }

  const times: number[] = [];
  collectNewest(join(root, 'config'), times);
  collectNewest(join(root, 'src'), times);
  for (const f of ['vite.config.ts', 'package.json', 'tsconfig.json', 'index.html', 'record.html', 'headless.html']) {
    const p = join(root, f);
    if (!existsSync(p)) continue;
    try {
      times.push(statSync(p).mtimeMs);
    } catch {
      /* ignore */
    }
  }
  return times.some((t) => t > targetTime);
}
