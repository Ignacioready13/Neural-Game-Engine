/**
 * PROVENANCE (Stage 2, Section 4.4)
 * =================================
 * Per-dataset provenance fields: a full-config hash, the git commit, a dirty
 * flag, and a content hash of the engine source + config. The source hash
 * matters because the user commits manually, so the tree is often dirty and the
 * commit alone would not identify the code that produced a dataset.
 */
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, relative } from 'node:path';
import type { EngineConfig } from '../../config/config';

/** SHA-256 (hex) of the FULL config object, not the manifest's display subset. */
export function configHash(config: EngineConfig): string {
  return createHash('sha256').update(JSON.stringify(config)).digest('hex');
}

/** `git rev-parse HEAD`, or '' when git is unavailable. */
export function engineCommit(root: string): string {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
  if (r.status !== 0) return '';
  return r.stdout.trim();
}

/** True when the working tree has any uncommitted change. */
export function dirtyFlag(root: string): boolean {
  const r = spawnSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' });
  if (r.status !== 0) return false;
  return r.stdout.trim().length > 0;
}

function collectFiles(dir: string, into: string[]): void {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    if (e.name === 'node_modules' || e.name === 'dist' || e.name.startsWith('.')) continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) collectFiles(p, into);
    else if (e.isFile()) into.push(p);
  }
}

/** Content hash of `engine/src` + `engine/config` (relative path + bytes). */
export function engineSourceHash(root: string): string {
  const files: string[] = [];
  collectFiles(join(root, 'src'), files);
  collectFiles(join(root, 'config'), files);
  files.sort();
  const h = createHash('sha256');
  for (const f of files) {
    h.update(relative(root, f));
    h.update('\0');
    h.update(readFileSync(f));
    h.update('\0');
  }
  return h.digest('hex');
}
