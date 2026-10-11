/**
 * DIAGNOSTICS REPORT (Stage 2, Section 4.8)
 * =========================================
 * Reads a dataset's manifests + aborted.jsonl (NOT the PNG pixels) and writes
 * two files into the plan directory: `report.json` (machine-readable) and
 * `report.md` (human summary). Exactly the ten items listed in 4.8.
 *
 * Usage: node node_modules/tsx/dist/cli.mjs src/headless/report.ts <planDir>
 *   (or `npm run report -- <planDir>`)
 */
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { CONFIG } from '../../config/config';
import type { AbortRecord, EpisodeManifest } from '../core/types';
import { resolveTarget } from '../control/doorwayResolution';

interface PersonaAgg {
  frames: number;
  clicks: number;
  straightestTotal: number;
  straightestHit: number;
  choiceTotal: number;
  choiceHit: number;
  intentTotal: number;
  intentHonored: number;
}

interface Report {
  dataset: string;
  totals: {
    episodes: number;
    frames: number;
    bytesOnDisk: number;
    meanPngSize: number;
  };
  heldSignalHistogram: number[];
  acceptedClickHistogram: number[];
  actionHasClickedFalseShare: number;
  reachableCountHistogram: number[];
  shareFrames2Plus: number;
  straightestTargetRate: { overall: number; perPersona: Record<string, number> };
  straightestChoiceRate: { overall: number; perPersona: Record<string, number> };
  intentHonoredRate: { overall: number; perPersona: Record<string, number> };
  engineFallbackRate: number;
  labelAmbiguityShare: number;
  collisionEventsPer1000: number;
  wallOverlapFrames: number;
  viewUnsafe: {
    /** Frames carrying `viewUnsafe: true`. */
    frames: number;
    frameShare: number;
    /** Episodes containing ≥1 flagged frame. */
    episodes: number;
    episodeShare: number;
    /** The clearance margin the flag was computed with (near + this). */
    margin: number;
  };
  perPersona: Record<string, { frameShare: number; clickRate: number }>;
  integrity: { incompleteEpisodes: string[]; seedsSharedBetweenSplits: number[] };
  clipping: {
    abortedEpisodes: number;
    abortedShare: number;
    abortedPerPersona: Record<string, number>;
    framesTrimmed: number;
    briefGrazeShare: number;
  };
}

const HIST_BINS = 20; // 0.1 bins over [-1, 1]

function histIndex(x: number): number {
  const i = Math.floor((x + 1) / 0.1);
  return Math.max(0, Math.min(HIST_BINS - 1, i));
}

function blankHist(): number[] {
  return new Array<number>(HIST_BINS).fill(0);
}

async function listDirs(dir: string): Promise<string[]> {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    return entries.filter((e) => e.isDirectory()).map((e) => e.name);
  } catch {
    return [];
  }
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, 'utf8')) as T;
}

function straightestId(reachable: Array<{ doorwayId: string; screenX: number }>): string | null {
  let best: string | null = null;
  let bestAbs = Infinity;
  for (const c of reachable) {
    const a = Math.abs(c.screenX);
    if (a < bestAbs) {
      bestAbs = a;
      best = c.doorwayId;
    }
  }
  return best;
}

async function main(): Promise<void> {
  const root = process.argv[2];
  if (!root) {
    console.error('usage: tsx src/headless/report.ts <planDir>');
    process.exit(1);
  }

  const manifests: Array<{ manifest: EpisodeManifest; dir: string }> = [];
  const splits = ['train', 'val', 'test'];
  const seenSeeds = new Map<number, string[]>();
  const incomplete: string[] = [];

  for (const split of splits) {
    const splitDir = join(root, split);
    for (const seedDir of await listDirs(splitDir)) {
      const epDir = join(splitDir, seedDir);
      const manifestPath = join(epDir, 'manifest.json');
      try {
        const manifest = await readJson<EpisodeManifest>(manifestPath);
        manifests.push({ manifest, dir: epDir });
        if (manifest.complete !== true) incomplete.push(relative(root, epDir));
      } catch {
        // A seed dir with no (or unreadable) manifest = partial episode.
        incomplete.push(relative(root, epDir));
        continue;
      }
      const seed = manifests[manifests.length - 1]!.manifest.seed;
      const list = seenSeeds.get(seed) ?? [];
      list.push(split);
      seenSeeds.set(seed, list);
    }
  }

  // --- Aborted records (clipping failsafe). ---
  const aborts: AbortRecord[] = [];
  try {
    const lines = (await readFile(join(root, 'aborted.jsonl'), 'utf8')).split('\n');
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        aborts.push(JSON.parse(line) as AbortRecord);
      } catch {
        /* ignore */
      }
    }
  } catch {
    /* no aborted file */
  }

  // --- Per-frame accumulation. ---
  let totalFrames = 0;
  let totalBytes = 0;
  let totalPngs = 0;
  const heldHist = blankHist();
  const acceptedHist = blankHist();
  let actionHasClickedFalse = 0;
  const reachHist: number[] = [];
  const personas = new Map<string, PersonaAgg>();
  const straightest: { total: number; hit: number } = { total: 0, hit: 0 };
  const choice: { total: number; hit: number } = { total: 0, hit: 0 };
  const intent: { total: number; honored: number } = { total: 0, honored: 0 };
  const fallback: { total: number; count: number } = { total: 0, count: 0 };
  let ambiguityTotal = 0;
  let ambiguityMismatch = 0;
  let maxCollisionEvents = 0;
  let wallOverlapFrames = 0;
  let viewUnsafeFrames = 0;
  let viewUnsafeEpisodes = 0;
  let briefGrazeRuns = 0;
  let framesTrimmed = 0;

  function persona(name: string): PersonaAgg {
    let p = personas.get(name);
    if (!p) {
      p = { frames: 0, clicks: 0, straightestTotal: 0, straightestHit: 0, choiceTotal: 0, choiceHit: 0, intentTotal: 0, intentHonored: 0 };
      personas.set(name, p);
    }
    return p;
  }

  for (const { manifest, dir } of manifests) {
    totalFrames += manifest.frames.length;
    framesTrimmed += manifest.trimmedFrames ?? 0;
    let episodeUnsafe = 0;

    // PNG sizes (stat, never decode).
    for (const f of manifest.frames) {
      try {
        const st = await stat(join(dir, f.imagePath));
        totalBytes += st.size;
        totalPngs += 1;
      } catch {
        /* trimmed/removed frame */
      }
    }

    const frames = manifest.frames;
    const debug = manifest.debugFrames ?? [];
    for (let i = 0; i < frames.length; i++) {
      const fr = frames[i]!;
      const d = debug[i];
      heldHist[histIndex(fr.controlSignalX)]! += 1;
      if (fr.clickAccepted === true && fr.clickX !== null) {
        acceptedHist[histIndex(fr.clickX)]! += 1;
      }
      if (fr.actionHasClicked === false) actionHasClickedFalse += 1;
      if (fr.viewUnsafe === true) {
        viewUnsafeFrames += 1;
        episodeUnsafe += 1;
      }

      if (!d) continue;
      const p = persona(d.persona);
      p.frames += 1;
      if (fr.clickX !== null) p.clicks += 1;

      const reachable = (d.candidates ?? []).filter((c) => c.reachable);
      const n = reachable.length;
      reachHist[n] = (reachHist[n] ?? 0) + 1;
      maxCollisionEvents = Math.max(maxCollisionEvents, d.collisionEvents ?? 0);
      if (d.wallOverlap) wallOverlapFrames += 1;

      // Straightest-target rate among frames with 2+ reachable candidates.
      if (n >= 2 && d.targetDoorwayId !== null) {
        const s = straightestId(reachable);
        if (s !== null) {
          straightest.total += 1;
          p.straightestTotal += 1;
          if (d.targetDoorwayId === s) {
            straightest.hit += 1;
            p.straightestHit += 1;
          }
        }
      }

      // Straightest-choice rate at real choice moments (accepted clicks, 2+ reachable):
      // did the RESOLVED target equal the straightest door the bot could see?
      if (fr.clickX !== null && fr.clickAccepted === true) {
        const reachableAtClick = (d.candidates ?? []).filter((c) => c.reachable);
        if (reachableAtClick.length >= 2) {
          const s = straightestId(reachableAtClick);
          const resolved = debug[i + 1]?.targetDoorwayId ?? null;
          if (s !== null && resolved !== null) {
            choice.total += 1;
            p.choiceTotal += 1;
            if (resolved === s) {
              choice.hit += 1;
              p.choiceHit += 1;
            }
          }
        }
      }

      // Intent-honored: did the bot's click land on the door it aimed at?
      if (fr.clickX !== null && fr.clickAccepted === true && d.intentDoorwayId !== null) {
        const nextTarget = debug[i + 1]?.targetDoorwayId ?? null;
        intent.total += 1;
        p.intentTotal += 1;
        if (nextTarget === d.intentDoorwayId) {
          intent.honored += 1;
          p.intentHonored += 1;
        }
      }

      // Engine fallback: resolved target differs from nearest-by-screen-X.
      if (fr.clickX !== null && fr.clickAccepted === true) {
        const nearest = [...(d.candidates ?? [])].sort(
          (a, b) => Math.abs(a.screenX - fr.clickX!) - Math.abs(b.screenX - fr.clickX!),
        )[0];
        const resolved = debug[i + 1]?.targetDoorwayId ?? null;
        fallback.total += 1;
        if (nearest && resolved !== nearest.doorwayId) fallback.count += 1;
      }

      // Label ambiguity: stateless resolution of the held X vs the real target.
      if (d.targetDoorwayId !== null || (d.candidates?.length ?? 0) > 0) {
        const stateless = resolveTarget(d.candidates ?? [], fr.controlSignalX, null);
        ambiguityTotal += 1;
        if (stateless !== d.targetDoorwayId) ambiguityMismatch += 1;
      }
    }

    // Brief graze (L3): slide runs shorter than CLIP_L2_RUN_LENGTH.
    let run = 0;
    for (const d of debug) {
      if (d.slid) run += 1;
      else {
        if (run > 0 && run < CONFIG.clipL2RunLength) briefGrazeRuns += run;
        run = 0;
      }
    }
    if (run > 0 && run < CONFIG.clipL2RunLength) briefGrazeRuns += run;

    if (episodeUnsafe > 0) viewUnsafeEpisodes += 1;
  }

  const abortedTotal = aborts.length;
  // Trimmed episodes keep a manifest AND have an aborted.jsonl record; only
  // deleted episodes (framesKept === 0) have no manifest and must be added in.
  const abortedDeleted = aborts.filter((a) => a.framesKept === 0).length;
  const totalEpisodes = manifests.length + abortedDeleted;
  const abortedPerPersona: Record<string, number> = {};
  for (const a of aborts) {
    abortedPerPersona[a.persona] = (abortedPerPersona[a.persona] ?? 0) + 1;
  }

  const perPersona: Record<string, { frameShare: number; clickRate: number }> = {};
  const straightestPerPersona: Record<string, number> = {};
  const choicePerPersona: Record<string, number> = {};
  const intentPerPersona: Record<string, number> = {};
  for (const [name, p] of personas) {
    perPersona[name] = {
      frameShare: totalFrames > 0 ? p.frames / totalFrames : 0,
      clickRate: p.frames > 0 ? p.clicks / p.frames : 0,
    };
    straightestPerPersona[name] = p.straightestTotal > 0 ? p.straightestHit / p.straightestTotal : 0;
    choicePerPersona[name] = p.choiceTotal > 0 ? p.choiceHit / p.choiceTotal : 0;
    intentPerPersona[name] = p.intentTotal > 0 ? p.intentHonored / p.intentTotal : 0;
  }

  const seedsSharedBetweenSplits = [...seenSeeds.entries()]
    .filter(([, splitsForSeed]) => new Set(splitsForSeed).size > 1)
    .map(([seed]) => seed)
    .sort((a, b) => a - b);

  const report: Report = {
    dataset: root,
    totals: {
      episodes: totalEpisodes,
      frames: totalFrames,
      bytesOnDisk: totalBytes,
      meanPngSize: totalPngs > 0 ? totalBytes / totalPngs : 0,
    },
    heldSignalHistogram: heldHist,
    acceptedClickHistogram: acceptedHist,
    actionHasClickedFalseShare: totalFrames > 0 ? actionHasClickedFalse / totalFrames : 0,
    reachableCountHistogram: reachHist,
    shareFrames2Plus: totalFrames > 0 ? (reachHist.slice(2).reduce((s, v) => s + (v ?? 0), 0)) / totalFrames : 0,
    straightestTargetRate: {
      overall: straightest.total > 0 ? straightest.hit / straightest.total : 0,
      perPersona: straightestPerPersona,
    },
    straightestChoiceRate: {
      overall: choice.total > 0 ? choice.hit / choice.total : 0,
      perPersona: choicePerPersona,
    },
    intentHonoredRate: {
      overall: intent.total > 0 ? intent.honored / intent.total : 0,
      perPersona: intentPerPersona,
    },
    engineFallbackRate: fallback.total > 0 ? fallback.count / fallback.total : 0,
    labelAmbiguityShare: ambiguityTotal > 0 ? ambiguityMismatch / ambiguityTotal : 0,
    collisionEventsPer1000: totalFrames > 0 ? (maxCollisionEvents / totalFrames) * 1000 : 0,
    wallOverlapFrames,
    viewUnsafe: {
      frames: viewUnsafeFrames,
      frameShare: totalFrames > 0 ? viewUnsafeFrames / totalFrames : 0,
      episodes: viewUnsafeEpisodes,
      episodeShare: totalEpisodes > 0 ? viewUnsafeEpisodes / totalEpisodes : 0,
      margin: CONFIG.viewUnsafeMargin,
    },
    perPersona,
    integrity: { incompleteEpisodes: incomplete, seedsSharedBetweenSplits },
    clipping: {
      abortedEpisodes: abortedTotal,
      abortedShare: totalEpisodes > 0 ? abortedTotal / totalEpisodes : 0,
      abortedPerPersona,
      framesTrimmed,
      briefGrazeShare: totalFrames > 0 ? briefGrazeRuns / totalFrames : 0,
    },
  };

  const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;

  const md = [
    `# Diagnostics — ${root}`,
    '',
    '## 1. Totals',
    `- episodes: ${report.totals.episodes}`,
    `- frames: ${report.totals.frames}`,
    `- bytes on disk: ${(report.totals.bytesOnDisk / 1e9).toFixed(2)} GB`,
    `- mean PNG size: ${report.totals.meanPngSize.toFixed(0)} bytes`,
    '',
    '## 2. Signal histograms (0.1 bins)',
    `- accepted clicks: ${report.acceptedClickHistogram.join(', ')}`,
    `- held signal: ${report.heldSignalHistogram.join(', ')}`,
    `- share of frames with actionHasClicked=false: ${pct(report.actionHasClickedFalseShare)}`,
    '',
    '## 3. Reachable candidates',
    `- histogram (0,1,2,...): ${report.reachableCountHistogram.map((v, i) => `${i}:${v ?? 0}`).join(', ')}`,
    `- share of frames with 2+: ${pct(report.shareFrames2Plus)}`,
    '',
    '## 4. Straightest-target rate (2+ reachable)',
    `- overall: ${pct(report.straightestTargetRate.overall)}`,
    ...Object.entries(report.straightestTargetRate.perPersona)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `- ${k}: ${pct(v)}`),
    '',
    '## 4b. Straightest-choice rate at accepted clicks (2+ reachable)',
    `- overall: ${pct(report.straightestChoiceRate.overall)}`,
    ...Object.entries(report.straightestChoiceRate.perPersona)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `- ${k}: ${pct(v)}`),
    '',
    '## 5. Intent-honored + engine fallback',
    `- intent-honored overall: ${pct(report.intentHonoredRate.overall)}`,
    ...Object.entries(report.intentHonoredRate.perPersona)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `- ${k}: ${pct(v)}`),
    `- engine fallback rate: ${pct(report.engineFallbackRate)}`,
    '',
    '## 6. Label ambiguity',
    `- share of frames where the real target differs from stateless resolution: ${pct(report.labelAmbiguityShare)}`,
    '',
    '## 7. Collision',
    `- collision events per 1,000 frames: ${report.collisionEventsPer1000.toFixed(1)}`,
    `- wall-overlap frames: ${report.wallOverlapFrames}`,
    '',
    '## 8. Per-persona',
    ...Object.entries(report.perPersona)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `- ${k}: frame share ${pct(v.frameShare)}, click rate ${pct(v.clickRate)}`),
    '',
    '## 9. Integrity',
    `- incomplete episodes: ${report.integrity.incompleteEpisodes.length}` +
      (report.integrity.incompleteEpisodes.length > 0 ? ` (${report.integrity.incompleteEpisodes.slice(0, 10).join(', ')}${report.integrity.incompleteEpisodes.length > 10 ? ', …' : ''})` : ''),
    `- seeds shared between splits: ${report.integrity.seedsSharedBetweenSplits.length === 0 ? 'none' : report.integrity.seedsSharedBetweenSplits.join(', ')}`,
    '',
    '## 10. Clipping failsafe',
    `- aborted episodes: ${report.clipping.abortedEpisodes} (${pct(report.clipping.abortedShare)})`,
    ...Object.entries(report.clipping.abortedPerPersona)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `- aborted ${k}: ${v}`),
    `- frames trimmed: ${report.clipping.framesTrimmed}`,
    `- brief-graze (L3) frame share: ${pct(report.clipping.briefGrazeShare)}`,
    '',
    '## 11. View safety (`viewUnsafe` label)',
    `- margin: ${report.viewUnsafe.margin} (flag fires when a rendered surface is closer than camera near + margin)`,
    `- flagged frames: ${report.viewUnsafe.frames} (${pct(report.viewUnsafe.frameShare)})`,
    `- episodes with ≥1 flagged frame: ${report.viewUnsafe.episodes} (${pct(report.viewUnsafe.episodeShare)})`,
    '- label only: it never aborts or truncates an episode; loaders must drop any',
    '  training window that touches a flagged frame.',
    '',
  ].join('\n');

  await writeFile(join(root, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  await writeFile(join(root, 'report.md'), md + '\n');
  console.log(`[report] wrote ${join(root, 'report.json')} and ${join(root, 'report.md')}`);
  console.log(md);
}

main().catch((err) => {
  console.error('[report] fatal:', err);
  process.exit(1);
});
