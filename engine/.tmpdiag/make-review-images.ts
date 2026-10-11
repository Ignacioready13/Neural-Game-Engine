/**
 * STAGE 2, 4.10 — review-image compositor (scratch, not shipped).
 * Reads pilot manifests, picks representative frames + entity transitions, and
 * composites two PNGs in the existing headless browser (2D canvas, no new deps):
 *   - legibility.png  : ~12 frames, each at 320x240 and box-downsampled to 96x72
 *                       and 64x48 then nearest-enlarged so pixels are visible.
 *   - transitions.png : 4-6 entity transitions, ~6 consecutive frames around each.
 * Selection uses manifest metadata only (currentEntityId, reachable candidates);
 * the category labels are honest best-effort heuristics.
 */
import { readFile, readdir, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import puppeteer from 'puppeteer';
import type { AbortRecord, EpisodeManifest } from '../src/core/types';
import { CHROME_PATH } from '../src/headless/driver';

const ROOT = '/home/ignac/NeuralGameEngine/engine/data/pilot_v1';
const OUT = join(ROOT, '_review');
const W = 320;
const H = 240;
const SPLITS = ['train', 'val', 'test'] as const;

interface FrameInfo {
  seed: number;
  split: string;
  frameIndex: number;
  entityId: string;
  imagePath: string;
  nReachable: number;
  minAbsScreenX: number;
  isRoom: boolean;
  nextEntityId: string | null;
}

async function listDirs(dir: string): Promise<string[]> {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    return entries.filter((e) => e.isDirectory()).map((e) => e.name);
  } catch {
    return [];
  }
}

async function main(): Promise<void> {
  const frames: FrameInfo[] = [];
  const byEpisode = new Map<string, FrameInfo[]>(); // episodeId -> frames

  for (const split of SPLITS) {
    const splitDir = join(ROOT, split);
    for (const seedDir of await listDirs(splitDir)) {
      const epDir = join(splitDir, seedDir);
      let manifest: EpisodeManifest;
      try {
        manifest = JSON.parse(await readFile(join(epDir, 'manifest.json'), 'utf8')) as EpisodeManifest;
      } catch {
        continue;
      }
      const fs = manifest.frames;
      const dbg = manifest.debugFrames ?? [];
      const ep: FrameInfo[] = [];
      for (let i = 0; i < fs.length; i++) {
        const f = fs[i]!;
        const d = dbg[i];
        const reach = (d?.candidates ?? []).filter((c) => c.reachable);
        let minAbs = Infinity;
        for (const c of reach) minAbs = Math.min(minAbs, Math.abs(c.screenX));
        ep.push({
          seed: manifest.seed,
          split,
          frameIndex: i,
          entityId: f.currentEntityId,
          imagePath: join(epDir, f.imagePath),
          nReachable: reach.length,
          minAbsScreenX: minAbs,
          isRoom: f.currentEntityId.startsWith('r'),
          nextEntityId: fs[i + 1]?.currentEntityId ?? null,
        });
      }
      frames.push(...ep);
      byEpisode.set(manifest.episodeId, ep);
    }
  }

  // Aborted episode ids (for the close-wall frame).
  const abortedIds = new Set<string>();
  try {
    const lines = (await readFile(join(ROOT, 'aborted.jsonl'), 'utf8')).split('\n');
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const a = JSON.parse(line) as AbortRecord;
        abortedIds.add(a.episodeId);
      } catch {
        /* ignore */
      }
    }
  } catch {
    /* none */
  }

  const used = new Set<string>();

  interface Slot {
    label: string;
    pick: FrameInfo | null;
  }
  const slots: Slot[] = [];

  // `find` marks the frame used immediately, so each slot gets a fresh frame.
  const find = (pred: (f: FrameInfo) => boolean, excludeSeed?: Set<number>): FrameInfo | null => {
    for (const f of frames) {
      const k = `${f.seed}:${f.frameIndex}`;
      if (used.has(k)) continue;
      if (excludeSeed?.has(f.seed)) continue;
      if (pred(f)) {
        used.add(k);
        return f;
      }
    }
    return null;
  };
  const findCloseWall = (): FrameInfo | null => {
    for (const id of abortedIds) {
      const ep = byEpisode.get(id);
      if (ep && ep.length > 0) {
        const f = ep[ep.length - 1]!;
        const k = `${f.seed}:${f.frameIndex}`;
        if (!used.has(k)) {
          used.add(k);
          return f;
        }
      }
    }
    return null;
  };

  // Rare/distinctive categories first so greedy picking doesn't steal their frames.
  slots.push({ label: 'close wall (pre-clip)', pick: findCloseWall() });
  slots.push({ label: 'bend (hallway→hallway)', pick: find((f) => !f.isRoom && f.nextEntityId?.startsWith('h') === true) });
  slots.push({ label: 'junction (4-way)', pick: find((f) => !f.isRoom && f.nReachable >= 4) });
  slots.push({ label: 'junction (3-way)', pick: find((f) => !f.isRoom && f.nReachable >= 3) });
  slots.push({ label: 'view through doorway into room', pick: find((f) => !f.isRoom && f.nextEntityId?.startsWith('r') === true) });
  slots.push({ label: 'view through doorway into hallway', pick: find((f) => f.isRoom && f.nextEntityId?.startsWith('h') === true) });
  const roomMulti = find((f) => f.isRoom && f.nReachable >= 2);
  slots.push({ label: 'room interior (multi-door)', pick: roomMulti });
  const roomSeeds = new Set<number>();
  if (roomMulti) roomSeeds.add(roomMulti.seed);
  slots.push({ label: 'room interior (second room)', pick: find((f) => f.isRoom && f.nReachable >= 1, roomSeeds) });
  slots.push({ label: 'long hallway (no door yet)', pick: find((f) => !f.isRoom && f.nReachable === 0) });
  slots.push({ label: 'hallway toward centered door', pick: find((f) => !f.isRoom && f.nReachable === 1 && f.minAbsScreenX < 0.15) });
  const hall2a = find((f) => !f.isRoom && f.nReachable === 2);
  slots.push({ label: 'hallway, 2 reachable (2nd-degree doorway)', pick: hall2a });
  const hall2Seeds = new Set<number>();
  if (hall2a) hall2Seeds.add(hall2a.seed);
  slots.push({ label: 'hallway, 2 reachable (spread)', pick: find((f) => !f.isRoom && f.nReachable === 2, hall2Seeds) });

  // Build transitions: consecutive currentEntityId changes.
  const transitions: Array<{ label: string; cells: FrameInfo[] }> = [];
  const usedTrans = new Set<string>();
  for (const f of frames) {
    if (transitions.length >= 5) break;
    if (f.nextEntityId === null || f.nextEntityId === f.entityId) continue;
    const key = `${f.seed}:${f.frameIndex}`;
    if (usedTrans.has(key)) continue;
    const ep = byEpisode.get(`${'pilot_v1'}-${f.split}-${f.seed}`);
    if (!ep) continue;
    const t = f.frameIndex;
    const start = Math.max(0, t - 3);
    const end = Math.min(ep.length - 1, t + 2);
    if (end - start + 1 < 5) continue;
    usedTrans.add(key);
    transitions.push({
      label: `${f.entityId}→${f.nextEntityId} @ seed ${f.seed} f${t}`,
      cells: ep.slice(start, end + 1),
    });
  }

  // Load PNG bytes -> data URI, then composite in the browser.
  const cellData: Array<{ key: string; src: string }> = [];
  const dataByKey = new Map<string, string>();
  const addCell = (key: string, path: string) => {
    if (dataByKey.has(key)) return;
    dataByKey.set(key, path);
  };

  const legibilityRows: Array<{ label: string; keys: string[] }> = [];
  for (const s of slots) {
    if (!s.pick) continue;
    const base = `L:${s.pick.seed}:${s.pick.frameIndex}`;
    legibilityRows.push({ label: s.label, keys: [base] });
    addCell(base, s.pick.imagePath);
  }

  const transitionRows: Array<{ label: string; keys: string[] }> = [];
  for (const t of transitions) {
    const keys: string[] = [];
    for (const c of t.cells) {
      const k = `T:${c.seed}:${c.frameIndex}`;
      keys.push(k);
      addCell(k, c.imagePath);
    }
    transitionRows.push({ label: t.label, keys });
  }

  // Read all PNG bytes.
  for (const [key, path] of dataByKey) {
    const buf = await readFile(path);
    cellData.push({ key, src: `data:image/png;base64,${buf.toString('base64')}` });
  }
  const srcByKey = new Map(cellData.map((c) => [c.key, c.src]));

  const browser = await puppeteer.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    const page = await browser.newPage();
    // Run the compositor as a string expression so tsx/esbuild never transforms
    // it (a serialized function would otherwise reference esbuild's __name helper).
    const payload = JSON.stringify({
      rows: legibilityRows,
      trRows: transitionRows,
      srcByKey: Object.fromEntries(srcByKey),
    });
    const result = (await page.evaluate(`(async () => {
      const { rows, trRows, srcByKey } = ${payload};
      const load = (src) =>
        new Promise((res, rej) => {
          const img = new Image();
          img.onload = () => res(img);
          img.onerror = rej;
          img.src = src;
        });
      const drawDown = (ctx, img, x, y, smallW, smallH) => {
        const off = document.createElement('canvas');
        off.width = smallW;
        off.height = smallH;
        const oc = off.getContext('2d');
        oc.imageSmoothingEnabled = true;
        oc.imageSmoothingQuality = 'high';
        oc.drawImage(img, 0, 0, smallW, smallH);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, x, y, 320, 240);
        ctx.imageSmoothingEnabled = true;
      };

      const legible = await (async () => {
        const labelW = 230, cellW = 320, cellH = 240, gap = 12, headerH = 30;
        const rowH = cellH + gap, cols = 3;
        const cw = labelW + cols * (cellW + gap) + gap;
        const ch = headerH + rows.length * rowH + gap;
        const canvas = document.createElement('canvas');
        canvas.width = cw; canvas.height = ch;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, cw, ch);
        ctx.fillStyle = '#eee'; ctx.font = '18px sans-serif';
        const heads = ['320×240', '96×72 → 320', '64×48 → 320'];
        for (let c = 0; c < cols; c++) {
          ctx.fillText(heads[c], labelW + c * (cellW + gap) + gap, headerH - 6);
        }
        for (let r = 0; r < rows.length; r++) {
          const y = headerH + r * rowH + gap;
          ctx.fillText(rows[r].label, 8, y + 20);
          const img = await load(srcByKey[rows[r].keys[0]]);
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(img, labelW + 0 * (cellW + gap) + gap, y, cellW, cellH);
          drawDown(ctx, img, labelW + 1 * (cellW + gap) + gap, y, 96, 72);
          drawDown(ctx, img, labelW + 2 * (cellW + gap) + gap, y, 64, 48);
        }
        return canvas.toDataURL('image/png');
      })();

      const trans = await (async () => {
        const labelW = 230, cellW = 320, cellH = 240, gap = 10, headerH = 30;
        const rowH = cellH + gap;
        const cols = trRows[0] ? trRows[0].keys.length : 6;
        const cw = labelW + cols * (cellW + gap) + gap;
        const ch = headerH + trRows.length * rowH + gap;
        const canvas = document.createElement('canvas');
        canvas.width = cw; canvas.height = ch;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, cw, ch);
        ctx.fillStyle = '#eee'; ctx.font = '18px sans-serif';
        for (let c = 0; c < cols; c++) ctx.fillText('f' + (c - 3), labelW + c * (cellW + gap) + gap, headerH - 6);
        for (let r = 0; r < trRows.length; r++) {
          const y = headerH + r * rowH + gap;
          ctx.fillText(trRows[r].label, 8, y + 20);
          for (let c = 0; c < trRows[r].keys.length; c++) {
            const img = await load(srcByKey[trRows[r].keys[c]]);
            ctx.drawImage(img, labelW + c * (cellW + gap) + gap, y, cellW, cellH);
          }
        }
        return canvas.toDataURL('image/png');
      })();

      return { legible, trans, nRows: rows.length, nTrans: trRows.length };
    })()`)) as { legible: string; trans: string; nRows: number; nTrans: number };

    const b64 = (dataUrl: string): Buffer => Buffer.from(dataUrl.split(',')[1]!, 'base64');
    await mkdir(OUT, { recursive: true });
    await writeFile(join(OUT, 'legibility.png'), b64(result.legible));
    await writeFile(join(OUT, 'transitions.png'), b64(result.trans));
    console.log(`[review] wrote legibility.png (${result.nRows} rows) and transitions.png (${result.nTrans} transitions)`);
    console.log('[review] legibility rows:', legibilityRows.map((r) => r.label).join(' | '));
    console.log('[review] transitions:', transitions.map((t) => t.label).join(' | '));
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[review] fatal:', err);
  process.exit(1);
});
