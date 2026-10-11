/**
 * PULSE PROBE (scratch diagnostic, read-only).
 * The click pulse is a warm disc (RGB 255,209,122) drawn at NDC y=0 (screen row
 * H/2) at x = the click's screenX, fading over CLICK_PULSE_SECONDS (0.4s = 8
 * frames). Capture happens BEFORE the step that applies a click, so a click at
 * frame i appears as a fresh pulse in captured frame i+1.
 *
 * Paired test: for each click at frame i, compare frame i+1 vs frame i at the
 * disc region (cx, H/2) and at control bands (cx, H/2 +/- 60). Scene motion
 * affects all three similarly; a pulse only brightens the row-H/2 region.
 * If pulses are absent from the PNGs, the disc delta ~= the control delta.
 */
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import puppeteer from 'puppeteer';
import { serveDir } from '../../src/headless/staticServer';
import { CHROME_PATH } from '../../src/headless/driver';
import type { EpisodeManifest } from '../../src/core/types';

const ROOT = '/home/ignac/NeuralGameEngine/engine/data/pilot_v1';
const W = 320;
const H = 240;
const SPLITS = ['train', 'val', 'test'] as const;
const MAX_PAIRS = 400;

async function listDirs(dir: string): Promise<string[]> {
  try {
    const e = await readdir(dir, { withFileTypes: true });
    return e.filter((x) => x.isDirectory()).map((x) => x.name);
  } catch {
    return [];
  }
}

interface Pair {
  prev: string; // relative path frame i
  next: string; // relative path frame i+1
  cx: number; // pixel x of the pulse center
}

async function main(): Promise<void> {
  const pairs: Pair[] = [];
  const episodes: string[] = [];
  outer: for (const split of SPLITS) {
    for (const seedDir of await listDirs(join(ROOT, split))) {
      const rel = `${split}/${seedDir}`;
      let m: EpisodeManifest;
      try {
        m = JSON.parse(await readFile(join(ROOT, rel, 'manifest.json'), 'utf8')) as EpisodeManifest;
      } catch {
        continue;
      }
      episodes.push(rel);
      const fs = m.frames;
      for (let i = 0; i + 1 < fs.length; i++) {
        const s = fs[i]!.clickX;
        if (s === null) continue;
        const cx = Math.round(((s + 1) / 2) * W);
        pairs.push({
          prev: `${rel}/${fs[i]!.imagePath}`,
          next: `${rel}/${fs[i + 1]!.imagePath}`,
          cx: Math.max(6, Math.min(W - 7, cx)),
        });
        if (pairs.length >= MAX_PAIRS) break outer;
      }
    }
  }

  console.log(`[pulse] ${pairs.length} click pairs across ${episodes.length} episodes`);

  const server = await serveDir(ROOT, 4611);
  const browser = await puppeteer.launch({ headless: true, executablePath: CHROME_PATH });
  try {
    const page = await browser.newPage();
    // Same-origin document so canvas getImageData is not tainted.
    await page.goto(server.url + '/', { waitUntil: 'domcontentloaded' }).catch(() => {});
    const payload = JSON.stringify({ pairs, url: server.url });
    const res = (await page.evaluate(`(async () => {
      const { pairs, url } = ${payload};
      const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
      const lum = (d, idx) => 0.2126*d[idx] + 0.7152*d[idx+1] + 0.0722*d[idx+2];
      // mean luminance + mean (R-B) over a square box in the frame's pixel data
      const sample = (data, cx, cy, r) => {
        let s = 0, rb = 0, n = 0;
        for (let y = cy - r; y <= cy + r; y++) {
          for (let x = cx - r; x <= cx + r; x++) {
            const idx = (y * ${W} + x) * 4;
            s += lum(data, idx);
            rb += data[idx] - data[idx+2];
            n++;
          }
        }
        return { lum: s / n, rb: rb / n };
      };
      const canvas = document.createElement('canvas');
      canvas.width = ${W}; canvas.height = ${H};
      const ctx = canvas.getContext('2d');
      let nPairs = 0;
      const acc = { discPulse: 0, discPrev: 0, ctrlPulse: 0, ctrlPrev: 0, rbPulse: 0, rbPrev: 0 };
      for (const p of pairs) {
        const [ip, inx] = await Promise.all([load(url + '/' + p.prev), load(url + '/' + p.next)]);
        ctx.drawImage(ip, 0, 0); const prev = ctx.getImageData(0, 0, ${W}, ${H}).data;
        ctx.drawImage(inx, 0, 0); const next = ctx.getImageData(0, 0, ${W}, ${H}).data;
        const dPulse = sample(next, p.cx, ${H/2}, 5);
        const dPrev = sample(prev, p.cx, ${H/2}, 5);
        const cPulse = sample(next, p.cx, ${H/2 - 60}, 5);
        const cPrev = sample(prev, p.cx, ${H/2 - 60}, 5);
        acc.discPulse += dPulse.lum; acc.discPrev += dPrev.lum;
        acc.ctrlPulse += cPulse.lum; acc.ctrlPrev += cPrev.lum;
        acc.rbPulse += dPulse.rb; acc.rbPrev += dPrev.rb;
        nPairs++;
      }
      return { nPairs, acc };
    })()`)) as { nPairs: number; acc: Record<string, number> };

    const n = res.nPairs || 1;
    const a = res.acc;
    const discDelta = (a.discPulse! - a.discPrev!) / n;
    const ctrlDelta = (a.ctrlPulse! - a.ctrlPrev!) / n;
    const rbDelta = (a.rbPulse! - a.rbPrev!) / n;
    console.log(`[pulse] n=${res.nPairs}`);
    console.log(`[pulse] disc-region lum delta (frame i+1 minus i): ${discDelta.toFixed(2)}`);
    console.log(`[pulse] control-region lum delta (above center):   ${ctrlDelta.toFixed(2)}`);
    console.log(`[pulse] disc minus control delta:                  ${(discDelta - ctrlDelta).toFixed(2)}`);
    console.log(`[pulse] disc-region (R-B) delta:                   ${rbDelta.toFixed(2)}`);
    console.log(
      `[pulse] VERDICT: ${
        discDelta - ctrlDelta > 6 ? 'PULSE PRESENT in PNGs' : 'no pulse detected'
      }`,
    );
  } finally {
    await browser.close();
    await server.close();
  }
}

main().catch((e) => {
  console.error('[pulse] fatal:', e);
  process.exit(1);
});
