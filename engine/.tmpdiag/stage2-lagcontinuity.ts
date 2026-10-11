/**
 * STAGE 2, A3(b) lag + A3(c) continuity (scratch).
 *
 * (b) At each active-entity change, render the identical state twice in a row
 *     and confirm byte-identical pixels (and that the frame is not blank). This
 *     detects async texture upload / shader-compile / nondeterminism in capture.
 * (c) Over short episodes, compute consecutive-frame MAD on a downsampled RGB
 *     comparison, then list the largest spikes and whether each coincides with
 *     an active-entity change.
 */
import { inflateSync } from 'node:zlib';
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH, assertGpuAcceleration } from '../src/headless/driver';
import { makePlaceholderBot } from '../src/headless/bot';
import { CONFIG } from '../config/config';

const DIST = '/home/ignac/NeuralGameEngine/engine/dist';
const EPISODES = Number(process.argv[2] ?? 10);
const FRAMES = Number(process.argv[3] ?? 120);
const SEED_BASE = Number(process.argv[4] ?? 5000);

/** Minimal PNG decoder for the browser's `toDataURL('image/png')` (8-bit, no interlace). */
function decodePng(buf: Buffer): { width: number; height: number; rgba: Buffer } {
  if (buf[0] !== 0x89 || buf[1] !== 0x50 || buf[2] !== 0x4e || buf[3] !== 0x47) throw new Error('not a PNG');
  let pos = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;
    if (type === 'IHDR') {
      width = data.readUInt32BE(0); height = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
  }
  if (bitDepth !== 8 || interlace !== 0 || (colorType !== 6 && colorType !== 2)) {
    throw new Error(`unsupported PNG bitDepth=${bitDepth} colorType=${colorType} interlace=${interlace}`);
  }
  const bpp = colorType === 6 ? 4 : 3;
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * bpp;
  const out = Buffer.alloc(height * stride);
  const paeth = (a: number, b: number, c: number) => {
    const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
  };
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const rowStart = y * (stride + 1) + 1;
    for (let x = 0; x < stride; x++) {
      const left = x >= bpp ? out[y * stride + x - bpp] : 0;
      const up = y > 0 ? out[(y - 1) * stride + x] : 0;
      const upLeft = y > 0 && x >= bpp ? out[(y - 1) * stride + x - bpp] : 0;
      const b = raw[rowStart + x];
      let v: number;
      if (filter === 0) v = b;
      else if (filter === 1) v = b + left;
      else if (filter === 2) v = b + up;
      else if (filter === 3) v = b + ((left + up) >> 1);
      else if (filter === 4) v = b + paeth(left, up, upLeft);
      else throw new Error('bad filter ' + filter);
      out[y * stride + x] = v & 0xff;
    }
  }
  return { width, height, rgba: out };
}

function meanLuma(rgba: Buffer, w: number, h: number): number {
  let sum = 0, n = 0;
  for (let y = 0; y < h; y += 4) for (let x = 0; x < w; x += 4) {
    const i = (y * w + x) * 4;
    sum += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    n++;
  }
  return sum / n;
}

function mad(a: { rgba: Buffer; width: number; height: number }, b: { rgba: Buffer; width: number; height: number }): number {
  let sum = 0, n = 0;
  for (let y = 0; y < a.height; y += 4) for (let x = 0; x < a.width; x += 4) {
    const i = (y * a.width + x) * 4;
    sum += Math.abs(a.rgba[i] - b.rgba[i]) + Math.abs(a.rgba[i + 1] - b.rgba[i + 1]) + Math.abs(a.rgba[i + 2] - b.rgba[i + 2]);
    n += 3;
  }
  return sum / n;
}

const server = await serveDir(DIST, 4604);
let entityEvents = 0;
let doubleRenderMismatches = 0;
let blankFirstRenders = 0;
const allSpikes: Array<{ seed: number; frame: number; mad: number; entity: boolean }> = [];

for (let e = 0; e < EPISODES; e++) {
  const seed = SEED_BASE + e;
  const browser = await puppeteer.launch({
    headless: true, executablePath: CHROME_PATH, args: GPU_LAUNCH_ARGS,
    env: { ...process.env, ...GPU_ENV },
    defaultViewport: { width: CONFIG.outputWidth, height: CONFIG.outputHeight },
  });
  try {
    const page = await browser.newPage();
    await page.evaluateOnNewDocument((s: number) => { (window as any).__BACKROOMS_SEED__ = s; }, seed);
    await page.goto(`${server.url}/headless.html`, { waitUntil: 'load' });
    await assertGpuAcceleration(page);
    await page.evaluate(() => window.__backrooms.init());

    const bot = makePlaceholderBot(seed);
    const caps: Array<{ dataUrl: string; entity: string }> = [];
    let prevEntity: string | null = null;

    for (let i = 0; i < FRAMES; i++) {
      const clickX = bot(i);
      const cap = await page.evaluate(() => window.__backrooms.captureFrame());
      const st = await page.evaluate(() => window.__backrooms.getState());
      caps.push({ dataUrl: cap.dataUrl, entity: st.activeEntityId as string });

      if (prevEntity !== null && st.activeEntityId !== prevEntity) {
        entityEvents += 1;
        // (b) double-render: identical state, no step in between.
        const a = await page.evaluate(() => window.__backrooms.captureFrame());
        const b = await page.evaluate(() => window.__backrooms.captureFrame());
        if (a.dataUrl !== b.dataUrl) doubleRenderMismatches += 1;
        const dec = decodePng(Buffer.from(cap.dataUrl.split(',')[1], 'base64'));
        if (meanLuma(dec.rgba, dec.width, dec.height) < 5) blankFirstRenders += 1;
      }
      prevEntity = st.activeEntityId as string;
      await page.evaluate((x: number | null) => window.__backrooms.step(x), clickX);
    }

    // (c) continuity MAD over consecutive frames.
    const dec = caps.map((c) => ({ ...decodePng(Buffer.from(c.dataUrl.split(',')[1], 'base64')), entity: c.entity }));
    for (let i = 1; i < dec.length; i++) {
      const m = mad(dec[i - 1], dec[i]);
      allSpikes.push({ seed, frame: i, mad: m, entity: dec[i].entity !== dec[i - 1].entity });
    }
  } finally {
    await browser.close();
  }
}
await server.close();

const mads = allSpikes.map((s) => s.mad).sort((a, b) => a - b);
const median = mads[Math.floor(mads.length / 2)]!;
const p95 = mads[Math.floor(mads.length * 0.95)]!;
const p99 = mads[Math.floor(mads.length * 0.99)]!;

console.log('=== 4.5(b) lag: double-render at entity change ===');
console.log(`  entity-change events: ${entityEvents}`);
console.log(`  double-render mismatches: ${doubleRenderMismatches}`);
console.log(`  blank first-render frames (mean luma < 5): ${blankFirstRenders}`);

console.log('\n=== 4.5(c) continuity: consecutive-frame MAD (downsampled 1/4) ===');
console.log(`  frames compared: ${allSpikes.length}`);
console.log(`  median MAD = ${median.toFixed(3)}   p95 = ${p95.toFixed(3)}   p99 = ${p99.toFixed(3)}`);
const top = [...allSpikes].sort((a, b) => b.mad - a.mad).slice(0, 20);
console.log('  top 20 spikes (seed, frame, MAD, coincides-with-entity-change):');
for (const s of top) {
  console.log(`    seed ${s.seed} frame ${String(s.frame).padStart(3)}  MAD ${s.mad.toFixed(3)}  entityChange=${s.entity}`);
}
const entitySpikes = allSpikes.filter((s) => s.entity).map((s) => s.mad).sort((a, b) => a - b);
const entityMedian = entitySpikes.length ? entitySpikes[Math.floor(entitySpikes.length / 2)]! : NaN;
console.log(`  entity-change frames: ${entitySpikes.length}, median MAD among them = ${entityMedian.toFixed(3)}`);
