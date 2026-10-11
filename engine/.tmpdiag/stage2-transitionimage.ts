/**
 * STAGE 2, A3(c) — dump a side-by-side of the frame just before and just after
 * an active-entity change (doorway crossing), so the user can judge how abrupt
 * the intended black-fill -> real-content swap is. Stitched in Node (tiny PNG
 * decoder + encoder) to avoid transpiled in-page helpers.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { inflateSync, deflateSync } from 'node:zlib';
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH, assertGpuAcceleration } from '../src/headless/driver';
import { makePlaceholderBot } from '../src/headless/bot';
import { CONFIG } from '../config/config';

const DIST = '/home/ignac/NeuralGameEngine/engine/dist';
const seed = Number(process.argv[2] ?? 5007);
const FRAMES = Number(process.argv[3] ?? 120);
const outDir = process.argv[4] ?? '/tmp/backrooms-transition';
mkdirSync(outDir, { recursive: true });

// ---- PNG decode ----
function decodePng(buf: Buffer): { width: number; height: number; rgba: Buffer } {
  let pos = 8, width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;
    if (type === 'IHDR') { width = data.readUInt32BE(0); height = data.readUInt32BE(4); bitDepth = data[8]; colorType = data[9]; interlace = data[12]; }
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
  }
  if (bitDepth !== 8 || interlace !== 0 || (colorType !== 6 && colorType !== 2)) throw new Error('unsupported PNG');
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
      else throw new Error('bad filter');
      out[y * stride + x] = v & 0xff;
    }
  }
  return { width, height, rgba: out };
}

// ---- PNG encode (RGBA, filter 0) ----
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const typeBuf = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])));
  return Buffer.concat([len, typeBuf, data, crc]);
}
function encodePng(w: number, h: number, rgba: Buffer): Buffer {
  const stride = w * 4;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0;
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const server = await serveDir(DIST, 4605);
const browser = await puppeteer.launch({
  headless: true, executablePath: CHROME_PATH, args: GPU_LAUNCH_ARGS,
  env: { ...process.env, ...GPU_ENV },
  defaultViewport: { width: CONFIG.outputWidth, height: CONFIG.outputHeight },
});
const page = await browser.newPage();
await page.evaluateOnNewDocument((s: number) => { (window as any).__BACKROOMS_SEED__ = s; }, seed);
await page.goto(`${server.url}/headless.html`, { waitUntil: 'load' });
await assertGpuAcceleration(page);
await page.evaluate(() => window.__backrooms.init());

const bot = makePlaceholderBot(seed);
let prev: { dataUrl: string; entity: string } | null = null;
let transition: { before: string; after: string; frame: number } | null = null;
for (let i = 0; i < FRAMES && !transition; i++) {
  const cap = await page.evaluate(() => window.__backrooms.captureFrame());
  const st = await page.evaluate(() => window.__backrooms.getState());
  if (prev && st.activeEntityId !== prev.entity) transition = { before: prev.dataUrl, after: cap.dataUrl, frame: i };
  prev = { dataUrl: cap.dataUrl, entity: st.activeEntityId as string };
  await page.evaluate((x: number | null) => window.__backrooms.step(x), bot(i));
}
await browser.close();
await server.close();

if (!transition) {
  console.log('no entity transition found in', FRAMES, 'frames');
} else {
  const a = decodePng(Buffer.from(transition.before.split(',')[1], 'base64'));
  const b = decodePng(Buffer.from(transition.after.split(',')[1], 'base64'));
  const w = a.width + b.width;
  const h = Math.max(a.height, b.height);
  const stitched = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++) {
    a.rgba.copy(stitched, y * w * 4, y * a.width * 4, (y + 1) * a.width * 4);
    b.rgba.copy(stitched, y * w * 4 + a.width * 4, y * b.width * 4, (y + 1) * b.width * 4);
  }
  const file = `${outDir}/seed${seed}_transition_frame${transition.frame}.png`;
  writeFileSync(file, encodePng(w, h, stitched));
  console.log('transition at frame', transition.frame, '-> wrote', file);
}
