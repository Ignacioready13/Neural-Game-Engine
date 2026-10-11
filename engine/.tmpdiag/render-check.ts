/** Render frames of a straight hallway walk and dump PNGs for visual checking. */
import { writeFileSync, mkdirSync } from 'node:fs';
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH } from '../src/headless/driver';

const seed = Number(process.argv[2] ?? 1);
const frames = Number(process.argv[3] ?? 40);
const outDir = process.argv[4] ?? '/tmp/backrooms-check';
mkdirSync(outDir, { recursive: true });

const server = await serveDir('/home/ignac/NeuralGameEngine/engine/dist', 4601);
const browser = await puppeteer.launch({
  headless: true,
  executablePath: CHROME_PATH,
  args: GPU_LAUNCH_ARGS,
  env: { ...process.env, ...GPU_ENV },
});
const page = await browser.newPage();
await page.evaluateOnNewDocument((s: number) => { (window as any).__BACKROOMS_SEED__ = s; }, seed);
await page.goto(`${server.url}/headless.html`, { waitUntil: 'load' });
await page.evaluate(() => (window as any).__backrooms.init());

// Walk straight ahead with no clicks.
for (let f = 1; f <= frames; f++) {
  await page.evaluate(() => (window as any).__backrooms.step(null));
}
// Capture a few consecutive frames.
for (let f = 1; f <= 3; f++) {
  await page.evaluate(() => (window as any).__backrooms.step(null));
  const cap = await page.evaluate(() => (window as any).__backrooms.captureFrame());
  writeFileSync(`${outDir}/f${frames + f}.png`, Buffer.from(cap.dataUrl.split(',')[1], 'base64'));
}
const st = await page.evaluate(() => (window as any).__backrooms.getState());
console.log('player', JSON.stringify(st.player));
console.log('active', st.activeEntityId);
await browser.close();
await server.close();
console.log('wrote', outDir);
