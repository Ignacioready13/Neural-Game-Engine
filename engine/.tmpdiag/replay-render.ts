/** Re-render selected frames of the recorded playtest from the same seed+clicks. */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH } from '../src/headless/driver';

const MANIFEST = '/home/ignac/NeuralGameEngine/engine/data/playtest_2026-10-09T02-53-19-924Z/manifest.json';
const OUT = process.argv[2] ?? '/tmp/diag/out-before';
const wantedArg = process.argv[3];
const wanted = new Set((wantedArg ?? '135,172,500').split(',').map(Number));
const maxFrame = Number(process.argv[4] ?? Math.max(...[...wanted]));

const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
mkdirSync(OUT, { recursive: true });

const server = await serveDir('/home/ignac/NeuralGameEngine/engine/dist', 4599);
const browser = await puppeteer.launch({
  headless: true,
  executablePath: CHROME_PATH,
  args: GPU_LAUNCH_ARGS,
  env: { ...process.env, ...GPU_ENV },
});
const page = await browser.newPage();
await page.evaluateOnNewDocument((seed: number) => { (window as any).__BACKROOMS_SEED__ = seed; }, manifest.seed);
await page.goto(`${server.url}/headless.html`, { waitUntil: 'load' });
await page.evaluate(() => (window as any).__backrooms.init());
const gpu = await page.evaluate(() => (window as any).__backrooms.gpuInfo());
console.log('GPU:', JSON.stringify(gpu));

for (const f of manifest.frames) {
  if (f.frameIndex > maxFrame) break;
  const cap = await page.evaluate((clickX: number | null) => {
    const api = (window as any).__backrooms;
    api.step(clickX);
    if ((window as any).__CAPTURE__) return api.captureFrame();
    return null;
  }, f.clickedX);
  if (f.frameIndex < maxFrame && !wanted.has(f.frameIndex)) continue;
  const st = await page.evaluate(() => (window as any).__backrooms.getState());
  if (wanted.has(f.frameIndex)) {
    const cap2 = await page.evaluate(() => (window as any).__backrooms.captureFrame());
    const b64 = cap2.dataUrl.split(',')[1];
    writeFileSync(`${OUT}/f${String(f.frameIndex).padStart(4, '0')}.png`, Buffer.from(b64, 'base64'));
    console.log(`frame ${f.frameIndex}: sim player=(${st.player.x.toFixed(2)},${st.player.z.toFixed(2)}) h=${st.player.heading.toFixed(3)} active=${st.activeEntityId} | manifest=(${f.playerX.toFixed(2)},${f.playerZ.toFixed(2)}) h=${f.playerHeading.toFixed(3)} active=${f.currentEntityId}`);
  } else if (f.frameIndex === maxFrame) {
    console.log(`final frame ${f.frameIndex}: sim player=(${st.player.x.toFixed(2)},${st.player.z.toFixed(2)}) active=${st.activeEntityId} | manifest=(${f.playerX.toFixed(2)},${f.playerZ.toFixed(2)}) active=${f.currentEntityId}`);
  }
}
await browser.close();
await server.close();
console.log('wrote', OUT);
