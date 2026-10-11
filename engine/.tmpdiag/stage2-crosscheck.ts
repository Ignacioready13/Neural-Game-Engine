/**
 * Cross-check: run seeds through the REAL headless pipeline and print the
 * engine's own cumulative `collisionEvents` (and wallOverlap) per episode, to
 * compare against the renderer-free replica's slide counts.
 */
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH, assertGpuAcceleration } from '../src/headless/driver';
import { makePlaceholderBot } from '../src/headless/bot';
import { CONFIG } from '../config/config';

const DIST = '/home/ignac/NeuralGameEngine/engine/dist';
const seeds = (process.argv[2] ?? '4000,4001,4002,4065,4066,4094').split(',').map(Number);
const FRAMES = Number(process.argv[3] ?? 600);

const server = await serveDir(DIST, 4603);
for (const seed of seeds) {
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
    for (let i = 0; i < FRAMES; i++) {
      const clickX = bot(i);
      await page.evaluate(() => window.__backrooms.captureFrame());
      await page.evaluate((x: number | null) => window.__backrooms.step(x), clickX);
    }
    const s = await page.evaluate(() => window.__backrooms.getState());
    console.log(`seed ${seed}: collisionEvents=${s.collisionEvents} wallOverlap=${s.wallOverlap} frames=${s.frameIndex}`);
  } finally {
    await browser.close();
  }
}
await server.close();
