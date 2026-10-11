/**
 * STAGE 2, SECTION 3 — signal-path reconciliation (scratch, not shipped).
 * Runs N short episodes through the REAL headless pipeline (with the existing
 * placeholder bot), then histograms the manifest's held `controlSignalX` per
 * frame and the ACCEPTED click values (derived from where controlSignalX
 * changes). No PNGs are written. This exists to find out why the earlier
 * investigation saw only three bins (0, +1, -1).
 */
import puppeteer from 'puppeteer';
import { serveDir } from '../src/headless/staticServer';
import { GPU_LAUNCH_ARGS, GPU_ENV, CHROME_PATH, assertGpuAcceleration } from '../src/headless/driver';
import { makePlaceholderBot } from '../src/headless/bot';
import { CONFIG } from '../config/config';

const DIST = '/home/ignac/NeuralGameEngine/engine/dist';
const EPISODES = Number(process.argv[2] ?? 5);
const FRAMES = Number(process.argv[3] ?? 600);

function bin(x: number, width = 0.1): number {
  return Math.floor(x / width + 1e-9) * width;
}

function fmtHist(hist: Map<number, number>, total: number, width = 0.1): string {
  const keys = [...hist.keys()].sort((a, b) => a - b);
  const lines: string[] = [];
  for (const k of keys) {
    const pct = ((hist.get(k)! / total) * 100).toFixed(2);
    lines.push(`  ${k.toFixed(1)} .. ${(k + width).toFixed(1)}: ${hist.get(k)}  (${pct}%)`);
  }
  return lines.join('\n');
}

const server = await serveDir(DIST, 4602);

const heldHist = new Map<number, number>();
const clickHist = new Map<number, number>();
let heldTotal = 0;
let clickCount = 0;
let acceptedCount = 0;

for (let e = 0; e < EPISODES; e++) {
  const seed = 9000 + e;
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: CHROME_PATH,
    args: GPU_LAUNCH_ARGS,
    env: { ...process.env, ...GPU_ENV },
    defaultViewport: { width: CONFIG.outputWidth, height: CONFIG.outputHeight },
  });
  try {
    const page = await browser.newPage();
    await page.evaluateOnNewDocument((s: number) => {
      (window as unknown as { __BACKROOMS_SEED__: number }).__BACKROOMS_SEED__ = s;
    }, seed);
    await page.goto(`${server.url}/headless.html`, { waitUntil: 'load' });
    await assertGpuAcceleration(page);
    await page.evaluate(() => window.__backrooms.init());

    const bot = makePlaceholderBot(seed);
    let prevSignal = 0; // controlSignalX before first click is 0
    let sawFirstClick = false;

    for (let i = 0; i < FRAMES; i++) {
      const clickX = bot(i);
      // Capture-before-step, same order as driver.ts.
      await page.evaluate(() => window.__backrooms.captureFrame());
      const state = await page.evaluate(() => window.__backrooms.getState());
      const sig = state.controlSignalX as number;

      heldHist.set(bin(sig), (heldHist.get(bin(sig)) ?? 0) + 1);
      heldTotal += 1;

      // Accepted click = a change in the held signal.
      if (sig !== prevSignal) {
        acceptedCount += 1;
        clickHist.set(bin(sig), (clickHist.get(bin(sig)) ?? 0) + 1);
        prevSignal = sig;
        sawFirstClick = sawFirstClick || sig !== 0;
      }

      await page.evaluate((x: number | null) => window.__backrooms.step(x), clickX);
    }

    const fin = await page.evaluate(() => window.__backrooms.getState());
    console.log(
      `[s3] seed ${seed}: final frame=${fin.frameIndex} controlSignalX=${fin.controlSignalX} ` +
        `acceptedClicks=${acceptedCount - (sawFirstClick ? 0 : 0)}`,
    );
  } finally {
    await browser.close();
  }
}

await server.close();

console.log('\n=== HELD controlSignalX per-frame histogram (0.1 bins) ===');
console.log(fmtHist(heldHist, heldTotal));
console.log(`\n=== ACCEPTED click-value histogram (0.1 bins), ${acceptedCount} clicks ===`);
console.log(fmtHist(clickHist, acceptedCount));
console.log(`\nheld frames total = ${heldTotal}, accepted clicks = ${acceptedCount}`);
