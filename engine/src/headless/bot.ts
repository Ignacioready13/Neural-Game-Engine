/**
 * MINIMAL PLACEHOLDER BOT (Sections 9.2 / 10.3)
 * =============================================
 * A deterministic policy that returns a pseudo-random normalized screen-X
 * click on a pseudo-random interval, and `null` (no new click) on all other
 * frames. The returned X value IS the control signal recorded in the manifest.
 *
 * This is explicitly a placeholder: it exists to exercise the pipeline end-to-
 * end. Real navigation/exploration policies are specified in a future prompt
 * and can be swapped in behind the same `(frameIndex) => number | null` shape.
 */
import { createRng, intRange } from '../core/rng';

export type PlaceholderBot = (frameIndex: number) => number | null;

/** Frames between consecutive placeholder clicks (spec suggests ~20-80). */
const CLICK_INTERVAL_MIN = 20;
const CLICK_INTERVAL_MAX = 80;

export function makePlaceholderBot(seed: number): PlaceholderBot {
  const rng = createRng(seed);
  const clicks = new Map<number, number>();
  let nextClickAt = intRange(rng, CLICK_INTERVAL_MIN, CLICK_INTERVAL_MAX);

  // Lazily extend the click schedule up to `frameIndex`. Intended for monotonic
  // frameIndex sequences (the driver walks 0..N-1 in order); once scheduled, a
  // click value never changes, so repeated calls are idempotent per index.
  function ensureThrough(frameIndex: number): void {
    while (nextClickAt <= frameIndex) {
      clicks.set(nextClickAt, rng() * 2 - 1); // uniform in [-1, 1)
      nextClickAt += intRange(rng, CLICK_INTERVAL_MIN, CLICK_INTERVAL_MAX);
    }
  }

  return (frameIndex: number) => {
    ensureThrough(frameIndex);
    return clicks.has(frameIndex) ? clicks.get(frameIndex)! : null;
  };
}
