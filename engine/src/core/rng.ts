/**
 * Deterministic seeded PRNG (mulberry32). World generation uses one instance
 * per episode so a given seed reproduces the same world — the seed is recorded
 * in the episode manifest for reproducibility.
 */

export type Rng = () => number;

/** Create a deterministic PRNG from a 32-bit integer seed. */
export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Float in [min, max). */
export function range(rng: Rng, min: number, max: number): number {
  return min + (max - min) * rng();
}

/** Integer in [min, max] inclusive. */
export function intRange(rng: Rng, min: number, max: number): number {
  return Math.floor(min + rng() * (max - min + 1));
}

/** Pick one element uniformly. */
export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)] as T;
}

/** Pick one key from a weights table (weights need not sum to 1). */
export function pickWeighted<T extends string>(
  rng: Rng,
  weights: Record<T, number>,
): T {
  const entries = Object.entries(weights) as Array<[T, number]>;
  const total = entries.reduce((s, [, w]) => s + w, 0);
  let roll = rng() * total;
  for (const [key, w] of entries) {
    roll -= w;
    if (roll <= 0) return key;
  }
  return entries[entries.length - 1]![0];
}

/** A random non-negative 32-bit integer seed derived from entropy. */
export function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}
