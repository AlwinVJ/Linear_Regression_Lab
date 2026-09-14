/**
 * Deterministic pseudo-random numbers and train/test splitting.
 * A fixed seed means every reader of the page sees the same numbers.
 */

import type { Point } from "@/algorithms/linearRegression";

/** Mulberry32 — small, fast, deterministic. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller, driven by a seeded uniform source. */
export function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export interface Split {
  train: Point[];
  test: Point[];
  seed: number;
}

/**
 * Deterministic shuffle-and-cut split. The test set is never used for fitting.
 * Keeps the two extreme x values in training so the curve is not extrapolating wildly.
 */
export function trainTestSplit(points: Point[], testRatio: number, seed: number): Split {
  if (points.length < 4) return { train: points, test: [], seed };
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const first = sorted[0]!;
  const last = sorted[sorted.length - 1]!;
  const middle = sorted.slice(1, -1);

  const rng = makeRng(seed);
  for (let i = middle.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [middle[i], middle[j]] = [middle[j]!, middle[i]!];
  }
  const testCount = Math.max(1, Math.round(middle.length * testRatio));
  const test = middle.slice(0, testCount);
  const train = [first, last, ...middle.slice(testCount)];
  return {
    train: train.sort((a, b) => a.x - b.x),
    test: test.sort((a, b) => a.x - b.x),
    seed,
  };
}
