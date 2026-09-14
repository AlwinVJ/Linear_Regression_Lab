import type { Point } from "@/algorithms/linearRegression";

let counter = 0;
export const nextId = () => `pt-${counter++}`;

export const toPoints = (xs: number[], ys: number[]): Point[] =>
  xs.map((x, i) => ({ id: nextId(), x, y: ys[i] ?? 0 }));

export const DEFAULT_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
export const DEFAULT_Y = [2, 4, 5, 4, 7, 8, 9, 10, 11, 13];

export const defaultDataset = (): Point[] => toPoints(DEFAULT_X, DEFAULT_Y);

export interface Preset {
  id: string;
  name: string;
  caption: string;
  points: () => Point[];
  /** When present, the preset starts in manual mode with these parameters. */
  manual?: { slope: number; intercept: number };
}

export const PRESETS: Preset[] = [
  {
    id: "good-fit",
    name: "Good fit",
    caption: "The default data, with the best-fitting line already found.",
    points: defaultDataset,
  },
  {
    id: "bad-line",
    name: "Bad line",
    caption: "Same data, but the line starts in a poor position on purpose.",
    points: defaultDataset,
    manual: { slope: 0.2, intercept: 9 },
  },
  {
    id: "almost-perfect",
    name: "Almost perfect",
    caption: "A dataset where the points sit exactly on one straight line.",
    points: () => toPoints([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]),
  },
];

/* ------------------------------------------------------------------ */
/* Phase 3 — polynomial datasets. All generated deterministically.     */
/* ------------------------------------------------------------------ */

import { gaussian, makeRng } from "@/utils/metrics";

export interface PolyPreset {
  id: string;
  name: string;
  caption: string;
  expectation: string;
  suggestedDegree: number;
  noise: number;
  build: (seed: number, noise: number) => Point[];
}

const curveFrom = (
  xs: number[],
  f: (x: number) => number,
  seed: number,
  noise: number,
): Point[] => {
  const rng = makeRng(seed);
  return xs.map((x) => ({ id: nextId(), x, y: f(x) + (noise > 0 ? gaussian(rng) * noise : 0) }));
};

export const PARABOLA_X = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];

export const POLY_PRESETS: PolyPreset[] = [
  {
    id: "quadratic",
    name: "Quadratic curve",
    caption: "y = x². The relationship is clearly curved.",
    expectation: "Degree 1 cannot bend. Degree 2 captures the relationship almost exactly.",
    suggestedDegree: 2,
    noise: 0,
    build: (seed, noise) => curveFrom(PARABOLA_X, (x) => x * x, seed, noise),
  },
  {
    id: "linear",
    name: "Linear",
    caption: "y = 2x. A straight-line relationship.",
    expectation: "Degree 1 already fits well; higher degrees add complexity for nothing.",
    suggestedDegree: 1,
    noise: 0,
    build: (seed, noise) => curveFrom([1, 2, 3, 4, 5], (x) => 2 * x, seed, noise),
  },
  {
    id: "noisy-curve",
    name: "Noisy curve",
    caption: "A smooth cubic-ish shape with measurement noise added.",
    expectation: "Somewhere around degree 3 the model captures the shape without chasing noise.",
    suggestedDegree: 3,
    noise: 2.5,
    build: (seed, noise) =>
      curveFrom(
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
        (x) => 0.35 * x ** 3 - 1.5 * x ** 2 - 2 * x + 8,
        seed,
        noise * 4,
      ),
  },
  {
    id: "overfit",
    name: "Overfitting experiment",
    caption: "A small, noisy sample from a gentle curve.",
    expectation: "High degrees drive training error towards zero while test error grows.",
    suggestedDegree: 8,
    noise: 3,
    build: (seed, noise) =>
      curveFrom([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], (x) => 1.2 * x ** 2 - 3 * x + 4, seed, noise),
  },
];

export const DEFAULT_POLY_SEED = 20260911;
