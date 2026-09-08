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
