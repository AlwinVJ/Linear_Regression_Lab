/**
 * A small generic feature layer shared by the Lasso page.
 *
 * Phase 3 gave us polynomial features of one input x. Lasso is most convincing
 * when the model also carries features that contribute little, so here a
 * "feature set" is simply a list of named functions of x:
 *
 *   x, x², x³ …           the polynomial expansion (Phase 3)
 *   noise_1, noise_2      wiggles unrelated to the target
 *   x_twin                a feature almost identical to x (correlated pair)
 *
 * Everything downstream (design matrix, standardisation, prediction, MSE,
 * ridge comparison) works on that list, so no polynomial-specific logic is
 * duplicated.
 */

import type { Point } from "@/algorithms/linearRegression";
import { solveLeastSquares } from "@/algorithms/polynomialRegression";
import { powerLabel } from "@/utils/polynomialFeatures";

export type FeatureKind = "poly" | "noise" | "corr";

export interface FeatureSpec {
  key: string;
  label: string;
  description: string;
  kind: FeatureKind;
  f: (x: number) => number;
}

export interface FeatureOptions {
  noise?: boolean;
  correlated?: boolean;
}

export function buildFeatures(degree: number, options: FeatureOptions = {}): FeatureSpec[] {
  const specs: FeatureSpec[] = [];
  for (let j = 1; j <= Math.max(1, Math.round(degree)); j++) {
    specs.push({
      key: `x${j}`,
      label: powerLabel(j),
      description: `polynomial term x^${j}`,
      kind: "poly",
      f: (x) => x ** j,
    });
  }
  if (options.noise) {
    specs.push({
      key: "noise_1",
      label: "noise_1",
      description: "sin(3x) — a wiggle with no relation to the target",
      kind: "noise",
      f: (x) => Math.sin(3 * x),
    });
    specs.push({
      key: "noise_2",
      label: "noise_2",
      description: "cos(5x) — another unrelated wiggle",
      kind: "noise",
      f: (x) => Math.cos(5 * x),
    });
  }
  if (options.correlated) {
    specs.push({
      key: "x_twin",
      label: "x_twin",
      description: "x + 0.05·sin(2x) — almost a copy of x",
      kind: "corr",
      f: (x) => x + 0.05 * Math.sin(2 * x),
    });
  }
  return specs;
}

/** Raw feature values for one x (intercept NOT included). */
export const rawRow = (x: number, specs: FeatureSpec[]): number[] => specs.map((s) => s.f(x));

export interface ColumnScaling {
  means: number[];
  stds: number[];
}

/** Column means and standard deviations, computed from the given rows only. */
export function computeColumnScaling(points: Point[], specs: FeatureSpec[]): ColumnScaling {
  const n = Math.max(points.length, 1);
  const means: number[] = [];
  const stds: number[] = [];
  specs.forEach((s, j) => {
    let sum = 0;
    for (const p of points) sum += s.f(p.x);
    const mean = sum / n;
    let varSum = 0;
    for (const p of points) varSum += (s.f(p.x) - mean) ** 2;
    const std = Math.sqrt(varSum / n);
    means[j] = mean;
    stds[j] = std > 1e-12 && Number.isFinite(std) ? std : 1;
  });
  return { means, stds };
}

/** Feature row in fitting space (standardised when a scaling is supplied). */
export function fitFeatureRow(
  x: number,
  specs: FeatureSpec[],
  scaling: ColumnScaling | null,
): number[] {
  return specs.map((s, j) =>
    scaling ? (s.f(x) - (scaling.means[j] ?? 0)) / (scaling.stds[j] ?? 1) : s.f(x),
  );
}

export const designMatrixFor = (
  points: Point[],
  specs: FeatureSpec[],
  scaling: ColumnScaling | null,
): number[][] => points.map((p) => fitFeatureRow(p.x, specs, scaling));

/** ŷ = β₀ + Σ βⱼ zⱼ(x) — β index 0 is the intercept. */
export function predictWith(
  beta: number[],
  x: number,
  specs: FeatureSpec[],
  scaling: ColumnScaling | null,
): number {
  const row = fitFeatureRow(x, specs, scaling);
  let sum = beta[0] ?? 0;
  for (let j = 0; j < row.length; j++) sum += (beta[j + 1] ?? 0) * row[j]!;
  return sum;
}

export function mseWith(
  points: Point[],
  beta: number[],
  specs: FeatureSpec[],
  scaling: ColumnScaling | null,
): number {
  if (points.length === 0) return Number.NaN;
  let sum = 0;
  for (const p of points) sum += (p.y - predictWith(beta, p.x, specs, scaling)) ** 2;
  return sum / points.length;
}

/** Equation string built from named features, skipping coefficients that are zero. */
export function featureEquation(beta: number[], specs: FeatureSpec[], digits = 3): string {
  const parts: string[] = [];
  const b0 = Number((beta[0] ?? 0).toFixed(digits));
  parts.push(`${b0 < 0 ? "−" : ""}${Math.abs(b0).toFixed(digits)}`);
  specs.forEach((s, j) => {
    const b = Number((beta[j + 1] ?? 0).toFixed(digits));
    if (b === 0) return;
    parts.push(`${b < 0 ? "−" : "+"} ${Math.abs(b).toFixed(digits)}·${s.label}`);
  });
  return `ŷ = ${parts.join(" ")}`;
}

/* ------------------------------------------------------------------ */
/* Ridge / OLS on the same feature set (for the side-by-side section)  */
/* ------------------------------------------------------------------ */

export interface MatrixFit {
  beta: number[]; // index 0 = intercept
  ok: boolean;
  reason?: string;
}

const INSTABILITY =
  "Numerical instability detected. Try reducing polynomial degree or enabling feature standardization.";

/**
 * Ridge (λ = 0 gives ordinary least squares) solved as an augmented
 * least-squares problem, reusing the Householder QR solver. The intercept
 * column is never penalised.
 */
export function fitRidgeMatrix(X: number[][], y: number[], lambda: number): MatrixFit {
  const n = X.length;
  if (n === 0) return { beta: [], ok: false, reason: "There are no observations to fit." };
  const p = X[0]!.length;
  const rows = X.map((r) => [1, ...r]);
  const targets = y.slice();
  const lam = Math.max(0, lambda);
  if (lam > 0) {
    const w = Math.sqrt(n * lam);
    for (let j = 1; j <= p; j++) {
      const row = new Array(p + 1).fill(0);
      row[j] = w;
      rows.push(row);
      targets.push(0);
    }
  } else if (n < p + 1) {
    return {
      beta: new Array(p + 1).fill(0),
      ok: false,
      reason: `Fitting ${p + 1} coefficients with no regularization needs at least ${p + 1} observations, but only ${n} are available.`,
    };
  }
  const beta = solveLeastSquares(rows, targets);
  if (!beta || !beta.every(Number.isFinite) || beta.some((b) => Math.abs(b) > 1e12)) {
    return { beta: new Array(p + 1).fill(0), ok: false, reason: INSTABILITY };
  }
  return { beta, ok: true };
}

export const MATRIX_INSTABILITY_MESSAGE = INSTABILITY;
