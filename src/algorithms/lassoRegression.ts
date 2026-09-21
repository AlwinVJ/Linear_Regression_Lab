/**
 * Lasso regression = the same linear model over expanded features, with an
 * L1 penalty added to the objective:
 *
 *   L(β) = MSE(β) + λ Σⱼ₌₁ |βⱼ|        (β₀ is never penalised)
 *
 * The solver is coordinate descent with soft thresholding — see
 * src/optimization/coordinateDescent.ts. Nothing here is faked: every value
 * the page shows comes out of this fit.
 */

import type { Point } from "@/algorithms/linearRegression";
import {
  computeColumnScaling,
  designMatrixFor,
  fitRidgeMatrix,
  predictWith,
  type ColumnScaling,
  type FeatureSpec,
} from "@/algorithms/featureRegression";
import {
  activeCount,
  l1Penalty,
  matrixMse,
  runCoordinateDescent,
  type HistoryEntry,
} from "@/optimization/coordinateDescent";

export interface LassoFit {
  specs: FeatureSpec[];
  scaling: ColumnScaling | null;
  X: number[][];
  y: number[];
  /** index 0 = intercept, then one per feature spec (fitting space) */
  beta: number[];
  history: HistoryEntry[];
  sweeps: number;
  converged: boolean;
  ok: boolean;
  reason?: string;
  mse: number;
  penalty: number;
  objective: number;
  active: number;
}

export interface LassoOptions {
  standardize?: boolean;
  maxSweeps?: number;
}

export function fitLasso(
  points: Point[],
  specs: FeatureSpec[],
  lambda: number,
  options: LassoOptions = {},
): LassoFit {
  const standardize = options.standardize ?? true;
  const scaling = standardize ? computeColumnScaling(points, specs) : null;
  const X = designMatrixFor(points, specs, scaling);
  const y = points.map((p) => p.y);
  const lam = Math.max(0, lambda);

  const run = runCoordinateDescent(X, y, lam, { maxSweeps: options.maxSweeps ?? 400 });
  const mse = matrixMse(X, y, run.beta);
  const penalty = l1Penalty(run.beta, lam);
  const base: LassoFit = {
    specs,
    scaling,
    X,
    y,
    beta: run.beta,
    history: run.history,
    sweeps: run.sweeps,
    converged: run.converged,
    ok: run.ok && Number.isFinite(mse),
    mse,
    penalty,
    objective: mse + penalty,
    active: activeCount(run.beta),
  };
  return run.reason ? { ...base, reason: run.reason } : base;
}

/** Ridge on exactly the same feature set, so the two can be compared fairly. */
export function fitRidgeFeatures(
  points: Point[],
  specs: FeatureSpec[],
  lambda: number,
  standardize = true,
): { beta: number[]; scaling: ColumnScaling | null; ok: boolean; reason?: string } {
  const scaling = standardize ? computeColumnScaling(points, specs) : null;
  const X = designMatrixFor(points, specs, scaling);
  const fit = fitRidgeMatrix(X, points.map((p) => p.y), lambda);
  return fit.reason
    ? { beta: fit.beta, scaling, ok: fit.ok, reason: fit.reason }
    : { beta: fit.beta, scaling, ok: fit.ok };
}

export const predictLasso = (
  fit: { beta: number[]; specs: FeatureSpec[]; scaling: ColumnScaling | null },
  x: number,
): number => predictWith(fit.beta, x, fit.specs, fit.scaling);

/** L2 penalty λ Σβⱼ² — used only for the Ridge side of the comparison. */
export function l2PenaltyOf(beta: number[], lambda: number): number {
  let sum = 0;
  for (let j = 1; j < beta.length; j++) sum += beta[j]! ** 2;
  return lambda * sum;
}
