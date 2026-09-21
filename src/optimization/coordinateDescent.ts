/**
 * Coordinate descent for Lasso.
 *
 * Objective (β₀ is never penalised):
 *
 *   L(β) = (1/n) Σᵢ (yᵢ − ŷᵢ)²  +  λ Σⱼ₌₁ |βⱼ|
 *
 * |β| has no derivative at 0, so plain gradient descent is the wrong tool.
 * Coordinate descent optimises one coefficient at a time, and for a single
 * coefficient the problem has a closed-form answer:
 *
 *   ρⱼ    = (2/n) Σᵢ xᵢⱼ (rᵢ + xᵢⱼ βⱼ)     (r = current residuals)
 *   denomⱼ = (2/n) Σᵢ xᵢⱼ²
 *   βⱼ    = S(ρⱼ, λ) / denomⱼ             (S = soft thresholding)
 *
 * The intercept is the mean of the residual left by the other terms.
 */

import { softThreshold } from "@/optimization/softThresholding";

export const LASSO_INSTABILITY_MESSAGE =
  "Numerical instability detected. Try reducing polynomial degree or enabling feature standardization.";

export const LASSO_NOT_CONVERGED_MESSAGE =
  "The optimizer has not converged yet. Try increasing the iteration limit or adjusting λ.";

export interface CoordinateUpdate {
  /** 1-based coefficient index in β (0 would be the intercept). */
  j: number;
  before: number;
  /** (2/n) Σ xᵢⱼ (rᵢ + xᵢⱼ βⱼ) — the "raw" correlation with the partial residual. */
  rho: number;
  /** (2/n) Σ xᵢⱼ² */
  denom: number;
  /** ρ / denom — what plain least squares would put here. */
  rawUpdate: number;
  threshold: number;
  after: number;
  zeroed: boolean;
}

/** Mean squared error of β on (X, y). β index 0 is the intercept. */
export function matrixMse(X: number[][], y: number[], beta: number[]): number {
  const n = X.length;
  if (n === 0) return Number.NaN;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let pred = beta[0] ?? 0;
    const row = X[i]!;
    for (let j = 0; j < row.length; j++) pred += (beta[j + 1] ?? 0) * row[j]!;
    sum += (y[i]! - pred) ** 2;
  }
  return sum / n;
}

/** λ Σⱼ₌₁ |βⱼ| — the intercept is excluded. */
export function l1Penalty(beta: number[], lambda: number): number {
  let sum = 0;
  for (let j = 1; j < beta.length; j++) sum += Math.abs(beta[j]!);
  return lambda * sum;
}

export const l1Magnitude = (beta: number[]): number => l1Penalty(beta, 1);

export const activeCount = (beta: number[]): number =>
  beta.slice(1).filter((b) => b !== 0).length;

export interface LassoObjective {
  mse: number;
  penalty: number;
  objective: number;
}

export function lassoObjective(
  X: number[][],
  y: number[],
  beta: number[],
  lambda: number,
): LassoObjective {
  const mse = matrixMse(X, y, beta);
  const penalty = l1Penalty(beta, lambda);
  return { mse, penalty, objective: mse + penalty };
}

/** Residuals rᵢ = yᵢ − ŷᵢ for the current β. */
export function residuals(X: number[][], y: number[], beta: number[]): number[] {
  return X.map((row, i) => {
    let pred = beta[0] ?? 0;
    for (let j = 0; j < row.length; j++) pred += (beta[j + 1] ?? 0) * row[j]!;
    return y[i]! - pred;
  });
}

/** Intercept update: the mean of what the other features leave unexplained. */
export function updateIntercept(X: number[][], y: number[], beta: number[]): number {
  const n = X.length;
  if (n === 0) return beta[0] ?? 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const row = X[i]!;
    let withoutIntercept = 0;
    for (let j = 0; j < row.length; j++) withoutIntercept += (beta[j + 1] ?? 0) * row[j]!;
    sum += y[i]! - withoutIntercept;
  }
  return sum / n;
}

/** One coordinate update for βⱼ (j ≥ 1). Returns the full arithmetic, for display. */
export function coordinateUpdate(
  X: number[][],
  y: number[],
  beta: number[],
  j: number,
  lambda: number,
): CoordinateUpdate {
  const n = X.length;
  const before = beta[j] ?? 0;
  const r = residuals(X, y, beta);
  let rho = 0;
  let denom = 0;
  for (let i = 0; i < n; i++) {
    const xij = X[i]![j - 1]!;
    rho += xij * (r[i]! + xij * before);
    denom += xij * xij;
  }
  rho = (2 / n) * rho;
  denom = (2 / n) * denom;
  if (!Number.isFinite(denom) || denom < 1e-14) {
    return { j, before, rho, denom, rawUpdate: before, threshold: lambda, after: before, zeroed: before === 0 };
  }
  const after = softThreshold(rho, lambda) / denom;
  return {
    j,
    before,
    rho,
    denom,
    rawUpdate: rho / denom,
    threshold: lambda,
    after,
    zeroed: after === 0,
  };
}

export interface HistoryEntry {
  iteration: number;
  beta: number[];
  mse: number;
  penalty: number;
  objective: number;
  active: number;
}

export interface LassoRun {
  beta: number[];
  history: HistoryEntry[];
  sweeps: number;
  converged: boolean;
  ok: boolean;
  reason?: string;
}

const unstable = (beta: number[]) =>
  beta.some((b) => !Number.isFinite(b) || Math.abs(b) > 1e12);

/** One full sweep: intercept, then every coefficient in order. */
export function sweepOnce(
  X: number[][],
  y: number[],
  beta: number[],
  lambda: number,
): { beta: number[]; maxChange: number } {
  const next = beta.slice();
  let maxChange = 0;
  const b0 = updateIntercept(X, y, next);
  maxChange = Math.max(maxChange, Math.abs(b0 - (next[0] ?? 0)));
  next[0] = b0;
  for (let j = 1; j < next.length; j++) {
    const update = coordinateUpdate(X, y, next, j, lambda);
    maxChange = Math.max(maxChange, Math.abs(update.after - update.before));
    next[j] = update.after;
  }
  return { beta: next, maxChange };
}

export function runCoordinateDescent(
  X: number[][],
  y: number[],
  lambda: number,
  options: { maxSweeps?: number; tol?: number; start?: number[] } = {},
): LassoRun {
  const maxSweeps = options.maxSweeps ?? 400;
  const tol = options.tol ?? 1e-9;
  const p = X[0]?.length ?? 0;
  let beta = options.start?.slice() ?? new Array(p + 1).fill(0);
  if (X.length === 0) {
    return { beta, history: [], sweeps: 0, converged: false, ok: false, reason: "There are no observations to fit." };
  }
  const history: HistoryEntry[] = [];
  const record = (iteration: number, b: number[]) => {
    const { mse, penalty, objective } = lassoObjective(X, y, b, lambda);
    history.push({ iteration, beta: b.slice(), mse, penalty, objective, active: activeCount(b) });
  };
  record(0, beta);

  let converged = false;
  let sweeps = 0;
  for (let s = 1; s <= maxSweeps; s++) {
    const result = sweepOnce(X, y, beta, lambda);
    beta = result.beta;
    sweeps = s;
    record(s, beta);
    if (unstable(beta)) {
      return { beta, history, sweeps, converged: false, ok: false, reason: LASSO_INSTABILITY_MESSAGE };
    }
    if (result.maxChange < tol) {
      converged = true;
      break;
    }
  }
  return converged
    ? { beta, history, sweeps, converged: true, ok: true }
    : { beta, history, sweeps, converged: false, ok: true, reason: LASSO_NOT_CONVERGED_MESSAGE };
}
