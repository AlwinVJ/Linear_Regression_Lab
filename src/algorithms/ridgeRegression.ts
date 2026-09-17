/**
 * Ridge Regression = the same linear/polynomial model, a different objective.
 *
 *   Objective = MSE + λ Σⱼ βⱼ²      (j = 1 … p — the intercept β₀ is NOT penalised)
 *
 * Closed form: instead of forming (XᵀX + λI)⁻¹Xᵀy, we solve the equivalent
 * augmented least-squares problem
 *
 *   min ‖ [ X ; √(nλ)·P ] β − [ y ; 0 ] ‖²
 *
 * with P selecting the penalised columns. That reuses the numerically stable
 * Householder QR solver from Phase 3 and never inverts a matrix explicitly.
 * The √(nλ) factor is there because our MSE divides by n while the penalty
 * does not: minimising (1/n)‖Xβ − y‖² + λ‖β₁…‖² is the same as minimising
 * ‖Xβ − y‖² + nλ‖β₁…‖².
 */

import type { Point } from "@/algorithms/linearRegression";
import { solveLeastSquares } from "@/algorithms/polynomialRegression";
import { computeScaling, fitRow, toRawCoefficients, type Scaling } from "@/utils/scaling";

export const INSTABILITY_MESSAGE =
  "Numerical instability detected. Try reducing polynomial degree or enabling feature standardization.";

export interface RidgeFit {
  degree: number;
  lambda: number;
  /** Coefficients in the fitting space (standardised, when standardisation is on). */
  fitCoefficients: number[];
  /** Coefficients of the plain polynomial ŷ = β₀ + β₁x + β₂x² + … */
  coefficients: number[];
  scaling: Scaling | null;
  ok: boolean;
  reason?: string;
}

const failed = (degree: number, lambda: number, reason: string): RidgeFit => ({
  degree,
  lambda,
  fitCoefficients: new Array(degree + 1).fill(0),
  coefficients: new Array(degree + 1).fill(0),
  scaling: null,
  ok: false,
  reason,
});

/** Closed-form Ridge (λ = 0 reduces exactly to ordinary least squares). */
export function fitRidge(
  points: Point[],
  degree: number,
  lambda: number,
  standardize = false,
): RidgeFit {
  const d = Math.max(0, Math.round(degree));
  const lam = Math.max(0, lambda);
  const n = points.length;
  if (n === 0) return failed(d, lam, "There are no observations to fit.");
  if (lam === 0 && n < d + 1) {
    return failed(
      d,
      lam,
      `A degree ${d} model with no regularization needs at least ${d + 1} points, but only ${n} are available.`,
    );
  }

  const scaling = standardize && d > 0 ? computeScaling(points, d) : null;
  const X = points.map((p) => fitRow(p.x, d, scaling));
  const y = points.map((p) => p.y);

  // Augment with the penalty rows: √(nλ) on each penalised coefficient.
  const rows = X.map((r) => r.slice());
  const targets = y.slice();
  if (lam > 0) {
    const w = Math.sqrt(n * lam);
    for (let j = 1; j <= d; j++) {
      const row = new Array(d + 1).fill(0);
      row[j] = w;
      rows.push(row);
      targets.push(0);
    }
  }

  const fitBeta = solveLeastSquares(rows, targets);
  if (!fitBeta) return failed(d, lam, INSTABILITY_MESSAGE);

  const coefficients = toRawCoefficients(fitBeta, scaling);
  if (!coefficients.every(Number.isFinite) || coefficients.some((b) => Math.abs(b) > 1e12)) {
    return failed(d, lam, INSTABILITY_MESSAGE);
  }
  return { degree: d, lambda: lam, fitCoefficients: fitBeta, coefficients, scaling, ok: true };
}

/** Prediction from fitting-space coefficients. */
export function predictFit(
  fitCoefficients: number[],
  x: number,
  degree: number,
  scaling: Scaling | null,
): number {
  const row = fitRow(x, degree, scaling);
  let sum = 0;
  for (let j = 0; j < fitCoefficients.length; j++) sum += fitCoefficients[j]! * (row[j] ?? 0);
  return sum;
}

/** λ Σⱼ₌₁ βⱼ² — the intercept is excluded. */
export function l2Penalty(fitCoefficients: number[], lambda: number): number {
  let sum = 0;
  for (let j = 1; j < fitCoefficients.length; j++) sum += fitCoefficients[j]! ** 2;
  return lambda * sum;
}

/** Σⱼ₌₁ βⱼ² — the raw coefficient magnitude, independent of λ. */
export function coefficientEnergy(fitCoefficients: number[]): number {
  let sum = 0;
  for (let j = 1; j < fitCoefficients.length; j++) sum += fitCoefficients[j]! ** 2;
  return sum;
}

export interface RidgeObjective {
  mse: number;
  penalty: number;
  objective: number;
}

export function ridgeObjective(mse: number, fitCoefficients: number[], lambda: number): RidgeObjective {
  const penalty = l2Penalty(fitCoefficients, lambda);
  return { mse, penalty, objective: mse + penalty };
}

/* ------------------------------------------------------------------ */
/* Gradient descent on the Ridge objective                             */
/* ------------------------------------------------------------------ */

export interface RidgeGradient {
  /** ∂MSE/∂βⱼ = −(2/n) Σ fᵢⱼ (yᵢ − ŷᵢ) */
  mse: number;
  /** ∂(λΣβ²)/∂βⱼ = 2λβⱼ, and 0 for the intercept */
  regularization: number;
  total: number;
}

export function ridgeGradients(
  points: Point[],
  fitCoefficients: number[],
  degree: number,
  scaling: Scaling | null,
  lambda: number,
): RidgeGradient[] {
  const n = points.length;
  const mseGrads = new Array(fitCoefficients.length).fill(0);
  if (n > 0) {
    for (const p of points) {
      const row = fitRow(p.x, degree, scaling);
      const residual = p.y - predictFit(fitCoefficients, p.x, degree, scaling);
      for (let j = 0; j < fitCoefficients.length; j++) {
        mseGrads[j] += (row[j] ?? 0) * residual;
      }
    }
    for (let j = 0; j < mseGrads.length; j++) mseGrads[j] = (-2 / n) * mseGrads[j];
  }
  return fitCoefficients.map((b, j) => {
    const regularization = j === 0 ? 0 : 2 * lambda * b;
    return { mse: mseGrads[j]!, regularization, total: mseGrads[j]! + regularization };
  });
}

export interface RidgeGdStep {
  before: number[];
  after: number[];
  gradients: RidgeGradient[];
  learningRate: number;
}

export function ridgeGradientDescentStep(
  points: Point[],
  fitCoefficients: number[],
  degree: number,
  scaling: Scaling | null,
  lambda: number,
  learningRate: number,
): RidgeGdStep {
  const gradients = ridgeGradients(points, fitCoefficients, degree, scaling, lambda);
  const after = fitCoefficients.map((b, j) => b - learningRate * gradients[j]!.total);
  return { before: fitCoefficients, after, gradients, learningRate };
}

export const ridgeDiverging = (coefficients: number[], value: number): boolean =>
  !Number.isFinite(value) ||
  value > 1e12 ||
  coefficients.some((b) => !Number.isFinite(b) || Math.abs(b) > 1e8);
