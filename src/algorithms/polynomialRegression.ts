/**
 * Polynomial regression = feature expansion + ordinary linear least squares.
 *
 * Model:  ŷ = β₀ + β₁x + β₂x² + … + βₙxⁿ
 * It is LINEAR in β — only the features are non-linear in x.
 *
 * Fitting uses Householder QR on the design matrix (numerically far more
 * stable than forming and inverting XᵀX). No external libraries.
 */

import type { Point } from "@/algorithms/linearRegression";
import { designMatrix, expandRow } from "@/utils/polynomialFeatures";

export interface PolyFit {
  degree: number;
  coefficients: number[]; // index j is the coefficient of xʲ
  ok: boolean;
  reason?: string;
}

/** Solve min ‖Ab − y‖ via Householder QR. Returns null if the system is degenerate. */
export function solveLeastSquares(A: number[][], y: number[]): number[] | null {
  const m = A.length;
  if (m === 0) return null;
  const n = A[0]!.length;
  if (m < n) return null;

  // Working copies.
  const R = A.map((row) => row.slice());
  const b = y.slice();

  for (let k = 0; k < n; k++) {
    // Householder vector for column k.
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i]![k]! ** 2;
    norm = Math.sqrt(norm);
    if (!Number.isFinite(norm) || norm < 1e-14) return null;

    const alpha = R[k]![k]! > 0 ? -norm : norm;
    const v: number[] = new Array(m).fill(0);
    v[k] = R[k]![k]! - alpha;
    for (let i = k + 1; i < m; i++) v[i] = R[i]![k]!;

    let vNorm2 = 0;
    for (let i = k; i < m; i++) vNorm2 += v[i]! ** 2;
    if (vNorm2 < 1e-300) continue;

    // Apply reflector to R and b.
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += v[i]! * R[i]![j]!;
      const factor = (2 * dot) / vNorm2;
      for (let i = k; i < m; i++) {
        const row = R[i]!;
        row[j] = row[j]! - factor * v[i]!;
      }
    }
    let dotB = 0;
    for (let i = k; i < m; i++) dotB += v[i]! * b[i]!;
    const factorB = (2 * dotB) / vNorm2;
    for (let i = k; i < m; i++) b[i] = b[i]! - factorB * v[i]!;
  }

  // Back substitution on the upper-triangular n×n block.
  const beta: number[] = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i]!;
    for (let j = i + 1; j < n; j++) sum -= R[i]![j]! * beta[j]!;
    const pivot = R[i]![i]!;
    if (!Number.isFinite(pivot) || Math.abs(pivot) < 1e-12) return null;
    beta[i] = sum / pivot;
  }
  return beta.every(Number.isFinite) ? beta : null;
}

export function fitPolynomial(points: Point[], degree: number): PolyFit {
  const d = Math.max(0, Math.round(degree));
  if (points.length < d + 1) {
    return {
      degree: d,
      coefficients: new Array(d + 1).fill(0),
      ok: false,
      reason: `A degree ${d} model needs at least ${d + 1} points, but only ${points.length} are available.`,
    };
  }
  const X = designMatrix(points.map((p) => p.x), d);
  const beta = solveLeastSquares(X, points.map((p) => p.y));
  if (!beta) {
    return {
      degree: d,
      coefficients: new Array(d + 1).fill(0),
      ok: false,
      reason: "Numerical instability detected. Try a lower degree or a different learning rate.",
    };
  }
  return { degree: d, coefficients: beta, ok: true };
}

/** ŷ = Σ βⱼ xʲ — evaluated with Horner's rule for stability. */
export function predictPoly(coefficients: number[], x: number): number {
  let result = 0;
  for (let j = coefficients.length - 1; j >= 0; j--) result = result * x + coefficients[j]!;
  return result;
}

export function polyResiduals(points: Point[], coefficients: number[]): number[] {
  return points.map((p) => p.y - predictPoly(coefficients, p.x));
}

export function polyMSE(points: Point[], coefficients: number[]): number {
  if (points.length === 0) return Number.NaN;
  let sum = 0;
  for (const p of points) sum += (p.y - predictPoly(coefficients, p.x)) ** 2;
  return sum / points.length;
}

export interface PolyRow {
  id: string;
  x: number;
  actual: number;
  prediction: number;
  error: number;
  squaredError: number;
  features: number[]; // x, x², …
}

export function polyBreakdown(points: Point[], coefficients: number[], degree: number): PolyRow[] {
  return points.map((p) => {
    const prediction = predictPoly(coefficients, p.x);
    const error = p.y - prediction;
    return {
      id: p.id,
      x: p.x,
      actual: p.y,
      prediction,
      error,
      squaredError: error ** 2,
      features: expandRow(p.x, degree).slice(1),
    };
  });
}

/* ------------------------------------------------------------------ */
/* Gradient descent over polynomial features                           */
/* ------------------------------------------------------------------ */

/** ∂MSE/∂βⱼ = −(2/n) Σ xᵢʲ (yᵢ − ŷᵢ) */
export function polyGradients(points: Point[], coefficients: number[]): number[] {
  const n = points.length;
  const grads = new Array(coefficients.length).fill(0);
  if (n === 0) return grads;
  for (const p of points) {
    const residual = p.y - predictPoly(coefficients, p.x);
    let power = 1;
    for (let j = 0; j < coefficients.length; j++) {
      grads[j] += power * residual;
      power *= p.x;
    }
  }
  return grads.map((g) => (-2 / n) * g);
}

export interface PolyGdStep {
  before: number[];
  after: number[];
  gradients: number[];
  mseBefore: number;
  mseAfter: number;
  learningRate: number;
}

export function polyGradientDescentStep(
  points: Point[],
  coefficients: number[],
  learningRate: number,
): PolyGdStep {
  const gradients = polyGradients(points, coefficients);
  const after = coefficients.map((b, j) => b - learningRate * gradients[j]!);
  return {
    before: coefficients,
    after,
    gradients,
    mseBefore: polyMSE(points, coefficients),
    mseAfter: polyMSE(points, after),
    learningRate,
  };
}

export const polyDiverging = (coefficients: number[], mse: number): boolean =>
  !Number.isFinite(mse) || mse > 1e12 || coefficients.some((b) => !Number.isFinite(b) || Math.abs(b) > 1e8);

/** Nicely formatted dynamic equation, skipping terms whose coefficient rounds to zero. */
export function polyEquation(coefficients: number[], digits = 3): string {
  const labels = ["", "x", "x²", "x³", "x⁴", "x⁵", "x⁶", "x⁷", "x⁸", "x⁹", "x¹⁰"];
  const parts: string[] = [];
  coefficients.forEach((b, j) => {
    const rounded = Number(b.toFixed(digits));
    if (rounded === 0 && j !== 0) return;
    const magnitude = Math.abs(rounded).toFixed(digits);
    const label = labels[j] ?? `x^${j}`;
    const term = j === 0 ? magnitude : `${magnitude}${label}`;
    if (parts.length === 0) parts.push(`${rounded < 0 ? "−" : ""}${term}`);
    else parts.push(`${rounded < 0 ? "−" : "+"} ${term}`);
  });
  if (parts.length === 0) return "ŷ = 0";
  return `ŷ = ${parts.join(" ")}`;
}
