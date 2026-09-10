/**
 * Manual gradient descent for single-variable linear regression.
 * Every number the UI shows comes from these functions — nothing is faked.
 *
 * Model:     ŷ = β₀ + β₁x
 * Loss:      MSE = (1/n) Σ(yᵢ − ŷᵢ)²
 * Gradients: ∂MSE/∂β₀ = -(2/n) Σ(yᵢ − ŷᵢ)
 *            ∂MSE/∂β₁ = -(2/n) Σ xᵢ(yᵢ − ŷᵢ)
 * Update:    β = β − α × gradient
 */

import {
  calculateMSE,
  calculateResiduals,
  predict,
  type ModelParams,
  type Point,
} from "@/algorithms/linearRegression";

export { predict, calculateMSE, calculateResiduals };

export interface Gradients {
  gradientIntercept: number; // ∂MSE/∂β₀
  gradientSlope: number; // ∂MSE/∂β₁
  residualSum: number; // Σ(yᵢ − ŷᵢ)
  weightedResidualSum: number; // Σ xᵢ(yᵢ − ŷᵢ)
  n: number;
}

export interface HistoryEntry {
  iteration: number;
  intercept: number;
  slope: number;
  mse: number;
  gradientIntercept: number;
  gradientSlope: number;
}

export interface StepResult {
  before: ModelParams;
  after: ModelParams;
  mseBefore: number;
  mseAfter: number;
  gradients: Gradients;
  learningRate: number;
}

export function calculateGradients(points: Point[], params: ModelParams): Gradients {
  const n = points.length;
  if (n === 0) {
    return {
      gradientIntercept: 0,
      gradientSlope: 0,
      residualSum: 0,
      weightedResidualSum: 0,
      n: 0,
    };
  }
  let residualSum = 0;
  let weightedResidualSum = 0;
  for (const p of points) {
    const residual = p.y - predict(params, p.x);
    residualSum += residual;
    weightedResidualSum += p.x * residual;
  }
  return {
    gradientIntercept: (-2 / n) * residualSum,
    gradientSlope: (-2 / n) * weightedResidualSum,
    residualSum,
    weightedResidualSum,
    n,
  };
}

export function updateParameters(
  params: ModelParams,
  gradients: Gradients,
  learningRate: number,
): ModelParams {
  return {
    intercept: params.intercept - learningRate * gradients.gradientIntercept,
    slope: params.slope - learningRate * gradients.gradientSlope,
  };
}

/** One full gradient descent iteration, with every intermediate value kept. */
export function gradientDescentStep(
  points: Point[],
  params: ModelParams,
  learningRate: number,
): StepResult {
  const gradients = calculateGradients(points, params);
  const after = updateParameters(params, gradients, learningRate);
  return {
    before: params,
    after,
    mseBefore: calculateMSE(points, params),
    mseAfter: calculateMSE(points, after),
    gradients,
    learningRate,
  };
}

export const isDiverging = (params: ModelParams, mse: number): boolean =>
  !Number.isFinite(mse) ||
  !Number.isFinite(params.slope) ||
  !Number.isFinite(params.intercept) ||
  Math.abs(params.slope) > 1e6 ||
  Math.abs(params.intercept) > 1e6 ||
  mse > 1e12;

export type ConvergenceStatus = "learning" | "converging" | "converged" | "diverged";

export interface Convergence {
  status: ConvergenceStatus;
  lossChange: number;
  label: string;
  detail: string;
}

const TOL = 1e-6;

export function assessConvergence(history: HistoryEntry[], diverged: boolean): Convergence {
  if (diverged) {
    return {
      status: "diverged",
      lossChange: Number.NaN,
      label: "Unstable",
      detail: "The numbers blew up. The learning rate may be too large. Try reducing it.",
    };
  }
  if (history.length < 2) {
    return {
      status: "learning",
      lossChange: Number.NaN,
      label: "Learning",
      detail: "Take a few steps to see how quickly the loss falls.",
    };
  }
  const last = history[history.length - 1]!;
  const prev = history[history.length - 2]!;
  const lossChange = Math.abs(prev.mse - last.mse);
  const gradMagnitude = Math.abs(last.gradientIntercept) + Math.abs(last.gradientSlope);
  if (lossChange < TOL && gradMagnitude < 1e-3) {
    return {
      status: "converged",
      lossChange,
      label: "Converged",
      detail: "The loss has stopped changing in any meaningful way. The line is as good as it gets.",
    };
  }
  if (lossChange < 1e-3) {
    return {
      status: "converging",
      lossChange,
      label: "Converging",
      detail: "The steps are getting small — the model is settling near the minimum.",
    };
  }
  return {
    status: "learning",
    lossChange,
    label: "Learning",
    detail: "The loss is still falling noticeably with each step.",
  };
}

/** Loss as a function of one parameter, holding the other fixed. Used for the loss landscape. */
export function lossCurveOverSlope(
  points: Point[],
  intercept: number,
  from: number,
  to: number,
  samples = 90,
): { slope: number; mse: number }[] {
  const out: { slope: number; mse: number }[] = [];
  for (let i = 0; i <= samples; i++) {
    const slope = from + ((to - from) * i) / samples;
    out.push({ slope, mse: calculateMSE(points, { intercept, slope }) });
  }
  return out;
}

export const LEARNING_RATE_PRESETS = [
  { id: "small", label: "Small", alpha: 0.001, note: "Learning is progressing slowly because the steps are small." },
  { id: "medium", label: "Medium", alpha: 0.01, note: "The model is moving toward a minimum efficiently." },
  { id: "large", label: "Large", alpha: 0.1, note: "The steps are too large and the model may overshoot the minimum." },
] as const;
