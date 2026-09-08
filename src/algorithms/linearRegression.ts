/**
 * Pure single-variable linear regression math.
 * No ML libraries — every number here is computed from the dataset.
 * Kept UI-free so later phases can reuse/extend it.
 */

export interface Point {
  id: string;
  x: number;
  y: number;
}

export interface ModelParams {
  intercept: number; // β₀
  slope: number; // β₁
}

export interface OlsSteps {
  n: number;
  meanX: number;
  meanY: number;
  deviations: { x: number; y: number; dx: number; dy: number; dxdy: number; dx2: number }[];
  numerator: number;
  denominator: number;
  slope: number;
  intercept: number;
}

export const mean = (values: number[]): number =>
  values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;

/** β₁ = Σ((xᵢ − x̄)(yᵢ − ȳ)) / Σ((xᵢ − x̄)²) */
export function calculateSlope(points: Point[]): number {
  if (points.length < 2) return 0;
  const meanX = mean(points.map((p) => p.x));
  const meanY = mean(points.map((p) => p.y));
  let numerator = 0;
  let denominator = 0;
  for (const p of points) {
    numerator += (p.x - meanX) * (p.y - meanY);
    denominator += (p.x - meanX) ** 2;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

/** β₀ = ȳ − β₁x̄ */
export function calculateIntercept(points: Point[], slope: number): number {
  return mean(points.map((p) => p.y)) - slope * mean(points.map((p) => p.x));
}

export function fitOLS(points: Point[]): ModelParams {
  const slope = calculateSlope(points);
  return { slope, intercept: calculateIntercept(points, slope) };
}

/** ŷ = β₀ + β₁x */
export function predict(params: ModelParams, x: number): number {
  return params.intercept + params.slope * x;
}

/** eᵢ = yᵢ − ŷᵢ */
export function calculateResiduals(points: Point[], params: ModelParams): number[] {
  return points.map((p) => p.y - predict(params, p.x));
}

/** MSE = (1/n) Σeᵢ² */
export function calculateMSE(points: Point[], params: ModelParams): number {
  if (points.length === 0) return 0;
  const residuals = calculateResiduals(points, params);
  return mean(residuals.map((e) => e ** 2));
}

export interface RowBreakdown {
  id: string;
  x: number;
  actual: number;
  prediction: number;
  error: number;
  squaredError: number;
}

export function buildBreakdown(points: Point[], params: ModelParams): RowBreakdown[] {
  return points.map((p) => {
    const prediction = predict(params, p.x);
    const error = p.y - prediction;
    return {
      id: p.id,
      x: p.x,
      actual: p.y,
      prediction,
      error,
      squaredError: error ** 2,
    };
  });
}

/** Full worked OLS walkthrough with the current dataset's real numbers. */
export function olsSteps(points: Point[]): OlsSteps {
  const meanX = mean(points.map((p) => p.x));
  const meanY = mean(points.map((p) => p.y));
  const deviations = points.map((p) => {
    const dx = p.x - meanX;
    const dy = p.y - meanY;
    return { x: p.x, y: p.y, dx, dy, dxdy: dx * dy, dx2: dx * dx };
  });
  const numerator = deviations.reduce((s, d) => s + d.dxdy, 0);
  const denominator = deviations.reduce((s, d) => s + d.dx2, 0);
  const slope = denominator === 0 ? 0 : numerator / denominator;
  return {
    n: points.length,
    meanX,
    meanY,
    deviations,
    numerator,
    denominator,
    slope,
    intercept: meanY - slope * meanX,
  };
}
