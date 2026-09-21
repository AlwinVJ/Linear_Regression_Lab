/**
 * Soft thresholding — the operator that gives Lasso its exact zeros.
 *
 *   S(z, γ) = sign(z) · max(|z| − γ, 0)
 *
 * Anything smaller than γ in magnitude is pulled all the way to 0; anything
 * larger is moved γ closer to 0. This is the exact minimiser of
 *   ½(b − z)² + γ|b|
 * and it is what a coordinate update of the Lasso objective reduces to.
 */

export function softThreshold(z: number, gamma: number): number {
  if (!Number.isFinite(z)) return Number.NaN;
  const g = Math.max(0, gamma);
  const magnitude = Math.abs(z) - g;
  if (magnitude <= 0) return 0;
  return Math.sign(z) * magnitude;
}

/** Subgradient of |β| — a single value away from zero, an interval at zero. */
export function absSubgradient(beta: number): { low: number; high: number } {
  if (beta > 0) return { low: 1, high: 1 };
  if (beta < 0) return { low: -1, high: -1 };
  return { low: -1, high: 1 };
}
