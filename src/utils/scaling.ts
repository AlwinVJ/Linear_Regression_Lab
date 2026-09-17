/**
 * Feature standardisation for the polynomial columns x, x², x³ …
 *
 *   zⱼ = (xʲ − meanⱼ) / stdⱼ
 *
 * Statistics are computed from the TRAINING observations only — test data is
 * never used to fit the scaler. Because a standardised fit is still a linear
 * model in the same features, the learned coefficients can be converted back
 * into ordinary polynomial coefficients:
 *
 *   ŷ = b₀ + Σ bⱼ (xʲ − meanⱼ)/stdⱼ
 *     = (b₀ − Σ bⱼ meanⱼ/stdⱼ) + Σ (bⱼ/stdⱼ) xʲ
 */

import type { Point } from "@/algorithms/linearRegression";

export interface Scaling {
  /** index 0 corresponds to the feature x¹ */
  means: number[];
  stds: number[];
}

export function computeScaling(points: Point[], degree: number): Scaling {
  const means: number[] = [];
  const stds: number[] = [];
  const n = Math.max(points.length, 1);
  for (let j = 1; j <= degree; j++) {
    let sum = 0;
    for (const p of points) sum += p.x ** j;
    const mean = sum / n;
    let varSum = 0;
    for (const p of points) varSum += (p.x ** j - mean) ** 2;
    const std = Math.sqrt(varSum / n);
    means.push(mean);
    stds.push(std > 1e-12 && Number.isFinite(std) ? std : 1);
  }
  return { means, stds };
}

/** Feature row in fitting space: [1, f₁ … f_d], standardised when a scaling is given. */
export function fitRow(x: number, degree: number, scaling: Scaling | null): number[] {
  const row = [1];
  for (let j = 1; j <= degree; j++) {
    const raw = x ** j;
    row.push(scaling ? (raw - scaling.means[j - 1]!) / scaling.stds[j - 1]! : raw);
  }
  return row;
}

/** Convert coefficients learned in standardised space back to plain polynomial coefficients. */
export function toRawCoefficients(fitBeta: number[], scaling: Scaling | null): number[] {
  if (!scaling) return fitBeta.slice();
  const out = fitBeta.slice();
  let intercept = fitBeta[0] ?? 0;
  for (let j = 1; j < fitBeta.length; j++) {
    const mean = scaling.means[j - 1] ?? 0;
    const std = scaling.stds[j - 1] ?? 1;
    out[j] = fitBeta[j]! / std;
    intercept -= (fitBeta[j]! * mean) / std;
  }
  out[0] = intercept;
  return out;
}
