/**
 * Polynomial feature expansion.
 * x  →  [1, x, x², x³, ... xᵈ]
 * Nothing here is ML-library magic: it is repeated multiplication.
 */

/** Feature row including the constant term (index 0 = 1, index j = xʲ). */
export function expandRow(x: number, degree: number): number[] {
  const row: number[] = [];
  for (let j = 0; j <= degree; j++) row.push(x ** j);
  return row;
}

/** Feature columns WITHOUT the constant term — what the table shows: x, x², x³ … */
export function expandVisibleRow(x: number, degree: number): number[] {
  const row: number[] = [];
  for (let j = 1; j <= degree; j++) row.push(x ** j);
  return row;
}

/** Design matrix X where Xᵢⱼ = xᵢʲ (j = 0 … degree). */
export function designMatrix(xs: number[], degree: number): number[][] {
  return xs.map((x) => expandRow(x, degree));
}

export const POWER_LABELS = ["1", "x", "x²", "x³", "x⁴", "x⁵", "x⁶", "x⁷", "x⁸", "x⁹", "x¹⁰"];

export const powerLabel = (j: number): string => POWER_LABELS[j] ?? `x^${j}`;

export const betaLabel = (j: number): string => {
  const subs = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀"];
  return `β${subs[j] ?? `_${j}`}`;
};
