/**
 * Regularization-strength helpers.
 * λ runs from 0 (no penalty) up to 1000 (very strong) on a log-ish ladder,
 * so a slider index maps to a useful range of strengths.
 */

/** λ = 0 first, then 10^-4 … 10^3 in quarter-decade steps. */
export const LAMBDA_STEPS: number[] = (() => {
  const out = [0];
  for (let e = -4; e <= 3.0001; e += 0.25) {
    out.push(Number((10 ** e).toPrecision(3)));
  }
  return out;
})();

/** λ values used for the coefficient-path and error-vs-λ sweeps. */
export const LAMBDA_PATH: number[] = LAMBDA_STEPS.filter((_, i) => i === 0 || i % 2 === 1);

export const lambdaLabel = (lambda: number): string => {
  if (lambda === 0) return "0";
  if (lambda >= 1000) return String(Math.round(lambda));
  if (lambda >= 1) return String(Number(lambda.toPrecision(3)));
  return lambda.toExponential(0).replace("e-", "e−");
};

export const nearestLambdaIndex = (lambda: number): number => {
  let best = 0;
  let bestDist = Infinity;
  LAMBDA_STEPS.forEach((v, i) => {
    const d = Math.abs(v - lambda);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
};

/** Position of a λ on a log axis where λ = 0 gets its own slot at the far left. */
export const lambdaAxisPosition = (lambda: number, min = 1e-4, max = 1e3): number => {
  if (lambda <= 0) return 0;
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const t = (Math.log10(Math.min(Math.max(lambda, min), max)) - lo) / (hi - lo);
  return 0.08 + 0.92 * t;
};
