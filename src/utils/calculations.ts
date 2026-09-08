export const fmt = (value: number, digits = 2): string => {
  if (!Number.isFinite(value)) return "—";
  const rounded = Number(value.toFixed(digits));
  return Object.is(rounded, -0) ? (0).toFixed(digits) : rounded.toFixed(digits);
};

export const signed = (value: number, digits = 2): string =>
  `${value < 0 ? "−" : "+"}${fmt(Math.abs(value), digits)}`;

/** Equation string like "ŷ = 1.23 + 1.17x" */
export const equationString = (intercept: number, slope: number): string =>
  `ŷ = ${fmt(intercept)} ${slope < 0 ? "−" : "+"} ${fmt(Math.abs(slope))}x`;

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));
