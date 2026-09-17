import { fmt } from "@/utils/calculations";
import { betaLabel } from "@/utils/polynomialFeatures";

export interface CurvePoint {
  beta: number;
  mse: number;
  objective: number;
}

const W = 620;
const H = 300;
const PAD = { top: 20, right: 24, bottom: 46, left: 66 };

/**
 * One-parameter intuition: hold every other coefficient fixed and sweep one βⱼ.
 * The MSE curve is the ordinary objective; the Ridge curve adds λβⱼ² on top,
 * which pulls its minimum toward zero. Deliberately one-dimensional — a real
 * loss surface has as many dimensions as there are coefficients.
 */
export function RidgeLossCurve({
  curve,
  index,
  olsBeta,
  ridgeBeta,
}: {
  curve: CurvePoint[];
  index: number;
  olsBeta: number;
  ridgeBeta: number;
}) {
  const betas = curve.map((c) => c.beta);
  const vals = curve.flatMap((c) => [c.mse, c.objective]).filter(Number.isFinite);
  const b0 = Math.min(...betas);
  const b1 = Math.max(...betas);
  const v0 = 0;
  const v1 = Math.max(...vals, 1e-9);
  const sx = (b: number) => PAD.left + ((b - b0) / (b1 - b0 || 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    H - PAD.bottom - ((Math.min(v, v1) - v0) / (v1 - v0 || 1)) * (H - PAD.top - PAD.bottom);

  const poly = (pick: (c: CurvePoint) => number) =>
    curve
      .filter((c) => Number.isFinite(pick(c)))
      .map((c) => `${sx(c.beta).toFixed(1)},${sy(pick(c)).toFixed(1)}`)
      .join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img" aria-label="Loss against a single coefficient">
        <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} className="stroke-axis" strokeWidth={1.5} />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} className="stroke-axis" strokeWidth={1.5} />
        <line x1={sx(0)} x2={sx(0)} y1={PAD.top} y2={H - PAD.bottom} className="stroke-grid" strokeWidth={1} />
        <polyline points={poly((c) => c.mse)} fill="none" className="stroke-axis" strokeWidth={2} strokeDasharray="5 4" />
        <polyline points={poly((c) => c.objective)} fill="none" className="stroke-line" strokeWidth={2.5} />
        <circle cx={sx(olsBeta)} cy={sy(curve.find((c) => c.beta >= olsBeta)?.mse ?? 0)} r={5} className="fill-axis" />
        <circle cx={sx(ridgeBeta)} cy={sy(curve.find((c) => c.beta >= ridgeBeta)?.objective ?? 0)} r={5} className="fill-point" />
        <text x={(W + PAD.left) / 2} y={H - 8} textAnchor="middle" className="fill-foreground text-[12px] font-medium">
          {betaLabel(index)} value
        </text>
        <text x={-H / 2} y={16} transform="rotate(-90)" textAnchor="middle" className="fill-foreground text-[12px] font-medium">
          Loss
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-axis" /> MSE only (OLS objective), minimum at{" "}
          {fmt(olsBeta, 3)}
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-line" /> MSE + λβ², minimum at {fmt(ridgeBeta, 3)}
        </span>
      </div>
    </div>
  );
}
