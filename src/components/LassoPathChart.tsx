import type { FeatureSpec } from "@/algorithms/featureRegression";
import { fmt } from "@/utils/calculations";
import { lambdaAxisPosition, lambdaLabel } from "@/utils/regularization";

export interface LassoPathRow {
  lambda: number;
  beta: number[]; // index 0 = intercept
  ok: boolean;
}

const W = 640;
const H = 360;
const PAD = { top: 20, right: 96, bottom: 46, left: 66 };

const TONES = [
  "stroke-line",
  "stroke-point",
  "stroke-resid-neg",
  "stroke-resid-pos",
  "stroke-axis",
];

/** Coefficient trajectories as λ grows. A hollow dot marks where a path hits exactly zero. */
export function LassoPathChart({
  rows,
  specs,
  selectedLambda,
}: {
  rows: LassoPathRow[];
  specs: FeatureSpec[];
  selectedLambda: number;
}) {
  const values = rows.flatMap((r) => (r.ok ? r.beta.slice(1) : [])).filter(Number.isFinite);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-6);
  const sx = (l: number) => PAD.left + lambdaAxisPosition(l) * (W - PAD.left - PAD.right);
  const sy = (v: number) => {
    const clamped = Math.min(Math.max(v, -maxAbs), maxAbs);
    return PAD.top + ((maxAbs - clamped) / (2 * maxAbs)) * (H - PAD.top - PAD.bottom);
  };

  const okRows = rows.filter((r) => r.ok);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Lasso coefficient paths against regularization strength"
      >
        <line x1={PAD.left} x2={W - PAD.right} y1={sy(0)} y2={sy(0)} className="stroke-axis" strokeWidth={1.5} />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} className="stroke-axis" strokeWidth={1.5} />
        <line
          x1={sx(selectedLambda)}
          x2={sx(selectedLambda)}
          y1={PAD.top}
          y2={H - PAD.bottom}
          className="stroke-accent"
          strokeWidth={2}
          strokeDasharray="4 4"
        />
        {specs.map((spec, i) => {
          const j = i + 1;
          const points = okRows
            .filter((r) => Number.isFinite(r.beta[j] ?? NaN))
            .map((r) => `${sx(r.lambda).toFixed(1)},${sy(r.beta[j]!).toFixed(1)}`)
            .join(" ");
          const last = okRows[okRows.length - 1]?.beta[j] ?? 0;
          return (
            <g key={spec.key}>
              <polyline points={points} fill="none" className={TONES[i % TONES.length]} strokeWidth={2} />
              {okRows
                .filter((r) => r.beta[j] === 0)
                .map((r) => (
                  <circle
                    key={`${spec.key}-${r.lambda}`}
                    cx={Number(sx(r.lambda).toFixed(2))}
                    cy={Number(sy(0).toFixed(2))}
                    r={3}
                    className="fill-background stroke-resid-neg"
                    strokeWidth={1.5}
                  />
                ))}
              <text
                x={W - PAD.right + 8}
                y={sy(last) + 4 + (i % 2 === 0 ? 0 : 11)}
                className="fill-muted-foreground text-[11px]"
              >
                {spec.label}
              </text>
            </g>
          );
        })}
        {rows
          .filter((_, i) => i % 3 === 0)
          .map((r) => (
            <text
              key={`t${r.lambda}`}
              x={sx(r.lambda)}
              y={H - PAD.bottom + 18}
              textAnchor="middle"
              className="fill-muted-foreground text-[10px]"
            >
              {lambdaLabel(r.lambda)}
            </text>
          ))}
        <text x={PAD.left - 10} y={sy(maxAbs) + 10} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(maxAbs, 2)}
        </text>
        <text x={PAD.left - 10} y={sy(0) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          0
        </text>
        <text x={PAD.left - 10} y={sy(-maxAbs) - 2} textAnchor="end" className="fill-muted-foreground text-[11px]">
          −{fmt(maxAbs, 2)}
        </text>
        <text
          x={(W + PAD.left - PAD.right) / 2}
          y={H - 8}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Regularization strength λ (log scale)
        </text>
        <text
          x={-H / 2}
          y={16}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Coefficient value
        </text>
      </svg>
      <p className="mt-2 text-sm text-muted-foreground">
        Each line is one feature, refitted at every λ. A hollow marker on the zero line means that
        coefficient is exactly zero at that λ — the feature has dropped out of the model. The dashed
        line is your current λ. β₀ is not drawn because it is not penalised.
      </p>
    </div>
  );
}
