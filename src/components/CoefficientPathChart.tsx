import { fmt } from "@/utils/calculations";
import { betaLabel } from "@/utils/polynomialFeatures";
import { lambdaAxisPosition, lambdaLabel } from "@/utils/regularization";

export interface PathRow {
  lambda: number;
  /** fitting-space coefficients, index 0 = intercept */
  coefficients: number[];
  ok: boolean;
}

const W = 640;
const H = 340;
const PAD = { top: 20, right: 90, bottom: 46, left: 66 };

const TONES = [
  "stroke-line",
  "stroke-point",
  "stroke-resid-neg",
  "stroke-resid-pos",
  "stroke-axis",
];

/** Coefficient paths: how each βⱼ (j ≥ 1) moves as λ grows. β₀ is excluded — it is not penalised. */
export function CoefficientPathChart({
  rows,
  degree,
  selectedLambda,
}: {
  rows: PathRow[];
  degree: number;
  selectedLambda: number;
}) {
  const values = rows.flatMap((r) => (r.ok ? r.coefficients.slice(1) : [])).filter(Number.isFinite);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-6);
  const sx = (l: number) => PAD.left + lambdaAxisPosition(l) * (W - PAD.left - PAD.right);
  const sy = (v: number) => {
    const clamped = Math.min(Math.max(v, -maxAbs), maxAbs);
    return (
      PAD.top +
      ((maxAbs - clamped) / (2 * maxAbs)) * (H - PAD.top - PAD.bottom)
    );
  };

  const paths = Array.from({ length: degree }, (_, i) => i + 1).map((j) => ({
    j,
    points: rows
      .filter((r) => r.ok && Number.isFinite(r.coefficients[j] ?? NaN))
      .map((r) => `${sx(r.lambda).toFixed(1)},${sy(r.coefficients[j]!).toFixed(1)}`)
      .join(" "),
    last: rows.filter((r) => r.ok)[rows.filter((r) => r.ok).length - 1]?.coefficients[j] ?? 0,
  }));

  const ticks = rows.filter((_, i) => i % 3 === 0);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Coefficient values against regularization strength"
      >
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={sy(0)}
          y2={sy(0)}
          className="stroke-axis"
          strokeWidth={1.5}
        />
        <line
          x1={PAD.left}
          x2={PAD.left}
          y1={PAD.top}
          y2={H - PAD.bottom}
          className="stroke-axis"
          strokeWidth={1.5}
        />
        <line
          x1={sx(selectedLambda)}
          x2={sx(selectedLambda)}
          y1={PAD.top}
          y2={H - PAD.bottom}
          className="stroke-accent"
          strokeWidth={2}
          strokeDasharray="4 4"
        />
        {paths.map((p, i) => (
          <polyline
            key={p.j}
            points={p.points}
            fill="none"
            className={TONES[i % TONES.length]}
            strokeWidth={2}
          />
        ))}
        {paths.map((p, i) => (
          <text
            key={`l${p.j}`}
            x={W - PAD.right + 8}
            y={sy(p.last) + 4 + (i % 2 === 0 ? 0 : 12)}
            className="fill-muted-foreground text-[11px]"
          >
            {betaLabel(p.j)}
          </text>
        ))}
        {ticks.map((r) => (
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
        <text
          x={PAD.left - 10}
          y={sy(maxAbs) + 10}
          textAnchor="end"
          className="fill-muted-foreground text-[11px]"
        >
          {fmt(maxAbs, 2)}
        </text>
        <text x={PAD.left - 10} y={sy(0) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          0
        </text>
        <text
          x={PAD.left - 10}
          y={sy(-maxAbs) - 2}
          textAnchor="end"
          className="fill-muted-foreground text-[11px]"
        >
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
        Each line is one coefficient, refitted at every λ. The dashed marker is your current λ. As λ
        grows the paths move toward zero — usually without ever reaching it. β₀ is not plotted
        because it is not penalised.
      </p>
    </div>
  );
}
