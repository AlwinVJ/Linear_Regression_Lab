import { lambdaAxisPosition, lambdaLabel } from "@/utils/regularization";

export interface SparsityRow {
  lambda: number;
  active: number;
}

const W = 640;
const H = 260;
const PAD = { top: 20, right: 24, bottom: 46, left: 56 };

/** Number of non-zero coefficients against λ, computed from real fits. */
export function SparsityChart({
  rows,
  total,
  selectedLambda,
}: {
  rows: SparsityRow[];
  total: number;
  selectedLambda: number;
}) {
  const sx = (l: number) => PAD.left + lambdaAxisPosition(l) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    H - PAD.bottom - (v / Math.max(total, 1)) * (H - PAD.top - PAD.bottom);

  const steps = rows
    .map((r, i) => {
      const prev = rows[i - 1];
      const start = prev ? `${sx(r.lambda).toFixed(1)},${sy(prev.active).toFixed(1)} ` : "";
      return `${start}${sx(r.lambda).toFixed(1)},${sy(r.active).toFixed(1)}`;
    })
    .join(" ");

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Number of non-zero coefficients against regularization strength"
      >
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={H - PAD.bottom}
          y2={H - PAD.bottom}
          className="stroke-axis"
          strokeWidth={1.5}
        />
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
        <polyline points={steps} fill="none" className="stroke-line" strokeWidth={2.5} />
        {rows.map((r) => (
          <circle
            key={r.lambda}
            cx={Number(sx(r.lambda).toFixed(2))}
            cy={Number(sy(r.active).toFixed(2))}
            r={3}
            className="fill-line"
          />
        ))}
        {Array.from({ length: total + 1 }, (_, i) => i)
          .filter((v) => total <= 8 || v % 2 === 0)
          .map((v) => (
            <text key={`y${v}`} x={PAD.left - 8} y={sy(v) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
              {v}
            </text>
          ))}
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
        <text
          x={(W + PAD.left) / 2}
          y={H - 8}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Regularization strength λ (log scale)
        </text>
        <text
          x={-H / 2}
          y={14}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Non-zero coefficients
        </text>
      </svg>
      <p className="mt-2 text-sm text-muted-foreground">
        Counted from the actual fits at each λ. Stronger regularization <em>may</em> produce sparser
        models, but the shape of this line depends on the dataset.
      </p>
    </div>
  );
}
