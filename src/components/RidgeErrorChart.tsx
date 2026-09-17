import { fmt } from "@/utils/calculations";
import { lambdaAxisPosition, lambdaLabel } from "@/utils/regularization";

export interface ErrorRow {
  lambda: number;
  trainMse: number;
  testMse: number | null;
  ok: boolean;
}

const W = 640;
const H = 320;
const PAD = { top: 20, right: 24, bottom: 46, left: 70 };

/** Training and test error against λ, on a log error axis. */
export function RidgeErrorChart({
  rows,
  selectedLambda,
  showTest,
  onSelectLambda,
}: {
  rows: ErrorRow[];
  selectedLambda: number;
  showTest: boolean;
  onSelectLambda: (l: number) => void;
}) {
  const values = rows
    .flatMap((r) => [r.trainMse, ...(showTest && r.testMse !== null ? [r.testMse] : [])])
    .filter((v) => Number.isFinite(v) && v > 0);
  const maxRaw = Math.max(...values, 1e-9);
  const minRaw = Math.max(Math.min(...values, maxRaw), maxRaw / 1e6);
  const lo = Math.log10(minRaw) - 0.2;
  const hi = Math.log10(maxRaw) + 0.2;

  const sx = (l: number) => PAD.left + lambdaAxisPosition(l) * (W - PAD.left - PAD.right);
  const sy = (v: number) => {
    const clamped = Math.min(Math.max(v, minRaw), maxRaw);
    const t = (Math.log10(clamped) - lo) / (hi - lo);
    return H - PAD.bottom - t * (H - PAD.top - PAD.bottom);
  };

  const line = (pick: (r: ErrorRow) => number | null) =>
    rows
      .map((r) => {
        const v = pick(r);
        if (v === null || !Number.isFinite(v) || v <= 0 || !r.ok) return null;
        return `${sx(r.lambda).toFixed(1)},${sy(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Training and test error against regularization strength"
      >
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={H - PAD.bottom}
          y2={H - PAD.bottom}
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
        <polyline points={line((r) => r.trainMse)} fill="none" className="stroke-line" strokeWidth={2.5} />
        {showTest && (
          <polyline
            points={line((r) => r.testMse)}
            fill="none"
            className="stroke-resid-neg"
            strokeWidth={2.5}
            strokeDasharray="6 4"
          />
        )}
        {rows.map((r) => (
          <g key={r.lambda}>
            <rect
              x={sx(r.lambda) - 10}
              y={PAD.top}
              width={20}
              height={H - PAD.top - PAD.bottom}
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label={`Select lambda ${lambdaLabel(r.lambda)}`}
              onClick={() => onSelectLambda(r.lambda)}
            />
            {r.ok && Number.isFinite(r.trainMse) && r.trainMse > 0 && (
              <circle cx={sx(r.lambda)} cy={sy(r.trainMse)} r={3.5} className="fill-line" />
            )}
            {showTest && r.ok && r.testMse !== null && r.testMse > 0 && Number.isFinite(r.testMse) && (
              <circle cx={sx(r.lambda)} cy={sy(r.testMse)} r={3.5} className="fill-resid-neg" />
            )}
          </g>
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
        <text x={PAD.left - 10} y={sy(maxRaw) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(maxRaw, 2)}
        </text>
        <text x={PAD.left - 10} y={sy(minRaw) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(minRaw, 4)}
        </text>
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
          y={16}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          MSE (log scale)
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-line" /> training MSE
        </span>
        {showTest && (
          <span className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-5 bg-resid-neg" /> test MSE
          </span>
        )}
        <span>Click anywhere on the chart to jump to that λ.</span>
      </div>
    </div>
  );
}
