import { fmt } from "@/utils/calculations";

export interface ComplexityRow {
  degree: number;
  trainMse: number;
  testMse: number | null;
  ok: boolean;
}

const W = 620;
const H = 340;
const PAD = { top: 24, right: 24, bottom: 46, left: 66 };

/** Model complexity (degree) versus error. Everything plotted is calculated, not drawn by hand. */
export function ComplexityChart({
  rows,
  selectedDegree,
  showTest,
  onSelectDegree,
}: {
  rows: ComplexityRow[];
  selectedDegree: number;
  showTest: boolean;
  onSelectDegree: (d: number) => void;
}) {
  const values = rows.flatMap((r) => [
    r.trainMse,
    ...(showTest && r.testMse !== null ? [r.testMse] : []),
  ]);
  const finite = values.filter((v) => Number.isFinite(v) && v >= 0);
  const maxRaw = Math.max(...finite, 1e-9);
  const minRaw = Math.max(Math.min(...finite, maxRaw), 1e-6);
  // Log scale: errors across degrees can span many orders of magnitude.
  const lo = Math.log10(minRaw) - 0.3;
  const hi = Math.log10(maxRaw) + 0.3;

  const degrees = rows.map((r) => r.degree);
  const dMin = Math.min(...degrees);
  const dMax = Math.max(...degrees);
  const sx = (d: number) =>
    PAD.left + ((d - dMin) / Math.max(dMax - dMin, 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => {
    const clamped = Math.min(Math.max(v, minRaw), maxRaw);
    const t = (Math.log10(clamped) - lo) / (hi - lo);
    return H - PAD.bottom - t * (H - PAD.top - PAD.bottom);
  };

  const line = (pick: (r: ComplexityRow) => number | null) =>
    rows
      .map((r) => {
        const v = pick(r);
        if (v === null || !Number.isFinite(v)) return null;
        return `${sx(r.degree).toFixed(1)},${sy(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Training and test error against polynomial degree"
      >
        {rows.map((r) => (
          <line
            key={`g${r.degree}`}
            x1={sx(r.degree)}
            x2={sx(r.degree)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            className="stroke-grid"
            strokeWidth={1}
          />
        ))}
        <rect
          x={sx(selectedDegree) - 12}
          y={PAD.top}
          width={24}
          height={H - PAD.top - PAD.bottom}
          className="fill-accent/20"
        />
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
          <g key={`p${r.degree}`}>
            {Number.isFinite(r.trainMse) && (
              <circle cx={sx(r.degree)} cy={sy(r.trainMse)} r={r.degree === selectedDegree ? 6 : 4} className="fill-line" />
            )}
            {showTest && r.testMse !== null && Number.isFinite(r.testMse) && (
              <circle
                cx={sx(r.degree)}
                cy={sy(r.testMse)}
                r={r.degree === selectedDegree ? 6 : 4}
                className="fill-resid-neg"
              />
            )}
            <rect
              x={sx(r.degree) - 12}
              y={PAD.top}
              width={24}
              height={H - PAD.top - PAD.bottom}
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label={`Select degree ${r.degree}`}
              onClick={() => onSelectDegree(r.degree)}
            />
            <text
              x={sx(r.degree)}
              y={H - PAD.bottom + 18}
              textAnchor="middle"
              className={
                r.degree === selectedDegree
                  ? "fill-foreground text-[11px] font-semibold"
                  : "fill-muted-foreground text-[11px]"
              }
            >
              {r.degree}
            </text>
          </g>
        ))}

        <text
          x={(W + PAD.left) / 2}
          y={H - 8}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Polynomial degree
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
        <text x={PAD.left - 10} y={sy(maxRaw) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(maxRaw, 2)}
        </text>
        <text x={PAD.left - 10} y={sy(minRaw) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(minRaw, 4)}
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
        <span>Click a degree to select it.</span>
      </div>
    </div>
  );
}
