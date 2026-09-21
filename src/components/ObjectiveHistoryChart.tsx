import type { HistoryEntry } from "@/optimization/coordinateDescent";
import { fmt } from "@/utils/calculations";

const W = 640;
const H = 260;
const PAD = { top: 18, right: 24, bottom: 42, left: 66 };

/** Lasso objective (and its two parts) against coordinate-descent iteration. */
export function ObjectiveHistoryChart({ history }: { history: HistoryEntry[] }) {
  const rows = history.filter((h) => Number.isFinite(h.objective));
  if (rows.length < 2) {
    return <p className="text-sm text-muted-foreground">Run at least one sweep to see the history.</p>;
  }
  const maxIter = rows[rows.length - 1]!.iteration || 1;
  const maxVal = Math.max(...rows.map((r) => r.objective), 1e-9);

  const sx = (i: number) => PAD.left + (i / maxIter) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - (Math.min(v, maxVal) / maxVal) * (H - PAD.top - PAD.bottom);
  const line = (pick: (h: HistoryEntry) => number) =>
    rows.map((h) => `${sx(h.iteration).toFixed(1)},${sy(pick(h)).toFixed(1)}`).join(" ");

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="Lasso objective against iteration"
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
        <polyline points={line((h) => h.mse)} fill="none" className="stroke-point" strokeWidth={2} strokeDasharray="5 4" />
        <polyline points={line((h) => h.penalty)} fill="none" className="stroke-resid-neg" strokeWidth={2} strokeDasharray="2 4" />
        <polyline points={line((h) => h.objective)} fill="none" className="stroke-line" strokeWidth={2.5} />
        <text x={PAD.left - 8} y={sy(maxVal) + 10} textAnchor="end" className="fill-muted-foreground text-[11px]">
          {fmt(maxVal, 3)}
        </text>
        <text x={PAD.left - 8} y={sy(0) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
          0
        </text>
        <text x={PAD.left} y={H - PAD.bottom + 18} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          0
        </text>
        <text x={W - PAD.right} y={H - PAD.bottom + 18} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          {maxIter}
        </text>
        <text
          x={(W + PAD.left) / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Coordinate-descent sweep
        </text>
        <text
          x={-H / 2}
          y={14}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Value
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-line" /> Lasso objective
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-point" /> MSE
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-resid-neg" /> L1 penalty
        </span>
      </div>
    </div>
  );
}
