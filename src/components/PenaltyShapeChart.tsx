import { fmt } from "@/utils/calculations";

const W = 520;
const H = 280;
const PAD = { top: 18, right: 18, bottom: 40, left: 46 };
const RANGE = 3;

/** |β| against β² — the actual functions, sampled. */
export function PenaltyShapeChart() {
  const sx = (v: number) => PAD.left + ((v + RANGE) / (2 * RANGE)) * (W - PAD.left - PAD.right);
  const maxY = RANGE ** 2;
  const sy = (v: number) => H - PAD.bottom - (v / maxY) * (H - PAD.top - PAD.bottom);

  const samples = Array.from({ length: 121 }, (_, i) => -RANGE + (2 * RANGE * i) / 120);
  const line = (f: (b: number) => number) =>
    samples.map((b) => `${sx(b).toFixed(1)},${sy(f(b)).toFixed(1)}`).join(" ");

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        role="img"
        aria-label="L1 and L2 penalty shapes against coefficient value"
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
          x1={sx(0)}
          x2={sx(0)}
          y1={PAD.top}
          y2={H - PAD.bottom}
          className="stroke-axis"
          strokeWidth={1.5}
        />
        <polyline points={line((b) => b * b)} fill="none" className="stroke-axis" strokeWidth={2.5} strokeDasharray="6 4" />
        <polyline points={line(Math.abs)} fill="none" className="stroke-line" strokeWidth={2.5} />
        {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
          <text
            key={t}
            x={sx(t)}
            y={H - PAD.bottom + 16}
            textAnchor="middle"
            className="fill-muted-foreground text-[11px]"
          >
            {t}
          </text>
        ))}
        {[0, 3, 6, 9].map((t) => (
          <text
            key={`y${t}`}
            x={PAD.left - 8}
            y={sy(t) + 4}
            textAnchor="end"
            className="fill-muted-foreground text-[11px]"
          >
            {fmt(t, 0)}
          </text>
        ))}
        <text
          x={(W + PAD.left) / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Coefficient β
        </text>
        <text
          x={-H / 2}
          y={14}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Penalty
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-line" /> Lasso: |β| grows linearly
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-axis" /> Ridge: β² grows quadratically
        </span>
      </div>
    </div>
  );
}
