import { fmt } from "@/utils/calculations";

export interface Series {
  key: string;
  label: string;
  values: number[];
  className: string;
  dashed?: boolean;
}

const W = 560;
const H = 260;
const PAD = { top: 18, right: 18, bottom: 38, left: 56 };

/**
 * Small SVG line chart used for "loss vs iteration" and "parameters vs iteration".
 * It only draws the numbers it is given — the values come from real gradient descent history.
 */
export function HistoryChart({
  series,
  xLabel,
  yLabel,
  emptyMessage = "Take a step to start plotting.",
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  emptyMessage?: string;
}) {
  const visible = series.filter((s) => s.values.length > 0);
  const maxLen = Math.max(0, ...visible.map((s) => s.values.length));

  if (maxLen < 1) {
    return (
      <div className="flex h-[200px] items-center justify-center rounded-md border border-dashed border-border text-sm text-muted-foreground">
        {emptyMessage}
      </div>
    );
  }

  const all = visible.flatMap((s) => s.values).filter(Number.isFinite);
  let minY = Math.min(...all);
  let maxY = Math.max(...all);
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
    minY = 0;
    maxY = 1;
  }
  if (maxY - minY < 1e-9) {
    maxY = minY + 1;
    minY = minY - 1;
  }
  const padY = (maxY - minY) * 0.1;
  minY -= padY;
  maxY += padY;
  const lastX = Math.max(1, maxLen - 1);

  const sx = (i: number) => PAD.left + (i / lastX) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    H - PAD.bottom - ((v - minY) / (maxY - minY)) * (H - PAD.top - PAD.bottom);

  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const xTicks = [0, Math.round(lastX / 2), lastX];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${yLabel} against ${xLabel}`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={sy(t)}
              y2={sy(t)}
              className="stroke-grid"
              strokeWidth={1}
            />
            <text x={PAD.left - 8} y={sy(t) + 4} textAnchor="end" className="fill-muted-foreground text-[10px]">
              {fmt(t, 2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text
            key={`x${t}`}
            x={sx(t)}
            y={H - PAD.bottom + 16}
            textAnchor="middle"
            className="fill-muted-foreground text-[10px]"
          >
            {t}
          </text>
        ))}
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

        {visible.map((s) => {
          const d = s.values
            .map((v, i) => `${i === 0 ? "M" : "L"}${sx(i)},${sy(Number.isFinite(v) ? v : minY)}`)
            .join(" ");
          const lastIdx = s.values.length - 1;
          const lastVal = s.values[lastIdx]!;
          return (
            <g key={s.key}>
              <path
                d={d}
                fill="none"
                className={s.className}
                strokeWidth={2}
                strokeDasharray={s.dashed ? "5 4" : undefined}
                strokeLinejoin="round"
              />
              {Number.isFinite(lastVal) && (
                <circle cx={sx(lastIdx)} cy={sy(lastVal)} r={3.5} className={s.className} fill="currentColor" />
              )}
            </g>
          );
        })}

        <text x={(W + PAD.left) / 2} y={H - 6} textAnchor="middle" className="fill-foreground text-[11px]">
          {xLabel}
        </text>
        <text x={-H / 2} y={13} transform="rotate(-90)" textAnchor="middle" className="fill-foreground text-[11px]">
          {yLabel}
        </text>
      </svg>
      <div className="mt-1 flex flex-wrap gap-4 text-xs text-muted-foreground">
        {visible.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className={`inline-block h-0.5 w-5 ${s.className.replace("stroke-", "bg-")}`} />
            {s.label} — now {fmt(s.values[s.values.length - 1] ?? 0, 3)}
          </span>
        ))}
      </div>
    </div>
  );
}
