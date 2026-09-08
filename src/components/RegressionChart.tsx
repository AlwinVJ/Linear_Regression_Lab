import { useMemo, useRef, useState } from "react";
import type { ModelParams, Point } from "@/algorithms/linearRegression";
import { predict } from "@/algorithms/linearRegression";
import { fmt } from "@/utils/calculations";

interface Props {
  points: Point[];
  params: ModelParams;
  showResiduals: boolean;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const W = 640;
const H = 440;
const PAD = { top: 24, right: 24, bottom: 52, left: 56 };

export function RegressionChart({ points, params, showResiduals, selectedId, onSelect }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);

  const scales = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ysAll = [
      ...points.map((p) => p.y),
      ...points.map((p) => predict(params, p.x)),
    ].filter(Number.isFinite);
    const minX = Math.min(0, ...xs);
    const maxX = Math.max(1, ...xs);
    const minY = Math.min(0, ...ysAll);
    const maxY = Math.max(1, ...ysAll);
    const padX = (maxX - minX) * 0.08 || 1;
    const padY = (maxY - minY) * 0.12 || 1;
    const x0 = minX - padX;
    const x1 = maxX + padX;
    const y0 = minY - padY;
    const y1 = maxY + padY;
    const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
    const sy = (v: number) => H - PAD.bottom - ((v - y0) / (y1 - y0)) * (H - PAD.top - PAD.bottom);
    return { x0, x1, y0, y1, sx, sy };
  }, [points, params]);

  const ticks = (a: number, b: number) => {
    const step = niceStep((b - a) / 6);
    const out: number[] = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Number(v.toFixed(6)));
    return out;
  };

  const activeId = hoverId ?? selectedId;
  const lineY0 = predict(params, scales.x0);
  const lineY1 = predict(params, scales.x1);

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-manipulation select-none"
        role="img"
        aria-label="Scatter plot of the dataset with the current regression line"
        onMouseLeave={() => setHoverId(null)}
      >
        {/* graph paper */}
        {ticks(scales.x0, scales.x1).map((t) => (
          <line
            key={`gx${t}`}
            x1={scales.sx(t)}
            x2={scales.sx(t)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            className="stroke-grid"
            strokeWidth={1}
          />
        ))}
        {ticks(scales.y0, scales.y1).map((t) => (
          <line
            key={`gy${t}`}
            y1={scales.sy(t)}
            y2={scales.sy(t)}
            x1={PAD.left}
            x2={W - PAD.right}
            className="stroke-grid"
            strokeWidth={1}
          />
        ))}

        {/* axes */}
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
        {ticks(scales.x0, scales.x1).map((t) => (
          <text
            key={`tx${t}`}
            x={scales.sx(t)}
            y={H - PAD.bottom + 18}
            textAnchor="middle"
            className="fill-muted-foreground text-[11px]"
          >
            {fmt(t, 0)}
          </text>
        ))}
        {ticks(scales.y0, scales.y1).map((t) => (
          <text
            key={`ty${t}`}
            x={PAD.left - 10}
            y={scales.sy(t) + 4}
            textAnchor="end"
            className="fill-muted-foreground text-[11px]"
          >
            {fmt(t, 0)}
          </text>
        ))}
        <text
          x={(W + PAD.left) / 2}
          y={H - 10}
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Input (x)
        </text>
        <text
          x={-H / 2}
          y={16}
          transform="rotate(-90)"
          textAnchor="middle"
          className="fill-foreground text-[12px] font-medium"
        >
          Output (y)
        </text>

        {/* regression line */}
        <line
          x1={scales.sx(scales.x0)}
          y1={scales.sy(lineY0)}
          x2={scales.sx(scales.x1)}
          y2={scales.sy(lineY1)}
          className="stroke-line transition-all duration-300 ease-out"
          strokeWidth={2.5}
          strokeLinecap="round"
        />

        {/* residuals */}
        {showResiduals &&
          points.map((p) => {
            const yhat = predict(params, p.x);
            const positive = p.y - yhat >= 0;
            return (
              <g key={`r${p.id}`}>
                <line
                  x1={scales.sx(p.x)}
                  x2={scales.sx(p.x)}
                  y1={scales.sy(p.y)}
                  y2={scales.sy(yhat)}
                  className={positive ? "stroke-resid-pos" : "stroke-resid-neg"}
                  strokeWidth={activeId === p.id ? 3 : 2}
                  strokeDasharray={positive ? undefined : "4 3"}
                />
                <rect
                  x={scales.sx(p.x) - 3}
                  y={scales.sy(yhat) - 3}
                  width={6}
                  height={6}
                  className="fill-line"
                  transform={`rotate(45 ${scales.sx(p.x)} ${scales.sy(yhat)})`}
                />
              </g>
            );
          })}

        {/* points */}
        {points.map((p) => {
          const active = activeId === p.id;
          const yhat = predict(params, p.x);
          return (
            <g key={p.id}>
              {active && (
                <>
                  <circle
                    cx={scales.sx(p.x)}
                    cy={scales.sy(p.y)}
                    r={12}
                    className="fill-accent/25"
                  />
                  <text
                    x={scales.sx(p.x) + 14}
                    y={scales.sy(p.y) - 10}
                    className="fill-foreground text-[11px] font-medium"
                  >
                    x {fmt(p.x)} · y {fmt(p.y)} · ŷ {fmt(yhat)}
                  </text>
                </>
              )}
              <circle
                cx={scales.sx(p.x)}
                cy={scales.sy(p.y)}
                r={active ? 7 : 5.5}
                className="fill-point stroke-background cursor-pointer transition-all"
                strokeWidth={1.5}
                tabIndex={0}
                role="button"
                aria-label={`Observation x ${fmt(p.x)}, y ${fmt(p.y)}, prediction ${fmt(yhat)}`}
                onMouseEnter={() => setHoverId(p.id)}
                onFocus={() => setHoverId(p.id)}
                onBlur={() => setHoverId(null)}
                onClick={() => onSelect(selectedId === p.id ? null : p.id)}
              />
            </g>
          );
        })}
      </svg>
      <p className="mt-3 text-sm text-muted-foreground">
        Each point represents an example the model is trying to learn from. Hover or click a point
        to inspect its prediction.
      </p>
    </div>
  );
}

function niceStep(raw: number) {
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const n = raw / pow;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * pow;
}
