import { useMemo, useState } from "react";
import type { Point } from "@/algorithms/linearRegression";
import { fmt } from "@/utils/calculations";

/**
 * Scatter plot + fitted curve for any model that can predict ŷ from x.
 * Same visual language as the Phase 3 polynomial chart, but the model is
 * supplied as a prediction function so feature sets other than plain
 * polynomials can be drawn too.
 */
interface Props {
  train: Point[];
  test: Point[];
  predict: (x: number) => number;
  /** Optional second curve drawn for comparison (e.g. Ridge). */
  comparePredict?: (x: number) => number;
  compareLabel?: string;
  showResiduals: boolean;
  showSplit: boolean;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const W = 660;
const H = 440;
const PAD = { top: 24, right: 24, bottom: 52, left: 62 };
const SAMPLES = 220;

export function FeatureCurveChart({
  train,
  test,
  predict,
  comparePredict,
  compareLabel,
  showResiduals,
  showSplit,
  selectedId,
  onSelect,
}: Props) {
  const [hoverId, setHoverId] = useState<string | null>(null);
  const all = useMemo(() => [...train, ...(showSplit ? test : [])], [train, test, showSplit]);

  const view = useMemo(() => {
    const xs = all.map((p) => p.x);
    const minX = Math.min(...xs, 0);
    const maxX = Math.max(...xs, 1);
    const padX = (maxX - minX) * 0.08 || 1;
    const x0 = minX - padX;
    const x1 = maxX + padX;

    const sample = (f: (x: number) => number) => {
      const out: { x: number; y: number }[] = [];
      for (let i = 0; i <= SAMPLES; i++) {
        const x = x0 + ((x1 - x0) * i) / SAMPLES;
        out.push({ x, y: f(x) });
      }
      return out;
    };

    const dataYs = all.map((p) => p.y);
    const minData = Math.min(...dataYs, 0);
    const maxData = Math.max(...dataYs, 1);
    const spread = Math.max(maxData - minData, 1);
    const y0 = minData - spread * 0.35;
    const y1 = maxData + spread * 0.35;

    const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
    const sy = (v: number) => H - PAD.bottom - ((v - y0) / (y1 - y0)) * (H - PAD.top - PAD.bottom);
    return {
      x0,
      x1,
      y0,
      y1,
      sx,
      sy,
      curve: sample(predict),
      compare: comparePredict ? sample(comparePredict) : null,
    };
  }, [all, predict, comparePredict]);

  const clampY = (v: number) => Math.min(view.y1, Math.max(view.y0, v));

  const buildPath = (curve: { x: number; y: number }[] | null) => {
    if (!curve) return "";
    let d = "";
    let pen = false;
    for (const pt of curve) {
      if (!Number.isFinite(pt.y)) {
        pen = false;
        continue;
      }
      const inside = pt.y >= view.y0 && pt.y <= view.y1;
      const yy = view.sy(clampY(pt.y));
      d += `${pen ? "L" : "M"}${view.sx(pt.x).toFixed(2)},${yy.toFixed(2)}`;
      pen = inside;
    }
    return d;
  };

  const path = buildPath(view.curve);
  const comparePath = buildPath(view.compare);
  const activeId = hoverId ?? selectedId;

  const ticks = (a: number, b: number) => {
    const step = niceStep((b - a) / 6);
    const out: number[] = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Number(v.toFixed(6)));
    return out;
  };

  const renderPoint = (p: Point, kind: "train" | "test") => {
    const active = activeId === p.id;
    const yhat = predict(p.x);
    const cx = view.sx(p.x);
    const cy = view.sy(clampY(p.y));
    const handlers = {
      onMouseEnter: () => setHoverId(p.id),
      onMouseLeave: () => setHoverId(null),
      onFocus: () => setHoverId(p.id),
      onBlur: () => setHoverId(null),
      onClick: () => onSelect(selectedId === p.id ? null : p.id),
    };
    return (
      <g key={p.id}>
        {active && (
          <>
            <circle cx={cx} cy={cy} r={12} className="fill-accent/25" />
            <text x={cx + 14} y={cy - 10} className="fill-foreground text-[11px] font-medium">
              x {fmt(p.x)} · y {fmt(p.y)} · ŷ {fmt(yhat)} · e {fmt(p.y - yhat)}
            </text>
          </>
        )}
        {kind === "test" ? (
          <rect
            x={cx - (active ? 6.5 : 5)}
            y={cy - (active ? 6.5 : 5)}
            width={active ? 13 : 10}
            height={active ? 13 : 10}
            className="cursor-pointer fill-background stroke-resid-neg"
            strokeWidth={2}
            tabIndex={0}
            role="button"
            aria-label={`Test observation x ${fmt(p.x)}, y ${fmt(p.y)}, prediction ${fmt(yhat)}`}
            {...handlers}
          />
        ) : (
          <circle
            cx={cx}
            cy={cy}
            r={active ? 7 : 5.5}
            className="cursor-pointer fill-point stroke-background transition-all"
            strokeWidth={1.5}
            tabIndex={0}
            role="button"
            aria-label={`Training observation x ${fmt(p.x)}, y ${fmt(p.y)}, prediction ${fmt(yhat)}`}
            {...handlers}
          />
        )}
      </g>
    );
  };

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-manipulation select-none"
        role="img"
        aria-label="Dataset with the fitted Lasso curve"
      >
        {ticks(view.x0, view.x1).map((t) => (
          <line
            key={`gx${t}`}
            x1={view.sx(t)}
            x2={view.sx(t)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            className="stroke-grid"
            strokeWidth={1}
          />
        ))}
        {ticks(view.y0, view.y1).map((t) => (
          <line
            key={`gy${t}`}
            y1={view.sy(t)}
            y2={view.sy(t)}
            x1={PAD.left}
            x2={W - PAD.right}
            className="stroke-grid"
            strokeWidth={1}
          />
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
        {ticks(view.x0, view.x1).map((t) => (
          <text
            key={`tx${t}`}
            x={view.sx(t)}
            y={H - PAD.bottom + 18}
            textAnchor="middle"
            className="fill-muted-foreground text-[11px]"
          >
            {fmt(t, 0)}
          </text>
        ))}
        {ticks(view.y0, view.y1).map((t) => (
          <text
            key={`ty${t}`}
            x={PAD.left - 10}
            y={view.sy(t) + 4}
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

        <clipPath id="lasso-clip">
          <rect
            x={PAD.left}
            y={PAD.top}
            width={W - PAD.left - PAD.right}
            height={H - PAD.top - PAD.bottom}
          />
        </clipPath>

        {comparePath && (
          <path
            d={comparePath}
            fill="none"
            className="stroke-axis"
            strokeWidth={2}
            strokeDasharray="6 4"
            clipPath="url(#lasso-clip)"
          />
        )}
        <path
          d={path}
          fill="none"
          className="stroke-line"
          strokeWidth={2.5}
          strokeLinecap="round"
          clipPath="url(#lasso-clip)"
        />

        {showResiduals &&
          all.map((p) => {
            const yhat = predict(p.x);
            if (!Number.isFinite(yhat)) return null;
            const positive = p.y - yhat >= 0;
            return (
              <line
                key={`r${p.id}`}
                x1={view.sx(p.x)}
                x2={view.sx(p.x)}
                y1={view.sy(clampY(p.y))}
                y2={view.sy(clampY(yhat))}
                className={positive ? "stroke-resid-pos" : "stroke-resid-neg"}
                strokeWidth={activeId === p.id ? 3 : 2}
                strokeDasharray={positive ? undefined : "4 3"}
                clipPath="url(#lasso-clip)"
              />
            );
          })}

        {train.map((p) => renderPoint(p, "train"))}
        {showSplit && test.map((p) => renderPoint(p, "test"))}
      </svg>
      <p className="mt-3 flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="inline-block size-2.5 rounded-full bg-point" /> training point
        </span>
        {showSplit && (
          <span className="flex items-center gap-2">
            <span className="inline-block size-2.5 border-2 border-resid-neg" /> test point (never
            used for fitting)
          </span>
        )}
        <span className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-5 bg-line" /> Lasso
        </span>
        {comparePath && (
          <span className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-5 bg-axis" /> {compareLabel ?? "comparison"}
          </span>
        )}
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
