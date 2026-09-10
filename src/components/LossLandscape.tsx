import { useMemo } from "react";
import type { Point } from "@/algorithms/linearRegression";
import { lossCurveOverSlope } from "@/algorithms/gradientDescent";
import { fmt } from "@/utils/calculations";

const W = 560;
const H = 280;
const PAD = { top: 20, right: 20, bottom: 42, left: 58 };

/**
 * Loss vs slope, holding the intercept fixed at its current value.
 * The curve, the marker and the arrow are all computed from the real MSE and gradient.
 */
export function LossLandscape({
  points,
  intercept,
  slope,
  gradientSlope,
  learningRate,
  slopeHistory,
}: {
  points: Point[];
  intercept: number;
  slope: number;
  gradientSlope: number;
  learningRate: number;
  slopeHistory: number[];
}) {
  const { curve, x0, x1, minMse, maxMse } = useMemo(() => {
    const span = Math.max(3, Math.abs(slope) * 1.5);
    const from = slope - span;
    const to = slope + span;
    const curve = lossCurveOverSlope(points, intercept, from, to);
    const mses = curve.map((c) => c.mse).filter(Number.isFinite);
    return {
      curve,
      x0: from,
      x1: to,
      minMse: Math.min(...mses, 0),
      maxMse: Math.max(...mses, 1),
    };
  }, [points, intercept, slope]);

  const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    H - PAD.bottom - ((v - minMse) / (maxMse - minMse || 1)) * (H - PAD.top - PAD.bottom);

  const currentMse = curve.reduce((best, c) =>
    Math.abs(c.slope - slope) < Math.abs(best.slope - slope) ? c : best,
  ).mse;

  const nextSlope = slope - learningRate * gradientSlope;
  const clampedNext = Math.max(x0, Math.min(x1, nextSlope));
  const path = curve
    .map((c, i) => `${i === 0 ? "M" : "L"}${sx(c.slope)},${sy(Math.min(c.mse, maxMse))}`)
    .join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Loss as a function of the slope">
        <defs>
          <marker id="ld-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" className="fill-line" />
          </marker>
        </defs>

        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={H - PAD.bottom}
          y2={H - PAD.bottom}
          className="stroke-axis"
          strokeWidth={1.5}
        />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} className="stroke-axis" strokeWidth={1.5} />

        <path d={path} fill="none" className="stroke-axis" strokeWidth={2} />

        {/* visited slopes */}
        {slopeHistory.slice(-40).map((s, i) => {
          if (s < x0 || s > x1) return null;
          const m = curve.reduce((best, c) => (Math.abs(c.slope - s) < Math.abs(best.slope - s) ? c : best)).mse;
          return <circle key={i} cx={sx(s)} cy={sy(Math.min(m, maxMse))} r={2.5} className="fill-muted-foreground/60" />;
        })}

        {/* gradient direction: uphill */}
        <line
          x1={sx(slope)}
          y1={sy(Math.min(currentMse, maxMse)) - 22}
          x2={sx(slope) + (gradientSlope > 0 ? 46 : -46)}
          y2={sy(Math.min(currentMse, maxMse)) - 22}
          className="stroke-resid-neg"
          strokeWidth={2}
          markerEnd="url(#ld-arrow)"
        />
        <text
          x={sx(slope) + (gradientSlope > 0 ? 52 : -52)}
          y={sy(Math.min(currentMse, maxMse)) - 26}
          textAnchor={gradientSlope > 0 ? "start" : "end"}
          className="fill-muted-foreground text-[10px]"
        >
          gradient (uphill)
        </text>

        {/* the step actually taken */}
        <line
          x1={sx(slope)}
          y1={sy(Math.min(currentMse, maxMse))}
          x2={sx(clampedNext)}
          y2={sy(Math.min(currentMse, maxMse))}
          className="stroke-line"
          strokeWidth={2.5}
          markerEnd="url(#ld-arrow)"
        />

        <circle cx={sx(slope)} cy={sy(Math.min(currentMse, maxMse))} r={6} className="fill-point" />
        <text x={sx(slope)} y={sy(Math.min(currentMse, maxMse)) + 22} textAnchor="middle" className="fill-foreground text-[10px]">
          β₁ = {fmt(slope, 3)}
        </text>

        <text x={(W + PAD.left) / 2} y={H - 8} textAnchor="middle" className="fill-foreground text-[11px]">
          Slope β₁
        </text>
        <text x={-H / 2} y={13} transform="rotate(-90)" textAnchor="middle" className="fill-foreground text-[11px]">
          MSE
        </text>
      </svg>
      <p className="mt-2 text-xs text-muted-foreground">
        The curve is the real MSE for every slope value, with β₀ held at {fmt(intercept, 3)}. The
        thin arrow points uphill (the gradient); the thick arrow is the step gradient descent
        actually takes — the opposite direction, scaled by α.
      </p>
    </div>
  );
}
