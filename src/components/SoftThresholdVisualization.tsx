import { Slider } from "@/components/ui/slider";
import { softThreshold } from "@/optimization/softThresholding";
import { fmt } from "@/utils/calculations";

const W = 520;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 42, left: 50 };
const RANGE = 6;

/** Interactive S(z, γ) = sign(z)·max(|z| − γ, 0). */
export function SoftThresholdVisualization({
  z,
  gamma,
  onZ,
  onGamma,
}: {
  z: number;
  gamma: number;
  onZ: (v: number) => void;
  onGamma: (v: number) => void;
}) {
  const sx = (v: number) => PAD.left + ((v + RANGE) / (2 * RANGE)) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    PAD.top + ((RANGE - Math.min(Math.max(v, -RANGE), RANGE)) / (2 * RANGE)) * (H - PAD.top - PAD.bottom);

  const samples = Array.from({ length: 241 }, (_, i) => -RANGE + (2 * RANGE * i) / 240);
  const curve = samples.map((v) => `${sx(v).toFixed(1)},${sy(softThreshold(v, gamma)).toFixed(1)}`).join(" ");
  const identity = samples.map((v) => `${sx(v).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
  const out = softThreshold(z, gamma);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_260px]">
      <div>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full select-none"
          role="img"
          aria-label="Soft thresholding function"
        >
          <line x1={PAD.left} x2={W - PAD.right} y1={sy(0)} y2={sy(0)} className="stroke-axis" strokeWidth={1.5} />
          <line x1={sx(0)} x2={sx(0)} y1={PAD.top} y2={H - PAD.bottom} className="stroke-axis" strokeWidth={1.5} />
          <rect
            x={sx(-gamma)}
            y={PAD.top}
            width={Math.max(sx(gamma) - sx(-gamma), 0)}
            height={H - PAD.top - PAD.bottom}
            className="fill-resid-neg/10"
          />
          <polyline points={identity} fill="none" className="stroke-grid" strokeWidth={2} strokeDasharray="4 4" />
          <polyline points={curve} fill="none" className="stroke-line" strokeWidth={2.5} />
          <line x1={sx(z)} x2={sx(z)} y1={sy(0)} y2={sy(out)} className="stroke-accent" strokeWidth={2} />
          <circle cx={sx(z)} cy={sy(out)} r={5} className="fill-point" />
          <text x={sx(z) + 8} y={sy(out) - 8} className="fill-foreground text-[11px]">
            S({fmt(z, 2)}, {fmt(gamma, 2)}) = {fmt(out, 2)}
          </text>
          <text
            x={(W + PAD.left) / 2}
            y={H - 8}
            textAnchor="middle"
            className="fill-foreground text-[12px] font-medium"
          >
            Raw coefficient z
          </text>
          <text
            x={-H / 2}
            y={14}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-foreground text-[12px] font-medium"
          >
            Lasso coefficient
          </text>
        </svg>
      </div>
      <div className="space-y-4">
        <div>
          <label className="text-xs uppercase tracking-widest text-muted-foreground">
            Raw coefficient z = {fmt(z, 2)}
          </label>
          <Slider className="mt-2" min={-6} max={6} step={0.1} value={[z]} onValueChange={([v]) => onZ(v ?? 0)} />
        </div>
        <div>
          <label className="text-xs uppercase tracking-widest text-muted-foreground">
            Threshold γ = {fmt(gamma, 2)}
          </label>
          <Slider
            className="mt-2"
            min={0}
            max={5}
            step={0.1}
            value={[gamma]}
            onValueChange={([v]) => onGamma(v ?? 0)}
          />
        </div>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed">{`S(z, γ) = sign(z)·max(|z| − γ, 0)

z = ${fmt(z, 2)}
γ = ${fmt(gamma, 2)}

|z| − γ = ${fmt(Math.abs(z) - gamma, 2)}
S(z, γ) = ${fmt(out, 2)}`}</pre>
        <p className="text-sm text-muted-foreground">
          {Math.abs(z) <= gamma
            ? "The raw value is smaller than the threshold, so the coefficient lands exactly on zero."
            : "The raw value survives the threshold, but it is pulled γ closer to zero."}
        </p>
      </div>
    </div>
  );
}
