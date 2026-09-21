import { Pause, Play, RotateCcw, SkipForward, StepForward } from "lucide-react";
import type { FeatureSpec } from "@/algorithms/featureRegression";
import { Button } from "@/components/ui/button";
import type { CoordinateUpdate } from "@/optimization/coordinateDescent";
import { fmt } from "@/utils/calculations";
import { betaLabel } from "@/utils/polynomialFeatures";

const STEPS = [
  "Start with coefficients",
  "Choose one coefficient",
  "Calculate partial residual",
  "Calculate coordinate update",
  "Apply soft thresholding",
  "Update coefficient",
  "Move to next coefficient",
  "Repeat",
  "Check convergence",
];

/**
 * "Watch Lasso learn" — one coordinate update at a time, with the real
 * arithmetic of that update on display.
 */
export function OptimizationPanel({
  beta,
  specs,
  coordinate,
  lastUpdate,
  sweeps,
  updates,
  playing,
  converged,
  objective,
  mse,
  penalty,
  onStep,
  onNextCoordinate,
  onPlayToggle,
  onReset,
}: {
  beta: number[];
  specs: FeatureSpec[];
  coordinate: number;
  lastUpdate: CoordinateUpdate | null;
  sweeps: number;
  updates: number;
  playing: boolean;
  converged: boolean;
  objective: number;
  mse: number;
  penalty: number;
  onStep: () => void;
  onNextCoordinate: () => void;
  onPlayToggle: () => void;
  onReset: () => void;
}) {
  const activeStep = lastUpdate ? (lastUpdate.zeroed ? 4 : 5) : 1;
  const spec = specs[coordinate - 1];

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={onStep} disabled={playing}>
            <StepForward className="mr-1 size-4" /> Step once
          </Button>
          <Button size="sm" variant="outline" onClick={onNextCoordinate} disabled={playing}>
            <SkipForward className="mr-1 size-4" /> Next coordinate
          </Button>
          <Button size="sm" variant={playing ? "secondary" : "default"} onClick={onPlayToggle}>
            {playing ? <Pause className="mr-1 size-4" /> : <Play className="mr-1 size-4" />}
            {playing ? "Pause" : "Start learning"}
          </Button>
          <Button size="sm" variant="ghost" onClick={onReset}>
            <RotateCcw className="mr-1 size-4" /> Reset
          </Button>
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            sweeps {sweeps} · updates {updates} {converged ? "· converged" : ""}
          </span>
        </div>

        <div className="rounded-md border border-border">
          <div className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
            Current coordinate
          </div>
          <div className="px-4 py-3">
            <div className="font-mono text-2xl">
              {coordinate === 0 ? "β₀ (intercept)" : `${betaLabel(coordinate)} — ${spec?.label ?? ""}`}
            </div>
            {lastUpdate ? (
              <pre className="mt-3 overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed">{`Before:            ${betaLabel(lastUpdate.j)} = ${fmt(lastUpdate.before, 4)}
Partial residual:  ρ = ${fmt(lastUpdate.rho, 4)}   (correlation with what the other terms leave over)
Raw update:        ρ / ${fmt(lastUpdate.denom, 4)} = ${fmt(lastUpdate.rawUpdate, 4)}
Soft threshold:    S(ρ, λ=${fmt(lastUpdate.threshold, 4)}) / ${fmt(lastUpdate.denom, 4)}
After:             ${betaLabel(lastUpdate.j)} = ${fmt(lastUpdate.after, 4)}${lastUpdate.zeroed ? "   ← exactly zero, feature removed" : ""}`}</pre>
            ) : (
              <p className="mt-3 text-sm text-muted-foreground">
                Press “Step once” to perform a single coordinate update and see its arithmetic.
              </p>
            )}
          </div>
        </div>

        <ul className="grid gap-1 rounded-md border border-border p-3 text-sm sm:grid-cols-2">
          {STEPS.map((s, i) => (
            <li
              key={s}
              className={`rounded px-2 py-1 ${
                i === activeStep ? "bg-accent/20 text-foreground" : "text-muted-foreground"
              }`}
            >
              {i + 1}. {s}
            </li>
          ))}
        </ul>
      </div>

      <div className="space-y-3">
        <div className="rounded-md border border-border px-4 py-3 font-mono text-sm">
          <div className="flex justify-between py-0.5">
            <span className="text-muted-foreground">MSE</span>
            <span className="tabular-nums">{fmt(mse, 4)}</span>
          </div>
          <div className="flex justify-between py-0.5">
            <span className="text-muted-foreground">L1 penalty</span>
            <span className="tabular-nums">{fmt(penalty, 4)}</span>
          </div>
          <div className="my-1 border-t border-border" />
          <div className="flex justify-between py-0.5 text-base">
            <span>Lasso objective</span>
            <span className="tabular-nums">{fmt(objective, 4)}</span>
          </div>
        </div>
        <ul className="divide-y divide-border/60 rounded-md border border-border text-sm">
          {beta.map((b, j) => (
            <li
              key={j}
              className={`flex items-center justify-between px-3 py-1.5 font-mono ${
                j === coordinate ? "bg-accent/20" : ""
              }`}
            >
              <span>
                {betaLabel(j)}
                <span className="ml-2 text-xs text-muted-foreground">
                  {j === 0 ? "intercept" : (specs[j - 1]?.label ?? "")}
                </span>
              </span>
              <span className={`tabular-nums ${j > 0 && b === 0 ? "text-resid-neg" : ""}`}>
                {j > 0 && b === 0 ? "0" : fmt(b, 4)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
