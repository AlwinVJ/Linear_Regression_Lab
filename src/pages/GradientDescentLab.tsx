import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, Dices, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import {
  assessConvergence,
  calculateGradients,
  gradientDescentStep,
  isDiverging,
  LEARNING_RATE_PRESETS,
  type HistoryEntry,
  type StepResult,
} from "@/algorithms/gradientDescent";
import {
  buildBreakdown,
  calculateMSE,
  fitOLS,
  type ModelParams,
  type Point,
} from "@/algorithms/linearRegression";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { FlowDiagram } from "@/components/FlowDiagram";
import { HistoryChart } from "@/components/HistoryChart";
import { LossLandscape } from "@/components/LossLandscape";
import { RegressionChart } from "@/components/RegressionChart";
import { ResidualTable } from "@/components/ResidualTable";
import { SiteNav } from "@/components/SiteNav";
import { Term } from "@/components/Term";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { TooltipProvider } from "@/components/ui/tooltip";
import { equationString, fmt } from "@/utils/calculations";
import { defaultDataset } from "@/utils/dataset";

const DEFAULT_INIT: ModelParams = { intercept: 0, slope: 0 };
const DEFAULT_ALPHA = 0.01;
const DEFAULT_MAX_ITER = 300;

const GD_PRESETS = [
  {
    id: "slow",
    name: "Slow learning",
    caption: "Very small steps — improvement is barely visible.",
    alpha: 0.001,
    init: { intercept: 0, slope: 0 },
  },
  {
    id: "good",
    name: "Good learning rate",
    caption: "Smooth, steady convergence.",
    alpha: 0.01,
    init: { intercept: 0, slope: 0 },
  },
  {
    id: "large",
    name: "Large learning rate",
    caption: "Steps overshoot the minimum — watch the loss bounce or explode.",
    alpha: 0.05,
    init: { intercept: 0, slope: 0 },
  },
] as const;

function Step({
  n,
  title,
  children,
}: {
  n: number;
  title: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="scroll-mt-8">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-xs text-muted-foreground">{String(n).padStart(2, "0")}</span>
        <h2 className="font-display text-2xl">{title}</h2>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="px-4 py-3">
      <div className="text-xs uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-2xl tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

const EXPLAIN_STEPS = [
  "Calculate predictions ŷ = β₀ + β₁x for every point.",
  "Calculate residuals: y − ŷ for every point.",
  "Calculate the loss: MSE = (1/n) Σ(y − ŷ)².",
  "Calculate the gradients: how the loss changes with β₀ and with β₁.",
  "Calculate the updates: α × gradient for each parameter.",
  "Update the model: subtract each update from its parameter.",
];

export default function GradientDescentLab() {
  const [points, setPoints] = useState<Point[]>(defaultDataset);
  const [init, setInit] = useState<ModelParams>(DEFAULT_INIT);
  const [alpha, setAlpha] = useState(DEFAULT_ALPHA);
  const [maxIter, setMaxIter] = useState(DEFAULT_MAX_ITER);
  const [params, setParams] = useState<ModelParams>(DEFAULT_INIT);
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [lastStep, setLastStep] = useState<StepResult | null>(null);
  const [playing, setPlaying] = useState(false);
  const [diverged, setDiverged] = useState(false);
  const [showResiduals, setShowResiduals] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showB0, setShowB0] = useState(true);
  const [showB1, setShowB1] = useState(true);
  const [explainIndex, setExplainIndex] = useState<number | null>(null);
  const [message, setMessage] = useState(
    "The line starts deliberately bad. Press Step Once to perform a single, real gradient descent iteration.",
  );

  const mse = useMemo(() => calculateMSE(points, params), [points, params]);
  const gradients = useMemo(() => calculateGradients(points, params), [points, params]);
  const rows = useMemo(() => buildBreakdown(points, params), [points, params]);
  const ols = useMemo(() => fitOLS(points), [points]);
  const olsMse = useMemo(() => calculateMSE(points, ols), [points, ols]);
  const initialMse = useMemo(() => calculateMSE(points, init), [points, init]);
  const convergence = useMemo(() => assessConvergence(history, diverged), [history, diverged]);
  const selected = rows.find((r) => r.id === selectedId) ?? null;

  const fullHistory = useMemo<HistoryEntry[]>(() => {
    const first: HistoryEntry = {
      iteration: 0,
      intercept: init.intercept,
      slope: init.slope,
      mse: initialMse,
      gradientIntercept: calculateGradients(points, init).gradientIntercept,
      gradientSlope: calculateGradients(points, init).gradientSlope,
    };
    return [first, ...history];
  }, [history, init, initialMse, points]);

  const doStep = useCallback(() => {
    setParams((current) => {
      const result = gradientDescentStep(points, current, alpha);
      if (isDiverging(result.after, result.mseAfter)) {
        setDiverged(true);
        setPlaying(false);
        setMessage("The numbers blew up. The learning rate may be too large. Try reducing it.");
        return current;
      }
      setLastStep(result);
      setIteration((i) => i + 1);
      setHistory((h) => [
        ...h,
        {
          iteration: h.length + 1,
          intercept: result.after.intercept,
          slope: result.after.slope,
          mse: result.mseAfter,
          gradientIntercept: result.gradients.gradientIntercept,
          gradientSlope: result.gradients.gradientSlope,
        },
      ]);
      return result.after;
    });
  }, [alpha, points]);

  // Play loop — one real iteration per animation tick.
  const rafRef = useRef<number | null>(null);
  const lastTick = useRef(0);
  useEffect(() => {
    if (!playing) return;
    const loop = (t: number) => {
      if (t - lastTick.current > 60) {
        lastTick.current = t;
        doStep();
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, doStep]);

  useEffect(() => {
    if (playing && (iteration >= maxIter || convergence.status === "converged")) {
      setPlaying(false);
      setMessage(
        convergence.status === "converged"
          ? "Gradient descent stopped: the loss is no longer changing meaningfully."
          : `Reached the maximum of ${maxIter} iterations.`,
      );
    }
  }, [playing, iteration, maxIter, convergence.status]);

  const resetToInit = () => {
    setPlaying(false);
    setParams(init);
    setIteration(0);
    setHistory([]);
    setLastStep(null);
    setDiverged(false);
    setExplainIndex(null);
  };

  const resetEverything = () => {
    setPlaying(false);
    setPoints(defaultDataset());
    setInit(DEFAULT_INIT);
    setAlpha(DEFAULT_ALPHA);
    setMaxIter(DEFAULT_MAX_ITER);
    setParams(DEFAULT_INIT);
    setIteration(0);
    setHistory([]);
    setLastStep(null);
    setDiverged(false);
    setShowResiduals(true);
    setSelectedId(null);
    setExplainIndex(null);
    setMessage("Everything is back to the starting configuration: β₀ = 0, β₁ = 0, α = 0.01.");
  };

  const setInitial = (patch: Partial<ModelParams>) => {
    const next = { ...init, ...patch };
    setInit(next);
    setPlaying(false);
    setParams(next);
    setIteration(0);
    setHistory([]);
    setLastStep(null);
    setDiverged(false);
  };

  const randomInit = () => {
    setInitial({
      intercept: Number((Math.random() * 16 - 8).toFixed(2)),
      slope: Number((Math.random() * 8 - 4).toFixed(2)),
    });
    setMessage(
      "New random starting point. The path and the number of steps change, but this loss surface has a single lowest point, so the model heads to the same place.",
    );
  };

  const applyPreset = (id: string) => {
    const p = GD_PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPlaying(false);
    setAlpha(p.alpha);
    setInit(p.init);
    setParams(p.init);
    setIteration(0);
    setHistory([]);
    setLastStep(null);
    setDiverged(false);
    setMessage(p.caption);
  };

  const rateNote =
    alpha <= 0.002
      ? "Learning is progressing slowly because the steps are small."
      : alpha <= 0.02
        ? "The model is moving toward a minimum efficiently."
        : "The steps are too large and the model may overshoot the minimum.";

  const statusDot =
    convergence.status === "converged"
      ? "✓"
      : convergence.status === "diverged"
        ? "✕"
        : "●";

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Gradient Descent</h1>
              <p className="mt-1 text-muted-foreground">
                Watch a bad line learn its own parameters, one real calculation at a time.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <SiteNav />
              <Button variant="ghost" size="sm" onClick={resetEverything}>
                <RotateCcw className="size-3.5" /> Reset everything
              </Button>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-14 px-5 py-10">
          <Step n={1} title="How does a model actually learn?">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  We already know how to calculate the best-fitting line using{" "}
                  <strong className="text-foreground">Ordinary Least Squares</strong> — one formula,
                  one answer.
                </p>
                <p className="text-foreground">
                  But what if we wanted the computer to learn the parameters step by step instead?
                </p>
                <p>
                  That is gradient descent: start with a bad guess, measure how wrong it is, work out
                  which direction makes it worse, and move the other way. Repeat.
                </p>
              </div>
              <div className="rounded-md border border-border p-4">
                <FlowDiagram
                  active
                  steps={[
                    "Bad model",
                    "Predictions",
                    "Errors",
                    "Loss",
                    "Gradients",
                    "Update parameters",
                    "Repeat",
                    "Better model",
                  ]}
                />
              </div>
            </div>
          </Step>

          {/* Playground */}
          <Step n={2} title="Gradient descent playground">
            <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
              <div className="space-y-6">
                <div className="rounded-md border border-border p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Starting point
                  </h3>
                  <div className="mt-3 space-y-4">
                    <div>
                      <div className="flex items-baseline justify-between text-sm">
                        <label htmlFor="init-b0">Initial intercept β₀</label>
                        <span className="font-mono tabular-nums">{fmt(init.intercept, 2)}</span>
                      </div>
                      <Slider
                        id="init-b0"
                        min={-20}
                        max={20}
                        step={0.1}
                        value={[init.intercept]}
                        onValueChange={(v) => setInitial({ intercept: v[0] ?? 0 })}
                        className="mt-2"
                      />
                    </div>
                    <div>
                      <div className="flex items-baseline justify-between text-sm">
                        <label htmlFor="init-b1">Initial slope β₁</label>
                        <span className="font-mono tabular-nums">{fmt(init.slope, 2)}</span>
                      </div>
                      <Slider
                        id="init-b1"
                        min={-5}
                        max={5}
                        step={0.05}
                        value={[init.slope]}
                        onValueChange={(v) => setInitial({ slope: v[0] ?? 0 })}
                        className="mt-2"
                      />
                    </div>
                    <Button variant="outline" size="sm" className="w-full" onClick={randomInit}>
                      <Dices className="size-3.5" /> Random initialization
                    </Button>
                    <p className="font-mono text-xs text-muted-foreground">
                      Initial β₀ = {fmt(init.intercept, 2)} · Initial β₁ = {fmt(init.slope, 2)} ·
                      Initial MSE = {fmt(initialMse, 3)}
                    </p>
                  </div>
                </div>

                <div className="rounded-md border border-border p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Learning rate α
                  </h3>
                  <div className="mt-3 space-y-3">
                    <div className="flex gap-1.5">
                      {LEARNING_RATE_PRESETS.map((p) => (
                        <button
                          key={p.id}
                          onClick={() => {
                            setAlpha(p.alpha);
                            setMessage(p.note);
                          }}
                          className={`flex-1 rounded-md border px-2 py-1.5 text-xs transition-colors ${
                            Math.abs(alpha - p.alpha) < 1e-9
                              ? "border-axis bg-accent/20"
                              : "border-border hover:bg-muted"
                          }`}
                        >
                          {p.label}
                          <div className="font-mono text-[10px] text-muted-foreground">
                            {p.alpha}
                          </div>
                        </button>
                      ))}
                    </div>
                    <div>
                      <div className="flex items-baseline justify-between text-sm">
                        <label htmlFor="alpha">α</label>
                        <span className="font-mono tabular-nums">{alpha.toFixed(4)}</span>
                      </div>
                      <Slider
                        id="alpha"
                        min={0.0005}
                        max={0.06}
                        step={0.0005}
                        value={[alpha]}
                        onValueChange={(v) => setAlpha(v[0] ?? DEFAULT_ALPHA)}
                        className="mt-2"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">{rateNote}</p>
                    <div>
                      <div className="flex items-baseline justify-between text-sm">
                        <label htmlFor="maxit">Maximum iterations</label>
                        <span className="font-mono tabular-nums">{maxIter}</span>
                      </div>
                      <Slider
                        id="maxit"
                        min={10}
                        max={2000}
                        step={10}
                        value={[maxIter]}
                        onValueChange={(v) => setMaxIter(v[0] ?? DEFAULT_MAX_ITER)}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-border p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Run it
                  </h3>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <Button size="sm" onClick={doStep} disabled={playing || diverged}>
                      <StepForward className="size-3.5" /> Step Once
                    </Button>
                    <Button
                      size="sm"
                      variant={playing ? "secondary" : "default"}
                      onClick={() => {
                        if (!playing && diverged) return;
                        setPlaying(!playing);
                      }}
                      disabled={diverged}
                    >
                      {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
                      {playing ? "Pause" : "Start Learning"}
                    </Button>
                    <Button size="sm" variant="outline" onClick={resetToInit}>
                      <RotateCcw className="size-3.5" /> Reset
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        resetToInit();
                        setPlaying(true);
                      }}
                    >
                      <Play className="size-3.5" /> Restart
                    </Button>
                  </div>
                  <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-sm">
                    <label htmlFor="gd-residuals">Show residuals</label>
                    <Switch
                      id="gd-residuals"
                      checked={showResiduals}
                      onCheckedChange={setShowResiduals}
                    />
                  </div>
                </div>

                <div className="rounded-md border border-border p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Experiment presets
                  </h3>
                  <div className="mt-2 space-y-1.5">
                    {GD_PRESETS.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => applyPreset(p.id)}
                        className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                          Math.abs(alpha - p.alpha) < 1e-9
                            ? "border-axis bg-accent/15"
                            : "border-border hover:bg-muted"
                        }`}
                      >
                        <div className="font-medium">{p.name}</div>
                        <div className="text-xs text-muted-foreground">{p.caption}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-5">
                <div className="rounded-md border border-border bg-card p-4">
                  <RegressionChart
                    points={points}
                    params={params}
                    showResiduals={showResiduals}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                  />
                </div>

                <div className="grid grid-cols-2 divide-x divide-y divide-border rounded-md border border-border sm:grid-cols-3 lg:grid-cols-6">
                  <Stat label="Iteration" value={String(iteration)} sub={`max ${maxIter}`} />
                  <Stat label="β₀" value={fmt(params.intercept, 3)} sub="intercept" />
                  <Stat label="β₁" value={fmt(params.slope, 3)} sub="slope" />
                  <Stat label="MSE" value={fmt(mse, 3)} sub="loss" />
                  <Stat label="∂/∂β₀" value={fmt(gradients.gradientIntercept, 3)} sub="gradient" />
                  <Stat label="∂/∂β₁" value={fmt(gradients.gradientSlope, 3)} sub="gradient" />
                </div>

                <div className="rounded-md border border-border p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                        Convergence
                      </h3>
                      <p className="mt-1 text-lg">
                        <span
                          className={
                            convergence.status === "converged"
                              ? "text-resid-pos"
                              : convergence.status === "diverged"
                                ? "text-resid-neg"
                                : "text-foreground"
                          }
                        >
                          {statusDot}
                        </span>{" "}
                        {convergence.label}
                      </p>
                    </div>
                    <div className="font-mono text-sm">
                      <div className="text-xs uppercase tracking-widest text-muted-foreground">
                        Loss change
                      </div>
                      {Number.isFinite(convergence.lossChange)
                        ? convergence.lossChange.toExponential(2)
                        : "—"}
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">{convergence.detail}</p>
                </div>

                <div className="rounded-md border border-border p-4 font-mono text-sm">
                  Current model: {equationString(params.intercept, params.slope)}
                </div>

                <ExplanationPanel message={message} />
              </div>
            </div>
          </Step>

          {/* One step numerically */}
          <Step n={3} title="One gradient descent step, written out">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-md border border-border p-5 font-mono text-sm">
                {lastStep ? (
                  <div className="space-y-3">
                    <div>
                      <div className="text-xs uppercase tracking-widest text-muted-foreground">
                        Before update
                      </div>
                      β₀ = {fmt(lastStep.before.intercept, 4)} <br />
                      β₁ = {fmt(lastStep.before.slope, 4)} <br />
                      MSE = {fmt(lastStep.mseBefore, 4)}
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-widest text-muted-foreground">
                        Gradients
                      </div>
                      ∂MSE/∂β₀ = {fmt(lastStep.gradients.gradientIntercept, 4)} <br />
                      ∂MSE/∂β₁ = {fmt(lastStep.gradients.gradientSlope, 4)}
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-widest text-muted-foreground">
                        Learning rate
                      </div>
                      α = {lastStep.learningRate}
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-widest text-muted-foreground">
                        Update
                      </div>
                      β₀(new) = {fmt(lastStep.before.intercept, 4)} − ({lastStep.learningRate} ×{" "}
                      {fmt(lastStep.gradients.gradientIntercept, 4)}) ={" "}
                      {fmt(lastStep.after.intercept, 4)}
                      <br />
                      β₁(new) = {fmt(lastStep.before.slope, 4)} − ({lastStep.learningRate} ×{" "}
                      {fmt(lastStep.gradients.gradientSlope, 4)}) = {fmt(lastStep.after.slope, 4)}
                      <br />
                      New MSE = {fmt(lastStep.mseAfter, 4)}
                    </div>
                  </div>
                ) : (
                  <p className="font-sans text-muted-foreground">
                    Press <strong className="text-foreground">Step Once</strong> and the exact
                    arithmetic for that iteration appears here — using your dataset and your current
                    parameters.
                  </p>
                )}
              </div>

              <div className="rounded-md border border-border p-5">
                <div className="flex items-center justify-between">
                  <h3 className="font-display text-lg">Explain this step</h3>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      setExplainIndex((i) =>
                        i === null || i >= EXPLAIN_STEPS.length - 1 ? 0 : i + 1,
                      )
                    }
                  >
                    {explainIndex === null ? "Explain This Step" : "Next"}
                  </Button>
                </div>
                <ol className="mt-4 space-y-2 text-sm">
                  {EXPLAIN_STEPS.map((s, i) => (
                    <li
                      key={s}
                      className={`rounded-md border px-3 py-2 transition-colors ${
                        explainIndex === i
                          ? "border-axis bg-accent/20 text-foreground"
                          : "border-border text-muted-foreground"
                      }`}
                    >
                      <span className="font-mono text-xs">Step {i + 1}</span> — {s}
                      {explainIndex === i && (
                        <div className="mt-1 font-mono text-xs">
                          {i === 0 && `e.g. x = ${fmt(points[0]?.x ?? 0)} → ŷ = ${fmt(rows[0]?.prediction ?? 0, 3)}`}
                          {i === 1 && `e.g. ${fmt(rows[0]?.actual ?? 0)} − ${fmt(rows[0]?.prediction ?? 0, 3)} = ${fmt(rows[0]?.error ?? 0, 3)}`}
                          {i === 2 && `MSE = ${fmt(mse, 4)}`}
                          {i === 3 &&
                            `∂β₀ = ${fmt(gradients.gradientIntercept, 4)} · ∂β₁ = ${fmt(gradients.gradientSlope, 4)}`}
                          {i === 4 &&
                            `Δβ₀ = ${fmt(alpha * gradients.gradientIntercept, 5)} · Δβ₁ = ${fmt(alpha * gradients.gradientSlope, 5)}`}
                          {i === 5 &&
                            `β₀ → ${fmt(params.intercept - alpha * gradients.gradientIntercept, 4)} · β₁ → ${fmt(params.slope - alpha * gradients.gradientSlope, 4)}`}
                        </div>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </Step>

          {/* Charts */}
          <Step n={4} title="Loss vs iteration">
            <div className="rounded-md border border-border p-4">
              <HistoryChart
                series={[
                  {
                    key: "mse",
                    label: "MSE",
                    values: fullHistory.map((h) => h.mse),
                    className: "stroke-line text-line",
                  },
                ]}
                xLabel="Iteration"
                yLabel="MSE"
              />
            </div>
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
              Every point on this curve is the real MSE after that iteration. A healthy learning rate
              gives a smooth downhill curve; too large a rate makes it jump around or climb.
            </p>
          </Step>

          <Step n={5} title="Parameter trajectory">
            <div className="rounded-md border border-border p-4">
              <div className="mb-3 flex flex-wrap gap-5 text-sm">
                <label className="flex items-center gap-2">
                  <Switch checked={showB0} onCheckedChange={setShowB0} /> Show β₀
                </label>
                <label className="flex items-center gap-2">
                  <Switch checked={showB1} onCheckedChange={setShowB1} /> Show β₁
                </label>
              </div>
              <HistoryChart
                series={[
                  ...(showB0
                    ? [
                        {
                          key: "b0",
                          label: "β₀",
                          values: fullHistory.map((h) => h.intercept),
                          className: "stroke-axis text-axis",
                        },
                      ]
                    : []),
                  ...(showB1
                    ? [
                        {
                          key: "b1",
                          label: "β₁",
                          values: fullHistory.map((h) => h.slope),
                          className: "stroke-point text-point",
                          dashed: true,
                        },
                      ]
                    : []),
                ]}
                xLabel="Iteration"
                yLabel="Parameter value"
              />
            </div>
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
              As learning proceeds the two lines flatten out: the parameters stop changing because
              the gradients have shrunk towards zero.
            </p>
          </Step>

          <Step n={6} title="What is the gradient actually telling us?">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Imagine standing on a hill in fog. The gradient tells you which direction goes
                  uphill most steeply. Gradient descent asks the opposite question: which way is
                  downhill?
                </p>
                <div className="rounded-md border border-border p-4">
                  <FlowDiagram
                    active
                    steps={[
                      "Gradient",
                      "Direction of increasing loss",
                      "Take the opposite direction",
                      "Loss decreases",
                    ]}
                  />
                </div>
                <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                  <div className="rounded-md border border-border p-3">
                    <div className="text-muted-foreground">Gradient for β₁</div>
                    <div className="mt-1 flex items-center gap-2 text-base">
                      {fmt(gradients.gradientSlope, 3)}
                      <ArrowDown
                        className={`size-4 ${gradients.gradientSlope > 0 ? "rotate-180" : ""}`}
                      />
                    </div>
                    <div className="mt-1 font-sans text-muted-foreground">
                      {gradients.gradientSlope > 0
                        ? "Increasing β₁ would increase the loss, so we decrease it."
                        : "Increasing β₁ would decrease the loss, so we increase it."}
                    </div>
                  </div>
                  <div className="rounded-md border border-border p-3">
                    <div className="text-muted-foreground">Gradient for β₀</div>
                    <div className="mt-1 flex items-center gap-2 text-base">
                      {fmt(gradients.gradientIntercept, 3)}
                      <ArrowDown
                        className={`size-4 ${gradients.gradientIntercept > 0 ? "rotate-180" : ""}`}
                      />
                    </div>
                    <div className="mt-1 font-sans text-muted-foreground">
                      {gradients.gradientIntercept > 0
                        ? "Increasing β₀ would increase the loss, so we decrease it."
                        : "Increasing β₀ would decrease the loss, so we increase it."}
                    </div>
                  </div>
                </div>
                <p>
                  A model does not understand anything here. It only follows arithmetic: subtract a
                  fraction of the gradient, over and over.
                </p>
              </div>
              <div className="rounded-md border border-border p-4">
                <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                  Loss landscape — loss vs slope
                </h3>
                <div className="mt-2">
                  <LossLandscape
                    points={points}
                    intercept={params.intercept}
                    slope={params.slope}
                    gradientSlope={gradients.gradientSlope}
                    learningRate={alpha}
                    slopeHistory={fullHistory.map((h) => h.slope)}
                  />
                </div>
              </div>
            </div>
          </Step>

          <Step n={7} title="Predictions right now">
            <ResidualTable rows={rows} mse={mse} selectedId={selectedId} onSelect={setSelectedId} />
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
              This is the same table as the Linear Regression page, but it now refreshes on every
              iteration. {selected ? `Selected point: x = ${fmt(selected.x)}, error = ${fmt(selected.error, 3)}.` : "Hover a row to highlight it."}
            </p>
          </Step>

          <Step n={8} title="Where did we end up?">
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted text-left">
                  <tr>
                    <th className="px-3 py-2 font-medium" />
                    <th className="px-3 py-2 font-medium">Gradient Descent</th>
                    <th className="px-3 py-2 font-medium">OLS</th>
                    <th className="px-3 py-2 font-medium">Difference</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  <tr className="border-t border-border">
                    <td className="px-3 py-1.5">β₀</td>
                    <td className="px-3 py-1.5">{fmt(params.intercept, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(ols.intercept, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(params.intercept - ols.intercept, 4)}</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="px-3 py-1.5">β₁</td>
                    <td className="px-3 py-1.5">{fmt(params.slope, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(ols.slope, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(params.slope - ols.slope, 4)}</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="px-3 py-1.5">MSE</td>
                    <td className="px-3 py-1.5">{fmt(mse, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(olsMse, 4)}</td>
                    <td className="px-3 py-1.5">{fmt(mse - olsMse, 4)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
              OLS calculates the least-squares solution directly for this simple problem. Gradient
              descent approaches the same minimum through repeated parameter updates. After enough
              iterations the two should be close — not exactly identical, because we stop after a
              finite number of steps and computers round numbers.
            </p>
          </Step>

          <Step n={9} title="Common gradient descent mistakes">
            <div className="grid gap-3 md:grid-cols-2">
              {[
                [
                  "Learning rate too small",
                  "Each step barely moves the parameters, so convergence takes an enormous number of iterations.",
                ],
                [
                  "Learning rate too large",
                  "Updates jump past the minimum; the loss can bounce upwards or explode to infinity.",
                ],
                [
                  "Wrong gradient sign",
                  "Adding the gradient instead of subtracting it walks uphill — the model gets steadily worse.",
                ],
                [
                  "Forgetting to update parameters",
                  "Computing gradients but never applying them means nothing ever changes.",
                ],
                [
                  "Mixing up loss and gradient",
                  "The loss tells us how bad the model is. The gradient tells us how the loss changes when a parameter changes.",
                ],
                [
                  "Reading a flat loss curve as success",
                  "A flat curve with a large loss usually means the steps are too small, not that we have finished.",
                ],
              ].map(([title, body]) => (
                <div key={title} className="rounded-md border border-border p-4">
                  <h3 className="text-sm font-medium">{title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{body}</p>
                </div>
              ))}
            </div>
          </Step>

          <Step n={10} title="From mathematics to code">
            <div className="grid gap-4 md:grid-cols-2">
              {[
                ["Prediction", "ŷ = β₀ + β₁x", "const prediction = intercept + slope * x;"],
                [
                  "Gradient for intercept",
                  "∂MSE/∂β₀ = -(2/n) Σ(yᵢ − ŷᵢ)",
                  "gradientIntercept =\n  (-2 / n) * residualSum;",
                ],
                [
                  "Gradient for slope",
                  "∂MSE/∂β₁ = -(2/n) Σxᵢ(yᵢ − ŷᵢ)",
                  "gradientSlope =\n  (-2 / n) * weightedResidualSum;",
                ],
                ["Update", "β = β − α × gradient", "parameter -= learningRate * gradient;"],
              ].map(([title, math, code]) => (
                <div key={title} className="rounded-md border border-border p-4">
                  <h3 className="text-sm font-medium">{title}</h3>
                  <pre className="mt-2 rounded-md bg-muted p-3 font-mono text-xs">{math}</pre>
                  <pre className="mt-2 rounded-md bg-muted p-3 font-mono text-xs">{code}</pre>
                </div>
              ))}
            </div>
          </Step>

          <Step n={11} title="Keeping the words apart">
            <div className="grid gap-4 text-sm md:grid-cols-4">
              <div className="rounded-md border border-border p-4">
                <div className="font-medium">
                  <Term name="model">Linear Regression</Term>
                </div>
                <p className="mt-1 font-mono text-xs">ŷ = β₀ + β₁x</p>
                <p className="mt-1 text-muted-foreground">The model — the rule itself.</p>
              </div>
              <div className="rounded-md border border-border p-4">
                <div className="font-medium">
                  <Term name="loss">Loss function</Term>
                </div>
                <p className="mt-1 font-mono text-xs">MSE = (1/n) Σ(y − ŷ)²</p>
                <p className="mt-1 text-muted-foreground">
                  Measures how bad the current parameters are.
                </p>
              </div>
              <div className="rounded-md border border-border p-4">
                <div className="font-medium">Gradient</div>
                <p className="mt-1 font-mono text-xs">∂MSE/∂β</p>
                <p className="mt-1 text-muted-foreground">
                  How the loss changes with respect to a parameter.
                </p>
              </div>
              <div className="rounded-md border border-border p-4">
                <div className="font-medium">Gradient descent</div>
                <p className="mt-1 font-mono text-xs">β ← β − α∇</p>
                <p className="mt-1 text-muted-foreground">
                  An optimisation algorithm — not the model.
                </p>
              </div>
            </div>
          </Step>

          <Step n={12} title="Can you explain what just happened?">
            <div className="space-y-2">
              {[
                ["What does β₀ represent?", "The intercept: the prediction when x is 0 — it shifts the line up and down."],
                ["What does β₁ represent?", "The slope: how much the prediction changes when x increases by 1."],
                ["What does MSE measure?", "The average squared gap between the actual values and the predictions."],
                ["What does the gradient tell us?", "How the loss changes if a parameter increases — the uphill direction."],
                ["Why do we subtract the gradient?", "Because uphill increases the loss; we want the opposite direction."],
                ["What does the learning rate control?", "How big each parameter update is."],
                ["What happens if the learning rate is too large?", "The updates overshoot the minimum; the loss can rise or become unstable."],
                ["What happens after many successful iterations?", "The gradients shrink, the parameters stabilise, and the line lands near the OLS solution."],
              ].map(([q, a], i) => (
                <details key={i} className="rounded-md border border-border px-4 py-3">
                  <summary className="cursor-pointer text-sm font-medium">
                    {i + 1}. {q}
                  </summary>
                  <p className="mt-2 text-sm text-muted-foreground">{a}</p>
                </details>
              ))}
            </div>
          </Step>

          <Step n={13} title="Gradient descent in one picture">
            <div className="rounded-md border border-border p-6">
              <ol className="mx-auto flex max-w-xs flex-col items-center gap-1 text-sm">
                {[
                  "DATA",
                  "Prediction",
                  "Residual",
                  "MSE",
                  "Gradient",
                  "Opposite direction",
                  "Parameter update",
                  "New model",
                  "Repeat",
                  "Better model",
                ].map((label, i, arr) => (
                  <li key={label} className="flex flex-col items-center gap-1">
                    <span className="rounded-full border border-border px-4 py-1">{label}</span>
                    {i < arr.length - 1 && <ArrowDown className="size-3 text-muted-foreground" />}
                  </li>
                ))}
              </ol>
            </div>
          </Step>

          <footer className="border-t border-border pt-6 text-xs text-muted-foreground">
            Regression Lab — Phase 2. Every parameter, gradient, loss value and animation frame on
            this page comes from the gradient descent implementation in TypeScript. Nothing is
            interpolated or pre-recorded.
          </footer>
        </main>
      </div>
    </TooltipProvider>
  );
}
