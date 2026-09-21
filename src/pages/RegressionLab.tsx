import { useEffect, useMemo, useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import {
  buildBreakdown,
  calculateMSE,
  fitOLS,
  olsSteps,
  predict,
  type Point,
} from "@/algorithms/linearRegression";
import { DatasetEditor } from "@/components/DatasetEditor";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { FlowDiagram } from "@/components/FlowDiagram";
import { FormulaPanel } from "@/components/FormulaPanel";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ModelControls } from "@/components/ModelControls";
import { RegressionChart } from "@/components/RegressionChart";
import { ResidualTable } from "@/components/ResidualTable";
import { SiteNav } from "@/components/SiteNav";
import { Term } from "@/components/Term";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { equationString, fmt } from "@/utils/calculations";
import { defaultDataset, nextId, PRESETS } from "@/utils/dataset";

const DEFAULT_EXPLANATION =
  "The line below is the model. Change the data or the parameters and every number on this page is recalculated from scratch in your browser.";

function Step({
  n,
  title,
  children,
  className = "",
}: {
  n: number;
  title: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`scroll-mt-8 ${className}`}>
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-xs text-muted-foreground">{String(n).padStart(2, "0")}</span>
        <h2 className="font-display text-2xl">{title}</h2>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

export default function RegressionLab() {
  const [points, setPoints] = useState<Point[]>(defaultDataset);
  const [manual, setManual] = useState(false);
  const [manualParams, setManualParams] = useState({ slope: 1, intercept: 0 });
  const [showResiduals, setShowResiduals] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [preset, setPreset] = useState("good-fit");
  const [explanation, setExplanation] = useState(DEFAULT_EXPLANATION);
  const explainTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => void (explainTimer.current && clearTimeout(explainTimer.current)), []);

  const explain = (text: string) => {
    setExplanation(text);
    if (explainTimer.current) clearTimeout(explainTimer.current);
    explainTimer.current = setTimeout(() => setExplanation(DEFAULT_EXPLANATION), 9000);
  };

  const best = useMemo(() => fitOLS(points), [points]);
  const params = manual ? manualParams : best;
  const rows = useMemo(() => buildBreakdown(points, params), [points, params]);
  const mse = useMemo(() => calculateMSE(points, params), [points, params]);
  const bestMse = useMemo(() => calculateMSE(points, best), [points, best]);
  const steps = useMemo(() => olsSteps(points), [points]);
  const selected = rows.find((r) => r.id === selectedId) ?? null;

  const gap = mse - bestMse;
  const close = gap <= Math.max(bestMse * 0.15, 0.05);

  const updateManual = (patch: Partial<typeof manualParams>, message: string) => {
    setManual(true);
    setManualParams((p) => ({ ...p, ...patch }));
    explain(message);
  };

  const setDataset = (next: Point[]) => {
    setPoints(next);
    setSelectedId(null);
  };

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPreset(id);
    setDataset(p.points());
    if (p.manual) {
      setManual(true);
      setManualParams(p.manual);
      explain(
        "This line was set deliberately badly. Look at how long the residual lines are and how large the MSE becomes.",
      );
    } else {
      setManual(false);
      explain(`Preset loaded: ${p.caption}`);
    }
  };

  const resetEverything = () => {
    setPoints(defaultDataset());
    setManual(false);
    setManualParams({ slope: 1, intercept: 0 });
    setShowResiduals(false);
    setSelectedId(null);
    setPreset("good-fit");
    setExplanation(DEFAULT_EXPLANATION);
  };

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Regression Lab</h1>
              <p className="mt-1 text-muted-foreground">
                See how Linear Regression learns from data.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <SiteNav />
              <Button variant="ghost" size="sm" onClick={resetEverything}>
                <RotateCcw className="size-3.5" /> Reset everything
              </Button>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-14 px-5 py-10">
          {/* 1 — Data + 2 — the line */}
          <div className="grid gap-8 lg:grid-cols-[300px_1fr]">
            <div className="space-y-8">
              <Step n={1} title="The data">
                <p className="mb-3 text-sm text-muted-foreground">
                  House Size vs Price — here simply <strong>x = Input</strong> and{" "}
                  <strong>y = Output</strong>. Edit anything; the model refits instantly.
                </p>
                <DatasetEditor
                  points={points}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  onChange={(id, key, value) => {
                    setPoints((ps) =>
                      ps.map((p) => (p.id === id ? { ...p, [key]: Number.isFinite(value) ? value : 0 } : p)),
                    );
                    explain(
                      "The dataset changed, so the best-fitting line, every prediction, every residual and the MSE were all recalculated.",
                    );
                  }}
                  onAdd={() => {
                    const lastX = points.length ? Math.max(...points.map((p) => p.x)) : 0;
                    setPoints((ps) => [...ps, { id: nextId(), x: lastX + 1, y: 0 }]);
                  }}
                  onDelete={(id) => setPoints((ps) => ps.filter((p) => p.id !== id))}
                  onReset={() => setDataset(defaultDataset())}
                />

                <div className="mt-5">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Experiment presets
                  </h3>
                  <div className="mt-2 space-y-1.5">
                    {PRESETS.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => applyPreset(p.id)}
                        className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                          preset === p.id
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
              </Step>

              <Step n={2} title="Draw a line">
                <ModelControls
                  manual={manual}
                  onManualChange={(m) => {
                    setManual(m);
                    if (m) {
                      setManualParams(params);
                      explain(
                        "You now control the line by hand. Try making it deliberately bad and watch the errors grow.",
                      );
                    } else {
                      explain(
                        "Back to automatic fit: the line is the Ordinary Least Squares solution, calculated directly from the data.",
                      );
                    }
                  }}
                  slope={params.slope}
                  intercept={params.intercept}
                  onSlope={(v) =>
                    updateManual(
                      { slope: v },
                      "You are changing β₁. This controls how steep the regression line is.",
                    )
                  }
                  onIntercept={(v) =>
                    updateManual(
                      { intercept: v },
                      "You are changing β₀. This moves the line up or down without changing its slope.",
                    )
                  }
                  onFitBest={() => {
                    setManual(false);
                    explain(
                      "The line jumped back to the one with the smallest possible MSE for this data.",
                    );
                  }}
                  showResiduals={showResiduals}
                  onShowResiduals={(v) => {
                    setShowResiduals(v);
                    if (v)
                      explain(
                        "These lines show the difference between what actually happened and what the model predicted.",
                      );
                  }}
                />
              </Step>
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

              <div className="rounded-md border border-border p-4">
                <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                  Move the line and this happens
                </h3>
                <div className="mt-3">
                  <FlowDiagram
                    active={manual}
                    steps={[
                      "Move line",
                      "Predictions change",
                      "Residuals change",
                      "Squared errors change",
                      "MSE changes",
                    ]}
                  />
                </div>
              </div>

              <MetricsPanel slope={params.slope} intercept={params.intercept} mse={mse} />
              <ExplanationPanel message={explanation} />
            </div>
          </div>

          {/* Our model */}
          <Step n={3} title="Our model">
            <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <div>
                <div className="rounded-md bg-muted p-5 font-mono text-2xl">
                  {equationString(params.intercept, params.slope)}
                </div>
                <div className="mt-3 font-mono text-sm text-muted-foreground">
                  β₀ = {fmt(params.intercept)} <br /> β₁ = {fmt(params.slope)}
                </div>
              </div>
              <dl className="space-y-4 text-sm">
                <div>
                  <dt className="font-medium">Intercept (β₀)</dt>
                  <dd className="text-muted-foreground">
                    Where the line crosses the y-axis — the prediction when the input is 0.
                  </dd>
                </div>
                <div>
                  <dt className="font-medium">Slope (β₁)</dt>
                  <dd className="text-muted-foreground">
                    How much the prediction changes when x increases by 1. Right now, +1 in x means{" "}
                    {params.slope >= 0 ? "+" : "−"}
                    {fmt(Math.abs(params.slope))} in ŷ.
                  </dd>
                </div>
                <div className="rounded-md border border-border p-3 text-xs leading-relaxed text-muted-foreground">
                  Words worth keeping apart: the <Term name="model" /> is the rule, the{" "}
                  <Term name="parameters" /> are β₀ and β₁, a <Term name="prediction" /> is ŷ, the{" "}
                  <Term name="error" /> is y − ŷ, and the <Term name="loss" /> is the MSE. Hover any
                  of them.
                </div>
              </dl>
            </div>
          </Step>

          {/* Predictions */}
          <Step n={4} title="Make a prediction">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-md border border-border p-5 font-mono text-sm">
                {selected ? (
                  <div className="space-y-1">
                    <div>Input x: {fmt(selected.x)}</div>
                    <div>Actual y: {fmt(selected.actual)}</div>
                    <div>
                      Prediction ŷ: {fmt(params.intercept)} + {fmt(params.slope)} ×{" "}
                      {fmt(selected.x)} = {fmt(selected.prediction)}
                    </div>
                    <div>
                      Error: {fmt(selected.actual)} − {fmt(selected.prediction)} ={" "}
                      {fmt(selected.error)}
                    </div>
                  </div>
                ) : (
                  <p className="font-sans text-muted-foreground">
                    Click or hover a point on the chart to see its prediction worked out.
                  </p>
                )}
              </div>
              <div className="text-sm text-muted-foreground">
                <p>
                  A prediction is made by putting the input into the model:{" "}
                  <span className="font-mono text-foreground">ŷ = β₀ + β₁x</span>. Nothing is stored
                  or looked up — the number is computed from the two parameters every time.
                </p>
                <p className="mt-3 font-mono text-xs">
                  predict(x) = {fmt(params.intercept)} + {fmt(params.slope)} · x → e.g. x = 5 gives ŷ
                  = {fmt(predict(params, 5))}
                </p>
              </div>
            </div>
          </Step>

          {/* Residuals */}
          <Step n={5} title="Residuals: how wrong is each prediction?">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  A <strong className="text-foreground">residual</strong> is the vertical distance
                  between the actual point and the line:
                </p>
                <pre className="rounded-md bg-muted p-4 font-mono text-xs">{`Residual = Actual − Prediction`}</pre>
                <p>
                  Turn on <strong className="text-foreground">Show residuals</strong> to draw one for
                  every observation. Residuals above the line are drawn as solid lines; residuals
                  below the line are dashed, so you can tell them apart without relying on colour.
                </p>
              </div>
              <div className="rounded-md border border-border p-5 font-mono text-sm">
                {selected ? (
                  <>
                    <div>Actual: {fmt(selected.actual)}</div>
                    <div>Prediction: {fmt(selected.prediction)}</div>
                    <div className="mt-3">Residual:</div>
                    <div>
                      {fmt(selected.actual)} − {fmt(selected.prediction)} = {fmt(selected.error)}
                    </div>
                    <div className="mt-3">Squared residual: {fmt(selected.squaredError)}</div>
                  </>
                ) : (
                  <p className="font-sans text-muted-foreground">
                    Select a point to see its residual written out.
                  </p>
                )}
              </div>
            </div>
          </Step>

          {/* MSE */}
          <Step n={6} title="How good is our line?">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3 text-sm text-muted-foreground">
                <pre className="rounded-md bg-muted p-4 font-mono text-xs">{`MSE = (1/n) Σ(yᵢ − ŷᵢ)²`}</pre>
                <p>
                  <strong className="text-foreground">
                    MSE measures how far our predictions are from the actual values.
                  </strong>{" "}
                  Square each error (so negatives don't cancel positives and big misses count more),
                  then take the average.
                </p>
                <div className="mt-2">
                  <FlowDiagram
                    active={!!selected}
                    steps={["Actual", "Prediction", "Residual", "Squared residual"]}
                  />
                </div>
                {selected && (
                  <div className="font-mono text-xs">
                    {fmt(selected.actual)} → {fmt(selected.prediction)} → {fmt(selected.error)} →{" "}
                    {fmt(selected.squaredError)}
                  </div>
                )}
              </div>
              <div className="rounded-md border border-border p-5">
                <h3 className="font-display text-lg">Can you find a better line?</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Switch to manual mode and move the sliders. The two numbers below update live.
                </p>
                <div className="mt-4 grid grid-cols-2 gap-4 font-mono">
                  <div>
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">
                      Your MSE
                    </div>
                    <div className="text-2xl tabular-nums">{fmt(mse, 3)}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">
                      Best MSE
                    </div>
                    <div className="text-2xl tabular-nums">{fmt(bestMse, 3)}</div>
                  </div>
                </div>
                <p className="mt-4 text-sm">
                  {close
                    ? "Great! You're getting close to the best-fitting line."
                    : "This line produces larger prediction errors. Try adjusting the slope or intercept."}
                </p>
              </div>
            </div>
            <div className="mt-6">
              <ResidualTable
                rows={rows}
                mse={mse}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>
          </Step>

          {/* Why this line */}
          <Step n={7} title="Why this line?">
            <p className="max-w-2xl text-sm text-muted-foreground">
              Linear Regression searches for the line that minimises the total squared error. Out of
              every line you could possibly draw, exactly one makes the MSE as small as it can be —
              and that is the one we call the best fit.
            </p>
            <div className="mt-4">
              <FlowDiagram
                steps={[
                  "Many possible lines",
                  "Calculate MSE for each",
                  "Find the smallest MSE",
                  "Best-fitting line",
                ]}
              />
            </div>
          </Step>

          <Step n={8} title="Ordinary Least Squares">
            <FormulaPanel steps={steps} codeOpenByDefault />
          </Step>

          <footer className="border-t border-border pt-6 text-xs text-muted-foreground">
            Regression Lab — Phase 1. Every value on this page is computed in TypeScript from the
            dataset above. No models, no server, no stored answers.
          </footer>
        </main>
      </div>
    </TooltipProvider>
  );
}
