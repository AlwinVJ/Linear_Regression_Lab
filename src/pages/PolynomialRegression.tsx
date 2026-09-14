import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Dices, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import type { Point } from "@/algorithms/linearRegression";
import {
  fitPolynomial,
  polyBreakdown,
  polyDiverging,
  polyEquation,
  polyGradientDescentStep,
  polyGradients,
  polyMSE,
} from "@/algorithms/polynomialRegression";
import { CoefficientPanel } from "@/components/CoefficientPanel";
import { ComplexityChart, type ComplexityRow } from "@/components/ComplexityChart";
import { DegreeControl } from "@/components/DegreeControl";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { FeatureExpansionTable } from "@/components/FeatureExpansionTable";
import { FlowDiagram } from "@/components/FlowDiagram";
import { PolynomialChart } from "@/components/PolynomialChart";
import { SiteNav } from "@/components/SiteNav";
import { TrainTestMetrics } from "@/components/TrainTestMetrics";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { TooltipProvider } from "@/components/ui/tooltip";
import { fmt } from "@/utils/calculations";
import { DEFAULT_POLY_SEED, POLY_PRESETS } from "@/utils/dataset";
import { trainTestSplit } from "@/utils/metrics";
import { betaLabel, powerLabel } from "@/utils/polynomialFeatures";

const MAX_DEGREE = 10;
const TEST_RATIO = 0.3;

function Step({ n, title, children }: { n: number; title: React.ReactNode; children: React.ReactNode }) {
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

const CHECKPOINTS: { q: string; a: string }[] = [
  {
    q: "What is a polynomial feature?",
    a: "A transformed copy of the input, such as x², x³ or x⁴. It is computed from x before the model sees it.",
  },
  {
    q: "Why does adding x² allow the model to represent curves?",
    a: "Because the prediction becomes β₀ + β₁x + β₂x². The x² term bends the shape, so the output no longer has to change at a constant rate.",
  },
  {
    q: "Why is polynomial regression still linear with respect to its coefficients?",
    a: "The prediction is a weighted sum of fixed feature values. Each coefficient appears once, multiplied by a number — never squared or multiplied by another coefficient. That is exactly what 'linear in the parameters' means.",
  },
  {
    q: "What happens when polynomial degree increases?",
    a: "The model gains more features and more coefficients, so it can take more shapes. Training error usually falls, because the model has more ways to pass near the training points.",
  },
  {
    q: "Why can high-degree models overfit?",
    a: "With enough flexibility the curve can bend to follow individual observations, including the noise in them. Noise does not repeat in new data, so the extra bends hurt on unseen points.",
  },
  {
    q: "Why should test data not be used to fit the model?",
    a: "Because then it is no longer unseen. Test error would only tell you how well the model memorised, not how well it generalises.",
  },
  {
    q: "What is the difference between training error and test error?",
    a: "Training error is measured on the observations used to choose the coefficients. Test error is measured on observations held back from fitting.",
  },
  {
    q: "Why might a model with lower training MSE still be worse?",
    a: "Lower training error can come from fitting noise. If test error is higher, the model describes this particular sample better but the underlying relationship worse.",
  },
  {
    q: "What is the relationship between polynomial regression and linear regression?",
    a: "Polynomial regression IS linear regression — applied to expanded features. You build x, x², x³ … and then run exactly the same least-squares or gradient descent fitting.",
  },
];

export default function PolynomialRegressionLab() {
  const [presetId, setPresetId] = useState(POLY_PRESETS[0]!.id);
  const preset = POLY_PRESETS.find((p) => p.id === presetId) ?? POLY_PRESETS[0]!;
  const [seed, setSeed] = useState(DEFAULT_POLY_SEED);
  const [noise, setNoise] = useState(preset.noise);
  const [degree, setDegree] = useState(1);
  const [method, setMethod] = useState<"ls" | "gd">("ls");
  const [showSplit, setShowSplit] = useState(false);
  const [showResiduals, setShowResiduals] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState(
    "Degree 1 is a straight line. Look at how badly it follows this curved data, then raise the degree.",
  );

  const points = useMemo(() => preset.build(seed, noise), [preset, seed, noise]);
  const split = useMemo(() => trainTestSplit(points, TEST_RATIO, seed), [points, seed]);
  const train = showSplit ? split.train : points;
  const test = showSplit ? split.test : [];

  /* ---------------- Least squares ---------------- */
  const lsFit = useMemo(() => fitPolynomial(train, degree), [train, degree]);

  /* ---------------- Gradient descent -------------- */
  const [alpha, setAlpha] = useState(0.0005);
  const [gdCoefs, setGdCoefs] = useState<number[]>(() => new Array(degree + 1).fill(0));
  const [gdIteration, setGdIteration] = useState(0);
  const [gdDiverged, setGdDiverged] = useState(false);
  const [playing, setPlaying] = useState(false);

  const resetGd = useCallback(() => {
    setPlaying(false);
    setGdCoefs(new Array(degree + 1).fill(0));
    setGdIteration(0);
    setGdDiverged(false);
  }, [degree]);

  // Any change of degree, data or split invalidates the gradient descent run.
  useEffect(() => {
    resetGd();
  }, [resetGd, train]);

  const gdStep = useCallback(() => {
    setGdCoefs((current) => {
      const result = polyGradientDescentStep(train, current, alpha);
      if (polyDiverging(result.after, result.mseAfter)) {
        setGdDiverged(true);
        setPlaying(false);
        setMessage(
          "Numerical instability detected. Try a lower degree or a different learning rate.",
        );
        return current;
      }
      setGdIteration((i) => i + 1);
      return result.after;
    });
  }, [train, alpha]);

  const rafRef = useRef<number | null>(null);
  const lastTick = useRef(0);
  useEffect(() => {
    if (!playing) return;
    const loop = (t: number) => {
      if (t - lastTick.current > 40) {
        lastTick.current = t;
        gdStep();
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, gdStep]);

  /* ---------------- Active model ------------------ */
  const coefficients = method === "ls" ? lsFit.coefficients : gdCoefs;
  const unstable = method === "ls" ? !lsFit.ok : gdDiverged;
  const instabilityReason =
    method === "ls"
      ? lsFit.reason
      : gdDiverged
        ? "Numerical instability detected. Try a lower degree or a different learning rate."
        : undefined;

  const trainMse = useMemo(() => polyMSE(train, coefficients), [train, coefficients]);
  const testMse = useMemo(
    () => (showSplit && test.length > 0 ? polyMSE(test, coefficients) : null),
    [showSplit, test, coefficients],
  );
  const rows = useMemo(
    () => polyBreakdown([...train, ...test], coefficients, degree),
    [train, test, coefficients, degree],
  );
  const selected = rows.find((r) => r.id === selectedId) ?? null;
  const gradients = useMemo(() => polyGradients(train, coefficients), [train, coefficients]);

  /* ---------------- Complexity sweep -------------- */
  const complexity: ComplexityRow[] = useMemo(() => {
    const out: ComplexityRow[] = [];
    for (let d = 1; d <= MAX_DEGREE; d++) {
      const fit = fitPolynomial(train, d);
      out.push({
        degree: d,
        trainMse: fit.ok ? polyMSE(train, fit.coefficients) : Number.NaN,
        testMse: fit.ok && showSplit && test.length > 0 ? polyMSE(test, fit.coefficients) : null,
        ok: fit.ok,
      });
    }
    return out;
  }, [train, test, showSplit]);

  const applyPreset = (id: string) => {
    const p = POLY_PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPresetId(id);
    setNoise(p.noise);
    setSelectedId(null);
    setMessage(`${p.caption} ${p.expectation}`);
  };

  const regenerate = () => {
    setSeed(Math.floor(Math.random() * 900000) + 100000);
    setSelectedId(null);
    setMessage("New dataset generated from a new random seed. The split is regenerated with it.");
  };

  const changeDegree = (d: number) => {
    setDegree(d);
    setMessage(
      d === 1
        ? "Degree 1 means only the feature x. The model can only draw a straight line."
        : `Degree ${d} means the features x${Array.from({ length: d - 1 }, (_, i) => `, ${powerLabel(i + 2)}`).join("")}. The model refits and every number below is recalculated.`,
    );
  };

  const featureList = useMemo(
    () => Array.from({ length: degree }, (_, i) => powerLabel(i + 1)).join(", "),
    [degree],
  );

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Polynomial Regression</h1>
              <p className="mt-1 text-muted-foreground">
                What if a straight line isn&rsquo;t enough?
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <SiteNav />
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-14 px-5 py-10">
          {/* 01 — the problem */}
          <Step n={1} title="What if a straight line isn't enough?">
            <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
              <div className="rounded-md border border-border p-4">
                <PolynomialChart
                  train={train}
                  test={test}
                  coefficients={coefficients}
                  showResiduals={showResiduals}
                  showSplit={showSplit}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
                <p className="mt-3 font-mono text-sm">
                  {unstable ? "—" : polyEquation(coefficients)}
                </p>
              </div>
              <div className="space-y-4">
                <p className="text-sm leading-relaxed">
                  A straight line can only represent a linear relationship: a constant change in y
                  for every unit of x. This data curves. The line has nowhere to bend, so it misses
                  the pattern no matter where you place it.
                </p>
                <p className="text-sm leading-relaxed">
                  The fix is not a different kind of model. It is a different set of{" "}
                  <strong>features</strong>. Give the same linear model x² as well as x, and it can
                  curve.
                </p>
                <div className="rounded-md border border-border p-4">
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">
                    Dataset
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {POLY_PRESETS.map((p) => (
                      <Button
                        key={p.id}
                        size="sm"
                        variant={p.id === presetId ? "default" : "outline"}
                        onClick={() => applyPreset(p.id)}
                      >
                        {p.name}
                      </Button>
                    ))}
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">{preset.caption}</p>
                  <label className="mt-4 block text-xs uppercase tracking-widest text-muted-foreground">
                    Noise: {fmt(noise, 2)}
                  </label>
                  <Slider
                    className="mt-2"
                    min={0}
                    max={8}
                    step={0.25}
                    value={[noise]}
                    onValueChange={([v]) => setNoise(v ?? 0)}
                  />
                  <div className="mt-4 flex items-center justify-between gap-3">
                    <span className="font-mono text-xs text-muted-foreground">
                      Random seed: {seed}
                    </span>
                    <Button size="sm" variant="ghost" onClick={regenerate}>
                      <Dices className="size-3.5" /> Regenerate
                    </Button>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-6 rounded-md border border-border p-4">
                  <label className="flex items-center gap-2 text-sm">
                    <Switch checked={showResiduals} onCheckedChange={setShowResiduals} />
                    Show residuals
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <Switch
                      checked={showSplit}
                      onCheckedChange={(v) => {
                        setShowSplit(v);
                        setMessage(
                          v
                            ? "The model is now fitted on the training points only. The square markers are held back and used purely to measure generalisation."
                            : "Train/test split off — every point is used for fitting again.",
                        );
                      }}
                    />
                    Show train/test split
                  </label>
                </div>
                <ExplanationPanel message={message} />
              </div>
            </div>
          </Step>

          {/* 02 — degree control */}
          <Step n={2} title="Polynomial degree">
            <div className="grid gap-6 lg:grid-cols-2">
              <DegreeControl degree={degree} onChange={changeDegree} max={MAX_DEGREE} />
              <div className="rounded-md border border-border p-4">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">
                  Fitting method
                </div>
                <div className="mt-3 flex gap-2">
                  <Button
                    size="sm"
                    variant={method === "ls" ? "default" : "outline"}
                    onClick={() => setMethod("ls")}
                  >
                    Least squares
                  </Button>
                  <Button
                    size="sm"
                    variant={method === "gd" ? "default" : "outline"}
                    onClick={() => setMethod("gd")}
                  >
                    Gradient descent
                  </Button>
                </div>
                {method === "ls" ? (
                  <p className="mt-3 text-sm text-muted-foreground">
                    The coefficients are solved directly from the feature matrix using a QR
                    decomposition — a numerically stable way to answer &ldquo;which coefficients
                    make the squared errors as small as possible?&rdquo;
                  </p>
                ) : (
                  <div className="mt-3 space-y-3">
                    <p className="text-sm text-muted-foreground">
                      The same gradient descent loop from the previous page, now with one
                      coefficient per polynomial feature.
                    </p>
                    <label className="block text-xs uppercase tracking-widest text-muted-foreground">
                      Learning rate α: {alpha}
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {[0.00001, 0.0001, 0.0005, 0.001].map((a) => (
                        <Button
                          key={a}
                          size="sm"
                          variant={alpha === a ? "default" : "outline"}
                          onClick={() => setAlpha(a)}
                        >
                          {a}
                        </Button>
                      ))}
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" onClick={gdStep} disabled={gdDiverged}>
                        <StepForward className="size-3.5" /> Step once
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setPlaying((p) => !p)}
                        disabled={gdDiverged}
                      >
                        {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
                        {playing ? "Pause" : "Start learning"}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={resetGd}>
                        <RotateCcw className="size-3.5" /> Reset
                      </Button>
                      <span className="font-mono text-xs text-muted-foreground">
                        iteration {gdIteration}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {unstable && (
              <p className="mt-4 rounded-md border border-resid-neg bg-resid-neg/10 px-4 py-3 text-sm">
                {instabilityReason}
              </p>
            )}

            <div className="mt-6">
              <TrainTestMetrics
                degree={degree}
                trainMse={trainMse}
                testMse={testMse}
                trainCount={train.length}
                testCount={test.length}
                showSplit={showSplit}
              />
            </div>

            <div className="mt-6 rounded-md border border-border p-4">
              <div className="text-xs uppercase tracking-widest text-muted-foreground">
                Current model equation
              </div>
              <p className="mt-2 font-mono text-lg">{unstable ? "—" : polyEquation(coefficients)}</p>
              <p className="mt-2 text-sm text-muted-foreground">
                General form for degree {degree}: ŷ = β₀
                {Array.from({ length: degree }, (_, i) => ` + ${betaLabel(i + 1)}${powerLabel(i + 1)}`).join("")}
              </p>
            </div>
          </Step>

          {/* 03 — feature expansion */}
          <Step n={3} title="Feature expansion — what the model actually receives">
            <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
              <div className="space-y-4">
                <p className="text-sm leading-relaxed">
                  Polynomial regression does not learn &ldquo;a curve&rdquo;. It receives a table of
                  numbers. For degree {degree}, each observation is turned into the features{" "}
                  <span className="font-mono">{featureList}</span>, and a plain linear model is
                  fitted to those columns.
                </p>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Original input
      ↓
      x
      ↓
Feature expansion
      ↓
[x, x², x³, ...]
      ↓
Linear model
      ↓
Prediction`}</pre>
                <div className="rounded-md border border-axis bg-accent/10 p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Important
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed">
                    Polynomial regression is still a <strong>linear model with respect to its
                    coefficients</strong>. In ŷ = β₀ + β₁x + β₂x², each of β₀, β₁ and β₂ appears
                    once, multiplied by a fixed number. The relationship between x and ŷ is curved;
                    the relationship between the coefficients and ŷ is not.
                  </p>
                </div>
              </div>
              <div>
                <FeatureExpansionTable points={train} degree={degree} />
                <p className="mt-3 text-sm text-muted-foreground">
                  Columns appear and disappear as you change the degree. These values are the actual
                  inputs to the fit.
                </p>
              </div>
            </div>
          </Step>

          {/* 04 — coefficients */}
          <Step n={4} title="Learned coefficients">
            <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
              <CoefficientPanel coefficients={coefficients} />
              <div className="space-y-4">
                <div className="rounded-md border border-border">
                  <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
                    Inspect an observation
                  </h3>
                  {selected ? (
                    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 px-4 py-3 font-mono text-sm">
                      <dt className="text-muted-foreground">x</dt>
                      <dd className="text-right">{fmt(selected.x)}</dd>
                      <dt className="text-muted-foreground">actual y</dt>
                      <dd className="text-right">{fmt(selected.actual)}</dd>
                      <dt className="text-muted-foreground">prediction ŷ</dt>
                      <dd className="text-right">{fmt(selected.prediction)}</dd>
                      <dt className="text-muted-foreground">residual e = y − ŷ</dt>
                      <dd className="text-right">{fmt(selected.error)}</dd>
                      <dt className="text-muted-foreground">squared error e²</dt>
                      <dd className="text-right">{fmt(selected.squaredError)}</dd>
                    </dl>
                  ) : (
                    <p className="px-4 py-3 text-sm text-muted-foreground">
                      Click a point on the chart to see its prediction, residual and squared error.
                    </p>
                  )}
                </div>
                <p className="text-sm leading-relaxed">
                  The residual is the vertical difference between the actual value and the model
                  prediction: <span className="font-mono">eᵢ = yᵢ − ŷᵢ</span>. The MSE is the
                  average of the squared residuals.
                </p>
              </div>
            </div>
          </Step>

          {/* 05 — complexity vs error */}
          <Step n={5} title="Model complexity vs error">
            <div className="rounded-md border border-border p-4">
              <ComplexityChart
                rows={complexity}
                selectedDegree={degree}
                showTest={showSplit && test.length > 0}
                onSelectDegree={changeDegree}
              />
            </div>
            <div className="mt-4 overflow-x-auto rounded-md border border-border">
              <table className="w-full min-w-[420px] text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/40 text-xs uppercase tracking-widest text-muted-foreground">
                    <th className="px-3 py-2 text-left">Degree</th>
                    <th className="px-3 py-2 text-right">Training MSE</th>
                    <th className="px-3 py-2 text-right">Test MSE</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {complexity.map((r) => (
                    <tr
                      key={r.degree}
                      className={`cursor-pointer border-b border-border/60 last:border-0 ${
                        r.degree === degree ? "bg-accent/20" : ""
                      }`}
                      onClick={() => changeDegree(r.degree)}
                    >
                      <td className="px-3 py-1.5">{r.degree}</td>
                      <td className="px-3 py-1.5 text-right">
                        {r.ok ? fmt(r.trainMse, 4) : "unstable"}
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {r.testMse === null ? "—" : fmt(r.testMse, 4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              Every number here is refitted on the training data for that degree. Turn on the
              train/test split to see the test column — that is the column that tells you whether
              extra complexity is paying off.
            </p>
          </Step>

          {/* 06 — underfit / good fit / overfit */}
          <Step n={6} title="Underfitting, a good fit, and overfitting">
            <div className="grid gap-4 md:grid-cols-3">
              <article className="rounded-md border border-border p-4">
                <h3 className="font-display text-lg">Underfitting</h3>
                <p className="mt-2 text-sm leading-relaxed">
                  The model is too simple to capture the underlying pattern. On curved data, degree
                  1 leaves large residuals everywhere: high training error and high test error, with
                  low complexity.
                </p>
                <Button className="mt-3" size="sm" variant="outline" onClick={() => changeDegree(1)}>
                  Show degree 1
                </Button>
              </article>
              <article className="rounded-md border border-border p-4">
                <h3 className="font-display text-lg">A good fit</h3>
                <p className="mt-2 text-sm leading-relaxed">
                  The model is flexible enough to capture the main pattern without following every
                  individual observation. Both errors are lower, complexity is moderate. There is no
                  universally correct degree — it depends on the data and on generalisation.
                </p>
                <Button
                  className="mt-3"
                  size="sm"
                  variant="outline"
                  onClick={() => changeDegree(preset.suggestedDegree)}
                >
                  Try degree {preset.suggestedDegree}
                </Button>
              </article>
              <article className="rounded-md border border-border p-4">
                <h3 className="font-display text-lg">
                  What happens when the model becomes too complex?
                </h3>
                <p className="mt-2 text-sm leading-relaxed">
                  A highly flexible model can begin fitting noise rather than the underlying
                  relationship. Watch the training curve fall while the test curve turns upward —
                  both are measured, not assumed.
                </p>
                <Button className="mt-3" size="sm" variant="outline" onClick={() => changeDegree(10)}>
                  Show degree 10
                </Button>
              </article>
            </div>
            <pre className="mt-4 overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Model complexity ↑
        ↓
Training error ↓
        ↓
Test error may ↑`}</pre>
            <pre className="mt-4 overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Degree 1        →  very simple model
Degree 2–3      →  more flexibility
Degree 5+       →  highly flexible
Very high degree →  potentially unstable / overfit`}</pre>
            <p className="mt-3 text-sm text-muted-foreground">
              Complexity is not good or bad in itself. Judge it by how well the model does on data
              it was not fitted on.
            </p>
          </Step>

          {/* 07 — experiment */}
          <Step n={7} title="Find a model that generalises">
            <p className="text-sm leading-relaxed">
              Load the <strong>Overfitting experiment</strong> dataset, turn on the train/test
              split, then increase the polynomial degree one step at a time. Watch both errors in
              the metrics above and on the complexity chart. What happens to training error? What
              happens to test error?
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={() => applyPreset("overfit")}>
                Load the experiment
              </Button>
              <Button size="sm" variant="outline" onClick={() => setShowSplit(true)}>
                Turn on the split
              </Button>
            </div>
            <Accordion type="single" collapsible className="mt-4">
              <AccordionItem value="answer">
                <AccordionTrigger>What did you find?</AccordionTrigger>
                <AccordionContent>
                  Training error almost always keeps falling: more features give the curve more ways
                  to pass close to the training points. Test error usually falls at first — the
                  model is learning real structure — and then starts rising once the extra
                  flexibility is being spent on noise. The degree where test error is lowest is the
                  one that generalised best <em>on this sample</em>; another sample or another split
                  can point to a different degree.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Step>

          {/* 08 — scaling warning */}
          <Step n={8} title="Why can high-degree polynomial features be difficult?">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-3 text-sm leading-relaxed">
                <p>
                  Polynomial features grow fast. For a single input of x = 10 the columns the model
                  receives look like this:
                </p>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs">{`x  = 10
x² = 100
x³ = 1,000
x⁴ = 10,000
x⁵ = 100,000`}</pre>
                <p>
                  Columns on wildly different scales make numerical optimisation harder: a learning
                  rate that suits x is far too large for x⁵, so gradient descent can oscillate or
                  blow up. This is one reason feature scaling matters once you use polynomial
                  features with gradient descent.
                </p>
              </div>
              <div className="rounded-md border border-border">
                <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
                  Linear vs polynomial regression
                </h3>
                <table className="w-full text-sm">
                  <tbody>
                    {[
                      ["Features", "x", "x, x², x³, …"],
                      ["Shape", "Straight line", "Can form curves"],
                      ["Parameters", "β₀, β₁", "β₀, β₁, β₂, …"],
                      ["Flexibility", "Lower", "Higher"],
                      ["Main concern", "Underfitting", "Overfitting"],
                    ].map(([k, a, b]) => (
                      <tr key={k} className="border-b border-border/60 last:border-0">
                        <th scope="row" className="px-4 py-2 text-left font-medium">
                          {k}
                        </th>
                        <td className="px-4 py-2">{a}</td>
                        <td className="px-4 py-2 font-mono">{b}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
                  Polynomial regression is created by expanding the features and then fitting a
                  linear model to those features.
                </p>
              </div>
            </div>
          </Step>

          {/* 09 — behind the scenes */}
          <Step n={9} title="What's happening behind the scenes?">
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              <AccordionItem value="ls">
                <AccordionTrigger>Least squares fitting</AccordionTrigger>
                <AccordionContent>
                  <pre className="overflow-x-auto text-xs leading-relaxed">{`1. Start with x
        ↓
2. Generate polynomial features
        ↓
3. Build feature matrix
        ↓
4. Fit linear model
        ↓
5. Calculate coefficients
        ↓
6. Generate predictions
        ↓
7. Calculate residuals
        ↓
8. Calculate MSE`}</pre>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="gd">
                <AccordionTrigger>How is the polynomial model learning? (gradient descent)</AccordionTrigger>
                <AccordionContent>
                  <pre className="overflow-x-auto text-xs leading-relaxed">{`1. Initialize coefficients
        ↓
2. Predict
        ↓
3. Calculate residuals
        ↓
4. Calculate MSE
        ↓
5. Calculate gradients
        ↓
6. Update coefficients
        ↓
7. Repeat`}</pre>
                  <p className="mt-3 font-mono text-sm">∂MSE/∂βⱼ = −(2/n) Σ xᵢʲ (yᵢ − ŷᵢ)</p>
                  <p className="mt-1 font-mono text-sm">βⱼ(new) = βⱼ(old) − α × gradient</p>
                  {method === "gd" && (
                    <ul className="mt-3 space-y-1 font-mono text-sm">
                      {gradients.map((g, j) => (
                        <li key={j} className="flex justify-between">
                          <span className="text-muted-foreground">∂MSE/∂{betaLabel(j)}</span>
                          <span>{Math.abs(g) >= 1e6 ? g.toExponential(2) : fmt(g, 4)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {method === "ls" && (
                    <p className="mt-3 text-sm text-muted-foreground">
                      Switch the fitting method to gradient descent to run this loop step by step.
                    </p>
                  )}
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="code">
                <AccordionTrigger>From mathematics to code</AccordionTrigger>
                <AccordionContent>
                  <p className="text-sm">Feature expansion — [x, x², x³]:</p>
                  <pre className="mt-2 overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">{`const features = [
    x,
    x ** 2,
    x ** 3
];`}</pre>
                  <p className="mt-4 text-sm">Prediction — ŷ = β₀ + β₁x + β₂x² + β₃x³:</p>
                  <pre className="mt-2 overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">{`let prediction = intercept;

for (let degree = 1; degree <= maxDegree; degree++) {
    prediction += coefficients[degree] * x ** degree;
}`}</pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Step>

          {/* 10 — checkpoint */}
          <Step n={10} title="Can you explain Polynomial Regression?">
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              {CHECKPOINTS.map((c, i) => (
                <AccordionItem key={c.q} value={`q${i}`}>
                  <AccordionTrigger>{`${i + 1}. ${c.q}`}</AccordionTrigger>
                  <AccordionContent>{c.a}</AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </Step>

          {/* 11 — summary */}
          <Step n={11} title="Polynomial Regression in one picture">
            <div className="rounded-md border border-border p-6">
              <pre className="overflow-x-auto text-center text-xs leading-relaxed">{`              INPUT x
                 ↓
        Polynomial Expansion
                 ↓
       x, x², x³, x⁴, ...
                 ↓
          Linear Model
                 ↓
       Learned Coefficients
                 ↓
            Prediction
                 ↓
             Residual
                 ↓
               MSE
                 ↓
        Evaluate Generalization
                 ↓
       Training vs Test Error
                 ↓
     Choose appropriate complexity`}</pre>
              <div className="mt-6">
                <FlowDiagram
                  steps={[
                    "Data",
                    "Features",
                    "Linear model",
                    "Coefficients",
                    "Predictions",
                    "Residuals",
                    "MSE",
                    "Generalisation",
                  ]}
                  active
                />
              </div>
            </div>
          </Step>
        </main>

        <footer className="border-t border-border">
          <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-muted-foreground">
            Every coefficient, prediction, curve and error on this page is computed in your browser
            from the dataset shown — no fitted values are hard-coded.
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}
