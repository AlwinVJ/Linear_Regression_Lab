import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Dices, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { polyEquation, polyMSE } from "@/algorithms/polynomialRegression";
import {
  coefficientEnergy,
  fitRidge,
  l2Penalty,
  predictFit,
  ridgeDiverging,
  ridgeGradientDescentStep,
  ridgeGradients,
  INSTABILITY_MESSAGE,
} from "@/algorithms/ridgeRegression";
import { CoefficientChart } from "@/components/CoefficientChart";
import { CoefficientPathChart, type PathRow } from "@/components/CoefficientPathChart";
import { DegreeControl } from "@/components/DegreeControl";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { FlowDiagram } from "@/components/FlowDiagram";
import { PolynomialChart } from "@/components/PolynomialChart";
import { RegularizationControl } from "@/components/RegularizationControl";
import { RegularizationPenalty } from "@/components/RegularizationPenalty";
import { RidgeErrorChart, type ErrorRow } from "@/components/RidgeErrorChart";
import { RidgeLossCurve, type CurvePoint } from "@/components/RidgeLossCurve";
import { RidgeMetrics } from "@/components/RidgeMetrics";
import { SiteNav } from "@/components/SiteNav";
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
import { LAMBDA_PATH, lambdaLabel } from "@/utils/regularization";
import { computeScaling, fitRow, toRawCoefficients } from "@/utils/scaling";

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

const MYTHS: { q: string; a: string }[] = [
  {
    q: "“Ridge removes features.”",
    a: "Ridge usually shrinks coefficients toward zero rather than making them exactly zero. Every feature is still in the model, just with a smaller weight.",
  },
  {
    q: "“Higher λ always improves the model.”",
    a: "Excessive regularization can cause underfitting: the coefficients become so small that the model can no longer represent the pattern.",
  },
  {
    q: "“Ridge minimizes MSE.”",
    a: "Ridge minimizes a regularized objective containing both MSE and an L2 penalty. Its MSE can be slightly higher than the OLS one while its objective is lower.",
  },
  {
    q: "“Regularization means reducing the number of features.”",
    a: "Regularization controls model complexity by penalizing parameter values. Feature selection is a separate concept.",
  },
  {
    q: "“Ridge and ordinary least squares are completely different models.”",
    a: "Ridge uses the same linear/polynomial model form but changes the optimization objective by adding an L2 penalty.",
  },
];

const CHECKPOINTS: { q: string; a: string }[] = [
  {
    q: "What problem is regularization trying to solve?",
    a: "A flexible model can fit the noise in the training sample. Regularization discourages that by making unnecessarily large coefficients costly.",
  },
  {
    q: "What does λ control?",
    a: "How much the penalty matters relative to prediction error. λ = 0 means no penalty at all; a very large λ means the penalty dominates.",
  },
  {
    q: "Why does Ridge shrink coefficients?",
    a: "Because every non-intercept coefficient adds λβ² to the objective. Keeping a large coefficient has to be paid for by a big enough reduction in MSE.",
  },
  {
    q: "Why are coefficients squared in Ridge?",
    a: "Squaring makes the penalty grow quickly with size — 2 costs 4 while 10 costs 100 — and it is smooth, so gradients are well behaved.",
  },
  {
    q: "Does Ridge normally make coefficients exactly zero?",
    a: "No. It pushes them toward zero, but they typically stay small and non-zero. Exact zeros are what Lasso tends to produce.",
  },
  {
    q: "Why isn't the intercept normally penalized?",
    a: "The intercept only shifts the whole model up or down. Penalizing it would drag predictions toward zero rather than toward simplicity.",
  },
  {
    q: "What happens when λ = 0?",
    a: "The penalty term disappears and Ridge becomes ordinary least squares — exactly the fit from the earlier pages.",
  },
  {
    q: "What happens when λ becomes very large?",
    a: "All penalized coefficients are squeezed toward zero, the curve flattens toward the intercept, and the model underfits.",
  },
  {
    q: "Why can Ridge help with overfitting?",
    a: "Overfitted polynomial models usually rely on large, opposing coefficients. Constraining their size removes the wild bends that chase noise.",
  },
  {
    q: "Why does feature scaling matter?",
    a: "The penalty is applied to coefficient magnitudes. If features live on very different scales, the same penalty hits them very unevenly.",
  },
  {
    q: "What is the difference between MSE and the Ridge objective?",
    a: "MSE asks how wrong the predictions are. The Ridge objective is MSE plus λ Σβⱼ² — prediction error plus the cost of the coefficients.",
  },
];

export default function RidgeRegressionLab() {
  const [presetId, setPresetId] = useState("noisy-curve");
  const preset = POLY_PRESETS.find((p) => p.id === presetId) ?? POLY_PRESETS[0]!;
  const [seed, setSeed] = useState(DEFAULT_POLY_SEED);
  const [noise, setNoise] = useState(preset.noise);
  const [degree, setDegree] = useState(7);
  const [lambda, setLambda] = useState(0);
  const [standardize, setStandardize] = useState(true);
  const [showSplit, setShowSplit] = useState(true);
  const [showResiduals, setShowResiduals] = useState(false);
  const [method, setMethod] = useState<"closed" | "gd">("closed");
  const [selectedCoef, setSelectedCoef] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState(
    "Start at λ = 0 — that is ordinary least squares. Then raise λ and watch the coefficients shrink.",
  );

  const points = useMemo(() => preset.build(seed, noise), [preset, seed, noise]);
  const split = useMemo(() => trainTestSplit(points, TEST_RATIO, seed), [points, seed]);
  const train = showSplit ? split.train : points;
  const test = showSplit ? split.test : [];

  const scaling = useMemo(
    () => (standardize && degree > 0 ? computeScaling(train, degree) : null),
    [standardize, train, degree],
  );

  const ridgeFit = useMemo(
    () => fitRidge(train, degree, lambda, standardize),
    [train, degree, lambda, standardize],
  );
  const olsFit = useMemo(() => fitRidge(train, degree, 0, standardize), [train, degree, standardize]);

  /* ---------------- Gradient descent on the Ridge objective ---------------- */
  const [alpha, setAlpha] = useState(0.05);
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

  useEffect(() => {
    resetGd();
  }, [resetGd, train, standardize]);

  const gdStep = useCallback(() => {
    setGdCoefs((current) => {
      const result = ridgeGradientDescentStep(train, current, degree, scaling, lambda, alpha);
      const raw = toRawCoefficients(result.after, scaling);
      const mse = polyMSE(train, raw);
      if (ridgeDiverging(result.after, mse)) {
        setGdDiverged(true);
        setPlaying(false);
        setMessage(INSTABILITY_MESSAGE);
        return current;
      }
      setGdIteration((i) => i + 1);
      return result.after;
    });
  }, [train, degree, scaling, lambda, alpha]);

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

  /* ---------------- Active model ---------------- */
  const fitCoefficients = method === "closed" ? ridgeFit.fitCoefficients : gdCoefs;
  const coefficients = useMemo(
    () => (method === "closed" ? ridgeFit.coefficients : toRawCoefficients(gdCoefs, scaling)),
    [method, ridgeFit, gdCoefs, scaling],
  );
  const unstable = method === "closed" ? !ridgeFit.ok : gdDiverged;
  const instabilityReason =
    method === "closed" ? ridgeFit.reason : gdDiverged ? INSTABILITY_MESSAGE : undefined;

  const trainMse = useMemo(() => polyMSE(train, coefficients), [train, coefficients]);
  const testMse = useMemo(
    () => (showSplit && test.length > 0 ? polyMSE(test, coefficients) : null),
    [showSplit, test, coefficients],
  );
  const penalty = l2Penalty(fitCoefficients, lambda);
  const energy = coefficientEnergy(fitCoefficients);
  const objective = trainMse + penalty;

  const olsTrainMse = useMemo(() => polyMSE(train, olsFit.coefficients), [train, olsFit]);
  const olsTestMse = useMemo(
    () => (showSplit && test.length > 0 ? polyMSE(test, olsFit.coefficients) : null),
    [showSplit, test, olsFit],
  );

  const gradients = useMemo(
    () => ridgeGradients(train, fitCoefficients, degree, scaling, lambda),
    [train, fitCoefficients, degree, scaling, lambda],
  );

  /* ---------------- Sweeps over λ ---------------- */
  const sweep = useMemo(
    () =>
      LAMBDA_PATH.map((l) => {
        const fit = fitRidge(train, degree, l, standardize);
        return {
          lambda: l,
          fit,
          trainMse: fit.ok ? polyMSE(train, fit.coefficients) : Number.NaN,
          testMse: fit.ok && showSplit && test.length > 0 ? polyMSE(test, fit.coefficients) : null,
        };
      }),
    [train, test, degree, standardize, showSplit],
  );

  const pathRows: PathRow[] = sweep.map((s) => ({
    lambda: s.lambda,
    coefficients: s.fit.fitCoefficients,
    ok: s.fit.ok,
  }));
  const errorRows: ErrorRow[] = sweep.map((s) => ({
    lambda: s.lambda,
    trainMse: s.trainMse,
    testMse: s.testMse,
    ok: s.fit.ok,
  }));

  /* ---------------- One-parameter loss intuition ---------------- */
  const curveIndex = Math.min(1, Math.max(degree, 1));
  const lossCurve: CurvePoint[] = useMemo(() => {
    if (degree < 1 || !olsFit.ok || train.length === 0) return [];
    const base = olsFit.fitCoefficients.slice();
    const center = base[curveIndex] ?? 0;
    const span = Math.max(Math.abs(center) * 2, 1);
    const out: CurvePoint[] = [];
    for (let i = 0; i <= 80; i++) {
      const beta = center - span + (2 * span * i) / 80;
      const trial = base.slice();
      trial[curveIndex] = beta;
      let sum = 0;
      for (const p of train) sum += (p.y - predictFit(trial, p.x, degree, scaling)) ** 2;
      const mse = sum / train.length;
      out.push({ beta, mse, objective: mse + lambda * beta ** 2 });
    }
    return out;
  }, [olsFit, train, degree, scaling, lambda, curveIndex]);

  const olsBeta = olsFit.fitCoefficients[curveIndex] ?? 0;
  const ridgeBeta = ridgeFit.fitCoefficients[curveIndex] ?? 0;

  /* ---------------- Three-state comparison ---------------- */
  const threeStates = useMemo(
    () =>
      [0, 1, 100].map((l) => {
        const fit = fitRidge(train, degree, l, standardize);
        return {
          lambda: l,
          fit,
          trainMse: fit.ok ? polyMSE(train, fit.coefficients) : Number.NaN,
          testMse: fit.ok && showSplit && test.length > 0 ? polyMSE(test, fit.coefficients) : null,
          energy: coefficientEnergy(fit.fitCoefficients),
        };
      }),
    [train, test, degree, standardize, showSplit],
  );

  const applyPreset = (id: string) => {
    const p = POLY_PRESETS.find((x) => x.id === id);
    if (!p) return;
    setPresetId(id);
    setNoise(p.noise);
    setMessage(`${p.caption} ${p.expectation}`);
  };

  const changeLambda = (l: number) => {
    setLambda(l);
    setMessage(
      l === 0
        ? "λ = 0: no penalty. This is ordinary least squares, exactly as on the earlier pages."
        : `λ = ${lambdaLabel(l)}: every coefficient except β₀ now costs λβ². The model refits and every number below is recalculated.`,
    );
  };

  const selectedFeature = selectedCoef !== null ? selectedCoef : null;

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Ridge Regression</h1>
              <p className="mt-1 text-muted-foreground">Why do we need regularization?</p>
            </div>
            <SiteNav />
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-14 px-5 py-10">
          {/* 01 — the problem */}
          <Step n={1} title="Why do we need regularization?">
            <div className="grid gap-8 lg:grid-cols-[1fr_340px]">
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
                <p className="mt-3 font-mono text-sm">{unstable ? "—" : polyEquation(coefficients)}</p>
              </div>
              <div className="space-y-4">
                <p className="text-sm leading-relaxed">
                  Raise the polynomial degree on this noisy data and the curve starts chasing
                  individual points. Nothing in the objective discourages that: ordinary least
                  squares only cares about prediction error.
                </p>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Without regularization
        ↓
Model only cares about prediction error

With Ridge regularization
        ↓
Model cares about prediction error
        +
large coefficients`}</pre>
                <div className="rounded-md border border-border p-4">
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">Dataset</div>
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
                    <span className="font-mono text-xs text-muted-foreground">Random seed: {seed}</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSeed(Math.floor(Math.random() * 900000) + 100000)}
                    >
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
                    <Switch checked={showSplit} onCheckedChange={setShowSplit} />
                    Show train/test split
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <Switch checked={standardize} onCheckedChange={setStandardize} />
                    Standardize features
                  </label>
                </div>
                <ExplanationPanel message={message} />
              </div>
            </div>
          </Step>

          {/* 02 — the objective */}
          <Step n={2} title="The Ridge objective">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-md border border-axis bg-accent/10 p-5">
                <p className="font-mono text-xl">Loss = MSE + λ Σβⱼ²</p>
                <dl className="mt-4 space-y-2 text-sm">
                  <div>
                    <dt className="font-mono">MSE</dt>
                    <dd className="text-muted-foreground">Measures prediction error.</dd>
                  </div>
                  <div>
                    <dt className="font-mono">λ</dt>
                    <dd className="text-muted-foreground">Controls regularization strength.</dd>
                  </div>
                  <div>
                    <dt className="font-mono">βⱼ²</dt>
                    <dd className="text-muted-foreground">Penalizes large coefficients.</dd>
                  </div>
                </dl>
                <p className="mt-4 text-sm leading-relaxed">
                  Ridge Regression asks the model to balance two goals: fit the data well and keep
                  the coefficients reasonably small.
                </p>
              </div>
              <div className="space-y-4">
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Ordinary Regression
        ↓
Minimize prediction error

Ridge Regression
        ↓
Minimize prediction error
        +
Penalize large coefficients`}</pre>
                <div className="rounded-md border border-border p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    The intercept rule
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed">
                    For ŷ = β₀ + β₁x + β₂x² + … the penalty applies to β₁, β₂, β₃ … but{" "}
                    <strong>not to β₀</strong>. The intercept only moves the model up or down; there
                    is no reason to punish it for being large. This implementation excludes it
                    everywhere: in the closed-form solution, in the penalty, and in the gradients.
                  </p>
                </div>
              </div>
            </div>
          </Step>

          {/* 03 — controls */}
          <Step n={3} title="Turn the regularization up">
            <div className="grid gap-6 lg:grid-cols-2">
              <RegularizationControl lambda={lambda} onChange={changeLambda} />
              <div className="space-y-4">
                <DegreeControl degree={degree} onChange={setDegree} max={MAX_DEGREE} />
                <div className="rounded-md border border-border p-4">
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">
                    Fitting method
                  </div>
                  <div className="mt-3 flex gap-2">
                    <Button
                      size="sm"
                      variant={method === "closed" ? "default" : "outline"}
                      onClick={() => setMethod("closed")}
                    >
                      Closed-form Ridge
                    </Button>
                    <Button
                      size="sm"
                      variant={method === "gd" ? "default" : "outline"}
                      onClick={() => setMethod("gd")}
                    >
                      Gradient descent
                    </Button>
                  </div>
                  {method === "closed" ? (
                    <p className="mt-3 text-sm text-muted-foreground">
                      The coefficients are solved directly, using a QR decomposition of the feature
                      matrix extended with the penalty rows — the stable way to get the same answer
                      as (XᵀX + λI)⁻¹Xᵀy without inverting anything.
                    </p>
                  ) : (
                    <div className="mt-3 space-y-3">
                      <p className="text-sm text-muted-foreground">
                        The same gradient descent loop as before, but each gradient now carries the
                        extra term 2λβⱼ (never for β₀).
                      </p>
                      <label className="block text-xs uppercase tracking-widest text-muted-foreground">
                        Learning rate α: {alpha}
                      </label>
                      <div className="flex flex-wrap gap-2">
                        {[0.001, 0.01, 0.05, 0.1].map((a) => (
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
                      {!standardize && (
                        <p className="text-sm text-muted-foreground">
                          Gradient descent on raw polynomial features is very sensitive. Turn on
                          feature standardization, or use a much smaller learning rate.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {unstable && (
              <p className="mt-4 rounded-md border border-resid-neg bg-resid-neg/10 px-4 py-3 text-sm">
                {instabilityReason ?? INSTABILITY_MESSAGE}
              </p>
            )}

            <div className="mt-6">
              <RidgeMetrics
                lambda={lambda}
                trainMse={trainMse}
                testMse={testMse}
                objective={objective}
                energy={energy}
                showSplit={showSplit}
              />
            </div>
          </Step>

          {/* 04 — coefficient shrinkage */}
          <Step n={4} title="Watch the coefficients shrink">
            <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
              <CoefficientChart
                ridge={coefficients}
                ols={olsFit.coefficients}
                selected={selectedCoef}
                onSelect={setSelectedCoef}
              />
              <div className="space-y-4">
                <div className="rounded-md border border-border">
                  <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
                    Inspect a coefficient
                  </h3>
                  {selectedFeature !== null ? (
                    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 px-4 py-3 font-mono text-sm">
                      <dt className="text-muted-foreground">Feature</dt>
                      <dd className="text-right">
                        {selectedFeature === 0 ? "constant 1" : powerLabel(selectedFeature)}
                      </dd>
                      <dt className="text-muted-foreground">Coefficient</dt>
                      <dd className="text-right">{fmt(coefficients[selectedFeature] ?? 0, 4)}</dd>
                      <dt className="text-muted-foreground">Contribution</dt>
                      <dd className="text-right">
                        {betaLabel(selectedFeature)}
                        {selectedFeature === 0 ? "" : powerLabel(selectedFeature)}
                      </dd>
                      <dt className="text-muted-foreground">Penalty λβ²</dt>
                      <dd className="text-right">
                        {selectedFeature === 0
                          ? "0 (not penalised)"
                          : fmt(lambda * (fitCoefficients[selectedFeature] ?? 0) ** 2, 4)}
                      </dd>
                    </dl>
                  ) : (
                    <p className="px-4 py-3 text-sm text-muted-foreground">
                      Click a coefficient to see its feature, its contribution to the prediction and
                      the penalty it pays.
                    </p>
                  )}
                </div>
                <RegularizationPenalty
                  mse={trainMse}
                  penalty={penalty}
                  objective={objective}
                  lambda={lambda}
                  energy={energy}
                />
                <div className="rounded-md border border-axis bg-accent/10 p-4">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                    Ridge does NOT simply delete features
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed">
                    Ridge shrinks coefficients toward zero, but typically does not make them exactly
                    zero. Compare the two columns on the left: the Ridge values are smaller, not
                    missing. The exact values depend on the dataset, the degree and λ.
                  </p>
                </div>
              </div>
            </div>
          </Step>

          {/* 05 — why squares */}
          <Step n={5} title="Why does Ridge shrink coefficients?">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-3 text-sm leading-relaxed">
                <p>Ridge adds a cost to large coefficients, and that cost is the square:</p>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs">{`Coefficient = 2   →  penalty 2²  = 4
Coefficient = 10  →  penalty 10² = 100`}</pre>
                <p>
                  Five times the coefficient costs twenty-five times as much. A large coefficient
                  therefore has to earn its place by reducing the MSE by more than it adds to the
                  penalty — otherwise the model is better off shrinking it.
                </p>
                <p className="text-muted-foreground">
                  A Ridge model can have a slightly higher MSE than OLS while still having a lower
                  overall Ridge objective. MSE alone is not what Ridge is minimising.
                </p>
              </div>
              <div className="rounded-md border border-border p-4">
                <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                  One coefficient at a time
                </h3>
                {lossCurve.length > 0 ? (
                  <div className="mt-3">
                    <RidgeLossCurve
                      curve={lossCurve}
                      index={curveIndex}
                      olsBeta={olsBeta}
                      ridgeBeta={ridgeBeta}
                    />
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-muted-foreground">
                    Increase the degree to at least 1 to see this curve.
                  </p>
                )}
                <p className="mt-2 text-sm text-muted-foreground">
                  Holding the other coefficients fixed, the MSE curve has its minimum where the data
                  wants it. Adding λβ² tilts the curve upward away from zero, so the combined
                  minimum sits closer to zero. A real loss surface has one dimension per
                  coefficient — this is a one-parameter slice for intuition.
                </p>
              </div>
            </div>
          </Step>

          {/* 06 — OLS vs Ridge */}
          <Step n={6} title="Ordinary least squares vs Ridge">
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full min-w-[520px] text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/40 text-xs uppercase tracking-widest text-muted-foreground">
                    <th className="px-3 py-2 text-left" />
                    <th className="px-3 py-2 text-right">OLS (λ = 0)</th>
                    <th className="px-3 py-2 text-right">Ridge (λ = {lambdaLabel(lambda)})</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  <tr className="border-b border-border/60">
                    <td className="px-3 py-1.5">Training MSE</td>
                    <td className="px-3 py-1.5 text-right">{fmt(olsTrainMse, 4)}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(trainMse, 4)}</td>
                  </tr>
                  <tr className="border-b border-border/60">
                    <td className="px-3 py-1.5">Test MSE</td>
                    <td className="px-3 py-1.5 text-right">
                      {olsTestMse === null ? "—" : fmt(olsTestMse, 4)}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      {testMse === null ? "—" : fmt(testMse, 4)}
                    </td>
                  </tr>
                  <tr className="border-b border-border/60">
                    <td className="px-3 py-1.5">Σβⱼ² (no β₀)</td>
                    <td className="px-3 py-1.5 text-right">
                      {fmt(coefficientEnergy(olsFit.fitCoefficients), 4)}
                    </td>
                    <td className="px-3 py-1.5 text-right">{fmt(energy, 4)}</td>
                  </tr>
                  <tr className="border-b border-border/60">
                    <td className="px-3 py-1.5">Regularization penalty</td>
                    <td className="px-3 py-1.5 text-right">0</td>
                    <td className="px-3 py-1.5 text-right">{fmt(penalty, 4)}</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-1.5">Objective</td>
                    <td className="px-3 py-1.5 text-right">{fmt(olsTrainMse, 4)}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(objective, 4)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              The OLS column never changes while you move λ — it is the same model as on the earlier
              pages. Only the Ridge column responds.
            </p>

            <h3 className="mt-6 font-display text-lg">Three states side by side</h3>
            <div className="mt-3 grid gap-4 md:grid-cols-3">
              {threeStates.map((s) => (
                <article key={s.lambda} className="rounded-md border border-border p-4">
                  <div className="flex items-baseline justify-between">
                    <h4 className="font-display text-base">
                      {s.lambda === 0
                        ? "No regularization"
                        : s.lambda === 1
                          ? "Moderate"
                          : "Strong"}
                    </h4>
                    <span className="font-mono text-sm">λ = {lambdaLabel(s.lambda)}</span>
                  </div>
                  <dl className="mt-3 space-y-1 font-mono text-sm">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Train MSE</dt>
                      <dd>{s.fit.ok ? fmt(s.trainMse, 4) : "unstable"}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Test MSE</dt>
                      <dd>{s.testMse === null ? "—" : fmt(s.testMse, 4)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Σβⱼ²</dt>
                      <dd>{fmt(s.energy, 3)}</dd>
                    </div>
                  </dl>
                  <Button
                    className="mt-3"
                    size="sm"
                    variant="outline"
                    onClick={() => changeLambda(s.lambda)}
                  >
                    Use this λ
                  </Button>
                </article>
              ))}
            </div>
          </Step>

          {/* 07 — coefficient paths */}
          <Step n={7} title="Coefficients vs regularization strength">
            <div className="rounded-md border border-border p-4">
              <CoefficientPathChart rows={pathRows} degree={degree} selectedLambda={lambda} />
            </div>
            <pre className="mt-4 overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`λ = 0
    ↓
Large coefficients

λ increases
    ↓
Coefficients shrink

λ becomes very large
    ↓
Coefficients approach zero`}</pre>
            <p className="mt-3 text-sm text-muted-foreground">
              Very strong regularization can shrink the coefficients so much that the model becomes
              too simple: less flexible, and potentially underfitting — the same idea as a
              too-low polynomial degree on the previous page.
            </p>
          </Step>

          {/* 08 — error vs lambda */}
          <Step n={8} title="Error vs regularization strength">
            <div className="rounded-md border border-border p-4">
              <RidgeErrorChart
                rows={errorRows}
                selectedLambda={lambda}
                showTest={showSplit && test.length > 0}
                onSelectLambda={changeLambda}
              />
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              Weak regularization allows a more complex model; increasing it reduces flexibility;
              too much of it underfits. Which λ generalises best depends on the data — not every
              dataset produces a neat U-shaped test curve.
            </p>
          </Step>

          {/* 09 — degree × lambda experiment */}
          <Step n={9} title="Experiment: degree and λ are separate dials">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <DegreeControl degree={degree} onChange={setDegree} max={MAX_DEGREE} />
                <RegularizationControl lambda={lambda} onChange={changeLambda} />
              </div>
              <div className="space-y-3 text-sm leading-relaxed">
                <p>
                  Degree controls how many features exist. λ controls how large their coefficients
                  are allowed to be. Try each combination and watch the curve, the errors and the
                  coefficient bars:
                </p>
                <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                  <li>Low degree + low λ — a simple model, barely constrained.</li>
                  <li>High degree + low λ — a potentially complex model chasing noise.</li>
                  <li>High degree + moderate λ — a rich feature space with constrained coefficients.</li>
                  <li>High degree + high λ — flexible features, but coefficients squeezed flat.</li>
                </ul>
                <p>
                  Ridge does not remove the polynomial features. It controls the magnitude of their
                  coefficients.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => { setDegree(1); changeLambda(0); }}>
                    Low degree + low λ
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => { setDegree(9); changeLambda(0); }}>
                    High degree + low λ
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => { setDegree(9); changeLambda(1); }}>
                    High degree + moderate λ
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => { setDegree(9); changeLambda(1000); }}>
                    High degree + high λ
                  </Button>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <RidgeMetrics
                lambda={lambda}
                trainMse={trainMse}
                testMse={testMse}
                objective={objective}
                energy={energy}
                showSplit={showSplit}
              />
            </div>
          </Step>

          {/* 10 — scaling */}
          <Step n={10} title="Why does scaling matter?">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-3 text-sm leading-relaxed">
                <p>
                  Ridge penalises coefficient <em>magnitude</em>. If one feature has values around 1
                  and another around 10,000, their coefficients live on completely different scales,
                  so the same penalty hits them very unevenly. Polynomial features are exactly this
                  situation: x, x², x³ … grow fast.
                </p>
                <p>
                  Features are therefore commonly standardised before applying Ridge. The toggle at
                  the top of this page does that, using the training observations only — test
                  statistics are never used to fit the scaler.
                </p>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Before scaling      x, x², x³, ...
        ↓
Standardization     zⱼ = (xʲ − mean) / std
        ↓
After scaling       z₁, z₂, z₃, ...`}</pre>
              </div>
              <div className="overflow-x-auto rounded-md border border-border">
                <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
                  {standardize ? "Standardised features (training statistics)" : "Raw features"}
                </h3>
                <table className="w-full min-w-[420px] text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs uppercase tracking-widest text-muted-foreground">
                      <th className="px-3 py-2 text-left">x</th>
                      {Array.from({ length: Math.min(degree, 4) }, (_, i) => (
                        <th key={i} className="px-3 py-2 text-right">
                          {standardize ? `z${i + 1}` : powerLabel(i + 1)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {train.slice(0, 6).map((p) => {
                      const row = fitRow(p.x, degree, scaling);
                      return (
                        <tr key={p.id} className="border-b border-border/60 last:border-0">
                          <td className="px-3 py-1.5">{fmt(p.x, 2)}</td>
                          {Array.from({ length: Math.min(degree, 4) }, (_, i) => (
                            <td key={i} className="px-3 py-1.5 text-right">
                              {fmt(row[i + 1] ?? 0, 3)}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
                  Standardisation puts features on comparable scales before regularization. The
                  fitted coefficients are converted back so the equation and the curve are still in
                  ordinary x terms.
                </p>
              </div>
            </div>
          </Step>

          {/* 11 — mathematics */}
          <Step n={11} title="Ridge Regression mathematics">
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              <AccordionItem value="s1">
                <AccordionTrigger>Step 1 — ordinary least squares</AccordionTrigger>
                <AccordionContent>
                  <p className="font-mono text-sm">MSE = (1/n) Σ(yᵢ − ŷᵢ)²</p>
                  <p className="mt-2 text-sm">
                    Choose the coefficients that make the squared prediction errors as small as
                    possible. Nothing else is taken into account.
                  </p>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="s2">
                <AccordionTrigger>Step 2 — add a penalty</AccordionTrigger>
                <AccordionContent>
                  <p className="font-mono text-sm">Objective = MSE + λ Σⱼ₌₁ βⱼ²</p>
                  <p className="mt-2 text-sm">
                    The sum starts at j = 1: β₀ is excluded. Larger coefficients now increase the
                    objective directly.
                  </p>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="s3">
                <AccordionTrigger>Step 3 — optimise the combined objective</AccordionTrigger>
                <AccordionContent>
                  <p className="text-sm">
                    Ridge chooses coefficients that balance fitting the data with keeping
                    coefficient magnitudes small. Either solve it directly, or walk downhill on it
                    with gradient descent.
                  </p>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="gd">
                <AccordionTrigger>How does Ridge learn? (gradient descent)</AccordionTrigger>
                <AccordionContent>
                  <pre className="overflow-x-auto text-xs leading-relaxed">{`Gradient =
Gradient from MSE
        +
Gradient from penalty`}</pre>
                  <p className="mt-3 font-mono text-sm">∂MSE/∂βⱼ = −(2/n) Σ fᵢⱼ (yᵢ − ŷᵢ)</p>
                  <p className="font-mono text-sm">∂penalty/∂βⱼ = 2λβⱼ   (0 for β₀)</p>
                  <p className="font-mono text-sm">βⱼ(new) = βⱼ(old) − α × gradient</p>
                  <div className="mt-4 overflow-x-auto rounded-md border border-border">
                    <table className="w-full min-w-[420px] text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted/40 text-xs uppercase tracking-widest text-muted-foreground">
                          <th className="px-3 py-2 text-left">Coefficient</th>
                          <th className="px-3 py-2 text-right">Prediction gradient</th>
                          <th className="px-3 py-2 text-right">Regularization gradient</th>
                          <th className="px-3 py-2 text-right">Ridge gradient</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono tabular-nums">
                        {gradients.map((g, j) => (
                          <tr key={j} className="border-b border-border/60 last:border-0">
                            <td className="px-3 py-1.5">{betaLabel(j)}</td>
                            <td className="px-3 py-1.5 text-right">{fmt(g.mse, 4)}</td>
                            <td className="px-3 py-1.5 text-right">{fmt(g.regularization, 4)}</td>
                            <td className="px-3 py-1.5 text-right">{fmt(g.total, 4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    In ordinary regression the gradient comes only from prediction error. In Ridge
                    it also contains a force pulling large coefficients toward zero — and that force
                    is exactly zero for the intercept.
                  </p>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="matrix">
                <AccordionTrigger>Advanced: matrix form</AccordionTrigger>
                <AccordionContent>
                  <p className="font-mono text-sm">β̂ridge = (XᵀX + λI)⁻¹Xᵀy</p>
                  <p className="mt-2 text-sm">
                    This is the closed-form Ridge solution, with the entry of I corresponding to the
                    intercept set to zero so β₀ is not penalised. This app never forms that inverse:
                    it solves the equivalent augmented least-squares problem with a QR
                    decomposition, which is better conditioned.
                  </p>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="code">
                <AccordionTrigger>From mathematics to code</AccordionTrigger>
                <AccordionContent>
                  <p className="text-sm">Ridge objective — MSE + λΣβⱼ²:</p>
                  <pre className="mt-2 overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">{`mse = calculate_mse(points, coefficients)

penalty = regularization_strength * sum(
    beta ** 2
    for beta in coefficients[1:]  # skip the intercept
)

ridge_loss = mse + penalty`}</pre>
                  <p className="mt-4 text-sm">The extra gradient term:</p>
                  <pre className="mt-2 overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">{`regularization_gradient = (
    0 if index == 0 else 2 * regularization_strength * beta
)`}</pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </Step>

          {/* 12 — myths */}
          <Step n={12} title="Ridge Regression myths">
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              {MYTHS.map((m, i) => (
                <AccordionItem key={m.q} value={`m${i}`}>
                  <AccordionTrigger>{`Myth ${i + 1} — ${m.q}`}</AccordionTrigger>
                  <AccordionContent>{m.a}</AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </Step>

          {/* 13 — checkpoint */}
          <Step n={13} title="Can you explain Ridge Regression?">
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              {CHECKPOINTS.map((c, i) => (
                <AccordionItem key={c.q} value={`q${i}`}>
                  <AccordionTrigger>{`${i + 1}. ${c.q}`}</AccordionTrigger>
                  <AccordionContent>{c.a}</AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </Step>

          {/* 14 — summary */}
          <Step n={14} title="Ridge Regression in one picture">
            <div className="rounded-md border border-border p-6">
              <pre className="overflow-x-auto text-center text-xs leading-relaxed">{`                 DATA
                   ↓
              Polynomial
               Features
                   ↓
                 Model
                   ↓
             Predictions
                   ↓
            Prediction Error
                   ↓
                  MSE
                   +
          L2 Regularization
                   ↓
       Penalize Large Coefficients
                   ↓
            Ridge Objective
                   ↓
          Smaller Coefficients
                   ↓
       Simpler / More Stable Model`}</pre>
              <div className="mt-6">
                <FlowDiagram
                  steps={[
                    "Data",
                    "Features",
                    "Model",
                    "Predictions",
                    "MSE",
                    "L2 penalty",
                    "Ridge objective",
                    "Smaller coefficients",
                  ]}
                  active
                />
              </div>
            </div>
          </Step>
        </main>

        <footer className="border-t border-border">
          <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-muted-foreground">
            Every coefficient, penalty, objective and curve on this page is computed in your browser
            from the dataset shown — no fitted values are hard-coded.
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}
