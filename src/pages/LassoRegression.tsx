import { useCallback, useEffect, useMemo, useState } from "react";
import { Dices } from "lucide-react";
import {
  buildFeatures,
  featureEquation,
  mseWith,
  predictWith,
  type ColumnScaling,
  type FeatureSpec,
} from "@/algorithms/featureRegression";
import { fitLasso, fitRidgeFeatures, l2PenaltyOf } from "@/algorithms/lassoRegression";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { FeatureCurveChart } from "@/components/FeatureCurveChart";
import { FeatureSelectionPanel } from "@/components/FeatureSelectionPanel";
import { LassoCoefficientChart } from "@/components/LassoCoefficientChart";
import { LassoPathChart, type LassoPathRow } from "@/components/LassoPathChart";
import { ObjectiveHistoryChart } from "@/components/ObjectiveHistoryChart";
import { OptimizationPanel } from "@/components/OptimizationPanel";
import { PenaltyShapeChart } from "@/components/PenaltyShapeChart";
import { RegularizationControl } from "@/components/RegularizationControl";
import { RidgeErrorChart, type ErrorRow } from "@/components/RidgeErrorChart";
import { SiteNav } from "@/components/SiteNav";
import { SoftThresholdVisualization } from "@/components/SoftThresholdVisualization";
import { SparsityChart } from "@/components/SparsityChart";
import { FlowDiagram } from "@/components/FlowDiagram";
import { DegreeControl } from "@/components/DegreeControl";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  activeCount,
  coordinateUpdate,
  l1Penalty,
  lassoObjective,
  updateIntercept,
  type CoordinateUpdate,
  type HistoryEntry,
} from "@/optimization/coordinateDescent";
import { fmt } from "@/utils/calculations";
import { DEFAULT_POLY_SEED, POLY_PRESETS } from "@/utils/dataset";
import { trainTestSplit } from "@/utils/metrics";
import { betaLabel } from "@/utils/polynomialFeatures";
import { LAMBDA_PATH, lambdaLabel } from "@/utils/regularization";

const TEST_RATIO = 0.3;
const MAX_DEGREE = 8;

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

const MYTHS = [
  {
    q: "“Lasso always removes features.”",
    a: "Lasso can produce zero coefficients, but whether it does depends on the data, λ, feature scaling and the optimization setup. At λ = 0 nothing is removed at all.",
  },
  {
    q: "“Lasso always performs better than Ridge.”",
    a: "Neither method is universally better. Which one helps depends on the dataset and on whether a sparse model is something you want.",
  },
  {
    q: "“A zero coefficient means the feature is objectively useless.”",
    a: "It means this fitted model, with this scaling and this λ, assigned that feature a zero weight. That is a statement about the fit, not about the world.",
  },
  {
    q: "“Lasso minimizes MSE.”",
    a: "Lasso minimizes MSE plus an L1 penalty. Its MSE can be higher than the OLS one while its own objective is lower.",
  },
  {
    q: "“Ridge and Lasso use the same penalty.”",
    a: "Ridge uses L2: λ Σβⱼ². Lasso uses L1: λ Σ|βⱼ|. The squared penalty shrinks; the absolute-value penalty can zero out.",
  },
  {
    q: "“Higher λ is always better.”",
    a: "Excessive regularization removes so much flexibility that the model underfits — the same trap seen with Ridge.",
  },
];

const CHECKPOINTS = [
  {
    q: "What problem does Lasso address?",
    a: "When a model carries many features, some contribute very little. Lasso adds a cost for coefficient size that can push those weights all the way to zero, leaving a simpler model.",
  },
  {
    q: "What is L1 regularization?",
    a: "A penalty equal to λ times the sum of the absolute values of the coefficients, λ Σ|βⱼ|, with the intercept excluded.",
  },
  {
    q: "What does λ control?",
    a: "How much the penalty counts relative to prediction error. λ = 0 means no penalty; a very large λ squeezes almost every coefficient to zero.",
  },
  {
    q: "Why can Lasso produce zero coefficients?",
    a: "The absolute value has a sharp corner at zero. In the coordinate update that corner becomes a threshold: if the raw update is smaller than the threshold, the exact optimum for that coefficient is zero.",
  },
  {
    q: "What is soft thresholding?",
    a: "S(z, γ) = sign(z)·max(|z| − γ, 0). It moves a value γ closer to zero, and anything smaller than γ becomes exactly zero.",
  },
  {
    q: "Why is Lasso useful for sparse models?",
    a: "Because the zeros are exact, the fitted model can be written with fewer terms — easier to read, and it needs fewer feature values to make a prediction.",
  },
  {
    q: "Does Lasso always remove features?",
    a: "No. With a small λ, or with features that all carry signal, every coefficient can stay non-zero.",
  },
  {
    q: "Why is feature scaling important?",
    a: "The penalty acts on coefficient magnitude. A feature measured in thousands naturally gets a tiny coefficient, so the same penalty hits features very unevenly unless they are standardized first.",
  },
  {
    q: "What is the difference between L1 and L2 regularization?",
    a: "L2 (Ridge) penalizes β², growing quadratically and shrinking smoothly toward zero. L1 (Lasso) penalizes |β|, growing linearly, and its constant pull can reach zero exactly.",
  },
  {
    q: "Why is coordinate descent commonly used for Lasso?",
    a: "Because |β| is not differentiable at zero, plain gradient descent has no well-defined step there. Optimizing one coefficient at a time has a closed-form answer — soft thresholding — that handles zero exactly.",
  },
  {
    q: "What happens when λ = 0?",
    a: "The penalty disappears and the objective is just MSE, so Lasso becomes ordinary least squares.",
  },
  {
    q: "What can happen when λ becomes extremely large?",
    a: "Every penalized coefficient is driven to zero, the model keeps only the intercept, and it predicts the mean of the training targets — severe underfitting.",
  },
];

export default function LassoRegressionLab() {
  const [presetId, setPresetId] = useState("noisy-curve");
  const preset = POLY_PRESETS.find((p) => p.id === presetId) ?? POLY_PRESETS[0]!;
  const [seed, setSeed] = useState(DEFAULT_POLY_SEED);
  const [noise, setNoise] = useState(preset.noise);
  const [degree, setDegree] = useState(5);
  const [lambda, setLambda] = useState(0);
  const [withNoiseFeatures, setWithNoiseFeatures] = useState(true);
  const [withCorrelated, setWithCorrelated] = useState(false);
  const [standardize, setStandardize] = useState(true);
  const [showSplit, setShowSplit] = useState(true);
  const [showResiduals, setShowResiduals] = useState(false);
  const [showRidgeCurve, setShowRidgeCurve] = useState(false);
  const [selectedCoef, setSelectedCoef] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [z, setZ] = useState(5);
  const [gamma, setGamma] = useState(2);

  const points = useMemo(() => preset.build(seed, noise), [preset, seed, noise]);
  const split = useMemo(() => trainTestSplit(points, TEST_RATIO, seed), [points, seed]);
  const train = showSplit ? split.train : points;
  const test = showSplit ? split.test : [];

  const specs = useMemo(
    () => buildFeatures(degree, { noise: withNoiseFeatures, correlated: withCorrelated }),
    [degree, withNoiseFeatures, withCorrelated],
  );

  const lasso = useMemo(
    () => fitLasso(train, specs, lambda, { standardize }),
    [train, specs, lambda, standardize],
  );
  const ridge = useMemo(
    () => fitRidgeFeatures(train, specs, lambda, standardize),
    [train, specs, lambda, standardize],
  );
  const ols = useMemo(() => fitRidgeFeatures(train, specs, 0, standardize), [train, specs, standardize]);

  const scaling: ColumnScaling | null = lasso.scaling;
  const predictL = useCallback(
    (x: number) => predictWith(lasso.beta, x, specs, scaling),
    [lasso.beta, specs, scaling],
  );
  const predictR = useCallback(
    (x: number) => predictWith(ridge.beta, x, specs, ridge.scaling),
    [ridge.beta, specs, ridge.scaling],
  );

  const trainMse = lasso.mse;
  const testMse = useMemo(
    () => (showSplit && test.length > 0 ? mseWith(test, lasso.beta, specs, scaling) : null),
    [showSplit, test, lasso.beta, specs, scaling],
  );
  const ridgeTrain = useMemo(() => mseWith(train, ridge.beta, specs, ridge.scaling), [train, ridge, specs]);
  const ridgeTest = useMemo(
    () => (showSplit && test.length > 0 ? mseWith(test, ridge.beta, specs, ridge.scaling) : null),
    [showSplit, test, ridge, specs],
  );
  const olsTrain = useMemo(() => mseWith(train, ols.beta, specs, ols.scaling), [train, ols, specs]);
  const olsTest = useMemo(
    () => (showSplit && test.length > 0 ? mseWith(test, ols.beta, specs, ols.scaling) : null),
    [showSplit, test, ols, specs],
  );

  /* ------------- λ sweep: paths, sparsity, error ------------- */
  const sweep = useMemo(
    () =>
      LAMBDA_PATH.map((l) => {
        const fit = fitLasso(train, specs, l, { standardize, maxSweeps: 200 });
        return {
          lambda: l,
          beta: fit.beta,
          ok: fit.ok,
          active: fit.active,
          trainMse: fit.mse,
          testMse: showSplit && test.length > 0 ? mseWith(test, fit.beta, specs, fit.scaling) : null,
        };
      }),
    [train, test, specs, standardize, showSplit],
  );

  const pathRows: LassoPathRow[] = sweep.map((s) => ({ lambda: s.lambda, beta: s.beta, ok: s.ok }));
  const sparsityRows = sweep.map((s) => ({ lambda: s.lambda, active: s.active }));
  const errorRows: ErrorRow[] = sweep.map((s) => ({
    lambda: s.lambda,
    trainMse: s.trainMse,
    testMse: s.testMse,
    ok: s.ok,
  }));

  /* ------------- Interactive coordinate descent ------------- */
  const p = specs.length;
  const [stepBeta, setStepBeta] = useState<number[]>(() => new Array(p + 1).fill(0));
  const [coordinate, setCoordinate] = useState(0);
  const [lastUpdate, setLastUpdate] = useState<CoordinateUpdate | null>(null);
  const [updates, setUpdates] = useState(0);
  const [sweeps, setSweeps] = useState(0);
  const [playing, setPlaying] = useState(false);

  const resetStepper = useCallback(() => {
    setPlaying(false);
    setStepBeta(new Array(p + 1).fill(0));
    setCoordinate(0);
    setLastUpdate(null);
    setUpdates(0);
    setSweeps(0);
  }, [p]);

  useEffect(() => {
    resetStepper();
  }, [resetStepper, lasso.X, lambda]);

  const doStep = useCallback(() => {
    const X = lasso.X;
    const y = lasso.y;
    if (X.length === 0) return;
    setStepBeta((current) => {
      const next = current.slice();
      if (coordinate === 0) {
        next[0] = updateIntercept(X, y, next);
        setLastUpdate(null);
      } else {
        const u = coordinateUpdate(X, y, next, coordinate, lambda);
        next[coordinate] = u.after;
        setLastUpdate(u);
      }
      return next;
    });
    setUpdates((n) => n + 1);
    const nextCoord = coordinate + 1 > p ? 0 : coordinate + 1;
    if (nextCoord === 0) setSweeps((s) => s + 1);
    setCoordinate(nextCoord);
  }, [lasso.X, lasso.y, coordinate, lambda, p]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(doStep, 150);
    return () => window.clearInterval(id);
  }, [playing, doStep]);

  const stepObjective = useMemo(
    () => lassoObjective(lasso.X, lasso.y, stepBeta, lambda),
    [lasso.X, lasso.y, stepBeta, lambda],
  );
  const stepperMatchesSolver = useMemo(
    () => stepBeta.every((b, j) => Math.abs(b - (lasso.beta[j] ?? 0)) < 1e-6),
    [stepBeta, lasso.beta],
  );

  const stepHistory: HistoryEntry[] = lasso.history;

  /* ------------- Derived numbers ------------- */
  const penalty = l1Penalty(lasso.beta, lambda);
  const magnitude = l1Penalty(lasso.beta, 1);
  const ridgePenalty = l2PenaltyOf(ridge.beta, lambda);
  const active = activeCount(lasso.beta);
  const ridgeActive = activeCount(ridge.beta);
  const num = (v: number | null) =>
    v === null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 1e6 ? v.toExponential(2) : fmt(v, 4);

  const message = !lasso.ok
    ? (lasso.reason ??
      "Numerical instability detected. Try reducing polynomial degree or enabling feature standardization.")
    : !lasso.converged
      ? "The optimizer has not converged yet. Try increasing the iteration limit or adjusting λ."
      : lambda === 0
        ? "λ is zero, so there is no penalty at all: this is ordinary least squares, and no coefficient is removed."
        : active === specs.length
          ? `λ = ${lambdaLabel(lambda)} is shrinking the coefficients, but none has reached exactly zero yet. Raise λ further.`
          : `λ = ${lambdaLabel(lambda)} left ${active} of ${specs.length} features active — the rest have coefficients of exactly zero.`;

  const applyPreset = (id: string) => {
    const next = POLY_PRESETS.find((pr) => pr.id === id);
    if (!next) return;
    setPresetId(id);
    setNoise(next.noise);
  };

  const selectedSpec = selectedCoef && selectedCoef > 0 ? specs[selectedCoef - 1] : null;
  const selectedValue = selectedCoef !== null ? (lasso.beta[selectedCoef] ?? 0) : 0;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
          <div>
            <h1 className="font-display text-4xl tracking-tight">Lasso Regression</h1>
            <p className="mt-1 text-muted-foreground">
              What if we want the model to ignore some features?
            </p>
          </div>
          <SiteNav />
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-14 px-5 py-10">
        {/* 01 */}
        <Step n={1} title="What if we want the model to ignore some features?">
          <div className="grid gap-8 lg:grid-cols-[1fr_340px]">
            <div className="rounded-md border border-border p-4">
              <FeatureCurveChart
                train={train}
                test={test}
                predict={predictL}
                {...(showRidgeCurve ? { comparePredict: predictR, compareLabel: "Ridge" } : {})}
                showResiduals={showResiduals}
                showSplit={showSplit}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
              <p className="mt-3 font-mono text-sm">
                {lasso.ok ? featureEquation(lasso.beta, specs) : "—"}
              </p>
            </div>
            <div className="space-y-4">
              <p className="text-sm leading-relaxed">
                This model carries {specs.length} features: the polynomial expansion x … x
                <sup>{degree}</sup>
                {withNoiseFeatures ? ", plus two wiggles that have nothing to do with the target" : ""}
                {withCorrelated ? ", plus a near-copy of x" : ""}. When we create many features, some
                contribute very little to the prediction. Can we encourage the model to remove them
                completely?
              </p>
              <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Ridge — L2 penalty
     ↓
Coefficients shrink
     ↓
Usually remain non-zero

Lasso — L1 penalty
     ↓
Coefficients shrink
     ↓
Some can become exactly zero
     ↓
Feature selection`}</pre>
              <div className="rounded-md border border-border p-4">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">Dataset</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {POLY_PRESETS.map((pr) => (
                    <Button
                      key={pr.id}
                      size="sm"
                      variant={pr.id === presetId ? "default" : "outline"}
                      onClick={() => applyPreset(pr.id)}
                    >
                      {pr.name}
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
                  <Button size="sm" variant="ghost" onClick={() => setSeed(Math.floor(Math.random() * 1e8))}>
                    <Dices className="mr-1 size-4" /> Regenerate
                  </Button>
                </div>
              </div>
              <div className="space-y-3 rounded-md border border-border p-4 text-sm">
                {[
                  { label: "Extra noise features", value: withNoiseFeatures, set: setWithNoiseFeatures },
                  { label: "Correlated feature (x_twin)", value: withCorrelated, set: setWithCorrelated },
                  { label: "Standardize features", value: standardize, set: setStandardize },
                  { label: "Show train/test split", value: showSplit, set: setShowSplit },
                  { label: "Show residuals", value: showResiduals, set: setShowResiduals },
                  { label: "Show Ridge curve too", value: showRidgeCurve, set: setShowRidgeCurve },
                ].map((row) => (
                  <label key={row.label} className="flex items-center justify-between gap-3">
                    <span>{row.label}</span>
                    <Switch checked={row.value} onCheckedChange={row.set} />
                  </label>
                ))}
              </div>
            </div>
          </div>
        </Step>

        {/* 02 */}
        <Step n={2} title="The Lasso objective">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-md border border-border p-4">
              <pre className="overflow-x-auto text-sm leading-relaxed">{`Ordinary regression
  minimize   MSE

Lasso regression
  minimize   MSE + λ Σ|βⱼ|     (j = 1 … p)`}</pre>
              <dl className="mt-4 space-y-2 text-sm">
                <div>
                  <dt className="font-medium">MSE</dt>
                  <dd className="text-muted-foreground">Measures prediction error.</dd>
                </div>
                <div>
                  <dt className="font-medium">λ</dt>
                  <dd className="text-muted-foreground">Controls regularization strength.</dd>
                </div>
                <div>
                  <dt className="font-medium">|βⱼ|</dt>
                  <dd className="text-muted-foreground">
                    The absolute magnitude of a coefficient. The intercept β₀ is excluded from the
                    penalty — it only shifts the model up or down.
                  </dd>
                </div>
              </dl>
            </div>
            <div className="rounded-md border border-border">
              <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
                Where does the extra loss come from?
              </h3>
              <dl className="px-4 py-3 font-mono text-sm">
                <div className="flex justify-between py-1">
                  <dt className="text-muted-foreground">MSE (prediction loss)</dt>
                  <dd className="tabular-nums">{num(trainMse)}</dd>
                </div>
                <div className="flex justify-between py-1">
                  <dt className="text-muted-foreground">L1 penalty = λ Σ|βⱼ|</dt>
                  <dd className="tabular-nums">{num(penalty)}</dd>
                </div>
                <div className="my-1 border-t border-border" />
                <div className="flex justify-between py-1 text-base">
                  <dt>Lasso objective</dt>
                  <dd className="tabular-nums">{num(lasso.objective)}</dd>
                </div>
              </dl>
              <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
                With λ = {lambdaLabel(lambda)} and Σ|βⱼ| = {num(magnitude)} (intercept excluded), the
                penalty is {lambdaLabel(lambda)} × {num(magnitude)} = {num(penalty)}.
              </p>
            </div>
          </div>
        </Step>

        {/* 03 */}
        <Step n={3} title="Turn up the regularization strength">
          <div className="grid gap-6 lg:grid-cols-2">
            <RegularizationControl lambda={lambda} onChange={setLambda} />
            <DegreeControl degree={degree} onChange={setDegree} max={MAX_DEGREE} />
          </div>
          <div className="mt-4 grid grid-cols-1 divide-y divide-border rounded-md border border-border sm:grid-cols-5 sm:divide-x sm:divide-y-0">
            {[
              { label: "λ", value: lambdaLabel(lambda), sub: "regularization strength" },
              { label: "Training MSE", value: num(trainMse), sub: "how wrong the predictions are" },
              {
                label: "Test MSE",
                value: showSplit ? num(testMse) : "—",
                sub: showSplit ? "unseen observations" : "turn on the train/test split",
              },
              { label: "Lasso objective", value: num(lasso.objective), sub: "MSE + λ Σ|βⱼ|" },
              {
                label: "Active features",
                value: `${active} / ${specs.length}`,
                sub: "coefficients that are not exactly zero",
              },
            ].map((it) => (
              <div key={it.label} className="px-4 py-3">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">{it.label}</div>
                <div className="mt-1 font-mono text-xl tabular-nums">{it.value}</div>
                <div className="text-xs text-muted-foreground">{it.sub}</div>
              </div>
            ))}
          </div>
          <div className="mt-4">
            <ExplanationPanel message={message} />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            At λ = 0 there is no penalty term at all, so the Lasso objective is exactly the ordinary
            least squares objective from the first page of this lab.
          </p>
        </Step>

        {/* 04 */}
        <Step n={4} title="Watch the coefficients change">
          <div className="grid gap-6 lg:grid-cols-2">
            <LassoCoefficientChart
              lasso={lasso.beta}
              specs={specs}
              selected={selectedCoef}
              onSelect={setSelectedCoef}
            />
            <div className="space-y-4">
              <FeatureSelectionPanel beta={lasso.beta} specs={specs} />
              {selectedCoef !== null && selectedCoef > 0 && (
                <div className="rounded-md border border-border p-4 text-sm">
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">
                    Inspecting {betaLabel(selectedCoef)}
                  </div>
                  <dl className="mt-2 space-y-1 font-mono">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Feature</dt>
                      <dd>{selectedSpec?.label}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Coefficient</dt>
                      <dd>{fmt(selectedValue, 4)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">L1 contribution λ|βⱼ|</dt>
                      <dd>{fmt(lambda * Math.abs(selectedValue), 4)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Status</dt>
                      <dd>{selectedValue === 0 ? "removed" : "selected"}</dd>
                    </div>
                  </dl>
                </div>
              )}
              <div className="rounded-md border border-dashed border-axis/60 bg-muted/40 p-4 text-sm">
                <h3 className="font-medium">Lasso can perform feature selection</h3>
                <p className="mt-2 text-muted-foreground">
                  When Lasso drives a coefficient to exactly zero, that feature is effectively
                  excluded from the model: multiplying it by zero contributes nothing. Ridge
                  generally shrinks coefficients toward zero, while Lasso <em>can</em> produce sparse
                  models with fewer active coefficients. Whether it does depends on the data, the
                  scaling and λ.
                </p>
              </div>
            </div>
          </div>
        </Step>

        {/* 05 */}
        <Step n={5} title="What the L1 penalty looks like">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-md border border-border p-4">
              <PenaltyShapeChart />
            </div>
            <div className="space-y-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-widest text-muted-foreground">
                    <th className="py-2 font-normal">β</th>
                    <th className="py-2 text-right font-normal">Lasso |β|</th>
                    <th className="py-2 text-right font-normal">Ridge β²</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60 font-mono">
                  {[-3, -2, -1, 0, 1, 2, 3].map((b) => (
                    <tr key={b}>
                      <td className="py-1.5">{b}</td>
                      <td className="py-1.5 text-right">{Math.abs(b)}</td>
                      <td className="py-1.5 text-right">{b * b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-sm text-muted-foreground">
                L1 treats positive and negative coefficients symmetrically, because the absolute
                value ignores the sign: β = 3 and β = −3 both cost 3. Ridge grows quadratically with
                coefficient magnitude; Lasso grows linearly — that constant pull is what lets it
                reach zero.
              </p>
            </div>
          </div>
        </Step>

        {/* 06 */}
        <Step n={6} title="Why does Lasso produce zeros?">
          <div className="grid gap-6 lg:grid-cols-2">
            <p className="text-sm leading-relaxed">
              The shape of the L1 penalty makes zero a special point in the optimization problem. A
              squared penalty gets gentler and gentler as a coefficient approaches zero, so it never
              quite arrives. The absolute value keeps pulling with the same strength all the way in.
              That does not mean the absolute value always forces coefficients to zero — under
              suitable conditions and a large enough λ, the optimal value for a coefficient
              <em> is</em> exactly zero, and the solver lands on it.
            </p>
            <Accordion type="single" collapsible className="rounded-md border border-border px-4">
              <AccordionItem value="sub" className="border-none">
                <AccordionTrigger>Advanced: why is Lasso different mathematically?</AccordionTrigger>
                <AccordionContent>
                  <p className="text-sm text-muted-foreground">
                    |β| is not differentiable at β = 0: the slope jumps from −1 to +1. Ordinary
                    derivatives are not enough there, so optimization uses the{" "}
                    <em>subgradient</em> — the set of slopes that stay below the function.
                  </p>
                  <pre className="mt-3 overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed">{`For β > 0:   subgradient = +1
For β < 0:   subgradient = −1
At  β = 0:   subgradient ∈ [−1, 1]`}</pre>
                  <p className="mt-3 text-sm text-muted-foreground">
                    Because zero admits a whole interval of slopes, the optimality condition can be
                    satisfied exactly at zero. That is what makes exact zeros possible, and it is
                    why the coordinate update ends in a threshold rather than a plain division.
                  </p>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </Step>

        {/* 07 */}
        <Step n={7} title="Soft thresholding">
          <div className="rounded-md border border-border p-4">
            <SoftThresholdVisualization z={z} gamma={gamma} onZ={setZ} onGamma={setGamma} />
            <p className="mt-4 text-sm text-muted-foreground">
              A large raw coefficient is shrunk toward zero; a small one becomes exactly zero. This
              is the single operation that gives Lasso its sparsity, and it is applied to every
              coefficient in every coordinate-descent update below.
            </p>
          </div>
        </Step>

        {/* 08 */}
        <Step n={8} title="Watch Lasso learn">
          <OptimizationPanel
            beta={stepBeta}
            specs={specs}
            coordinate={coordinate}
            lastUpdate={lastUpdate}
            sweeps={sweeps}
            updates={updates}
            playing={playing}
            converged={stepperMatchesSolver && updates > 0}
            objective={stepObjective.objective}
            mse={stepObjective.mse}
            penalty={stepObjective.penalty}
            onStep={doStep}
            onNextCoordinate={() => setCoordinate((c) => (c + 1 > p ? 0 : c + 1))}
            onPlayToggle={() => setPlaying((v) => !v)}
            onReset={resetStepper}
          />
          <div className="mt-6 rounded-md border border-border p-4">
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
              Lasso objective vs iteration
            </h3>
            <div className="mt-3">
              <ObjectiveHistoryChart history={stepHistory} />
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              This is the history of the full solver at the current λ: one point per sweep over all
              coefficients. MSE + L1 penalty = Lasso objective, and the objective decreases as the
              optimization converges.
            </p>
          </div>
        </Step>

        {/* 09 */}
        <Step n={9} title="Coefficient paths and sparsity">
          <div className="space-y-8">
            <div className="rounded-md border border-border p-4">
              <LassoPathChart rows={pathRows} specs={specs} selectedLambda={lambda} />
            </div>
            <div className="rounded-md border border-border p-4">
              <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
                Sparsity vs regularization
              </h3>
              <div className="mt-3">
                <SparsityChart rows={sparsityRows} total={specs.length} selectedLambda={lambda} />
              </div>
            </div>
          </div>
        </Step>

        {/* 10 */}
        <Step n={10} title="Ridge vs Lasso, same data and same λ">
          <div className="grid gap-6 lg:grid-cols-2">
            <LassoCoefficientChart
              lasso={lasso.beta}
              ridge={ridge.beta}
              specs={specs}
              title="Coefficients: Lasso vs Ridge"
            />
            <div className="space-y-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-widest text-muted-foreground">
                    <th className="py-2 font-normal">Metric</th>
                    <th className="py-2 text-right font-normal">OLS</th>
                    <th className="py-2 text-right font-normal">Ridge</th>
                    <th className="py-2 text-right font-normal">Lasso</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60 font-mono">
                  <tr>
                    <td className="py-1.5">Training MSE</td>
                    <td className="py-1.5 text-right">{num(olsTrain)}</td>
                    <td className="py-1.5 text-right">{num(ridgeTrain)}</td>
                    <td className="py-1.5 text-right">{num(trainMse)}</td>
                  </tr>
                  <tr>
                    <td className="py-1.5">Test MSE</td>
                    <td className="py-1.5 text-right">{showSplit ? num(olsTest) : "—"}</td>
                    <td className="py-1.5 text-right">{showSplit ? num(ridgeTest) : "—"}</td>
                    <td className="py-1.5 text-right">{showSplit ? num(testMse) : "—"}</td>
                  </tr>
                  <tr>
                    <td className="py-1.5">Active coefficients</td>
                    <td className="py-1.5 text-right">{activeCount(ols.beta)}</td>
                    <td className="py-1.5 text-right">{ridgeActive}</td>
                    <td className="py-1.5 text-right">{active}</td>
                  </tr>
                  <tr>
                    <td className="py-1.5">Penalty value</td>
                    <td className="py-1.5 text-right">0</td>
                    <td className="py-1.5 text-right">{num(ridgePenalty)}</td>
                    <td className="py-1.5 text-right">{num(penalty)}</td>
                  </tr>
                  <tr>
                    <td className="py-1.5">Penalty type</td>
                    <td className="py-1.5 text-right">none</td>
                    <td className="py-1.5 text-right">L2 · λΣβ²</td>
                    <td className="py-1.5 text-right">L1 · λΣ|β|</td>
                  </tr>
                </tbody>
              </table>
              <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Prediction error
        +
Regularization penalty
        =
Optimization objective

Ridge:  MSE + λΣβ²
Lasso:  MSE + λΣ|β|`}</pre>
              <p className="text-sm text-muted-foreground">
                Both models use the same features, the same split and the same λ — only the penalty
                differs. Neither is ranked here: the point is to see how differently they behave.
                Turn on “Show Ridge curve too” at the top to see both curves on the data.
              </p>
            </div>
          </div>
        </Step>

        {/* 11 */}
        <Step n={11} title="Error and sparsity as λ changes">
          <div className="rounded-md border border-border p-4">
            <RidgeErrorChart
              rows={errorRows}
              selectedLambda={lambda}
              showTest={showSplit}
              onSelectLambda={setLambda}
            />
          </div>
          <div className="mt-6 overflow-x-auto rounded-md border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-widest text-muted-foreground">
                  <th className="px-4 py-2 font-normal">λ</th>
                  <th className="px-4 py-2 text-right font-normal">Active features</th>
                  <th className="px-4 py-2 text-right font-normal">Training MSE</th>
                  <th className="px-4 py-2 text-right font-normal">Test MSE</th>
                  <th className="px-4 py-2 text-right font-normal" />
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-mono">
                {sweep
                  .filter((_, i) => i % 2 === 0)
                  .map((s) => (
                    <tr key={s.lambda} className={s.lambda === lambda ? "bg-accent/20" : ""}>
                      <td className="px-4 py-1.5">{lambdaLabel(s.lambda)}</td>
                      <td className="px-4 py-1.5 text-right">
                        {s.active} / {specs.length}
                      </td>
                      <td className="px-4 py-1.5 text-right">{num(s.trainMse)}</td>
                      <td className="px-4 py-1.5 text-right">{showSplit ? num(s.testMse) : "—"}</td>
                      <td className="px-4 py-1.5 text-right">
                        <Button size="sm" variant="ghost" onClick={() => setLambda(s.lambda)}>
                          Use this λ
                        </Button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-muted-foreground">
            Increasing regularization can reduce model flexibility, but excessive regularization can
            also cause underfitting. The smallest test MSE is one signal among several — a simpler
            model with fewer active coefficients may be preferable for other reasons, and the right
            λ depends on the problem.
          </p>
        </Step>

        {/* 12 */}
        <Step n={12} title="Feature scaling, correlated features, and what a zero really means">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Raw features
     ↓
Standardization  (mean and std from the TRAINING set only)
     ↓
Lasso`}</pre>
              <p className="text-sm text-muted-foreground">
                Lasso penalizes coefficient magnitude, so scale matters: x⁵ over x ∈ [−5, 5] reaches
                3125 while x barely reaches 5, and the coefficient sizes are not comparable. With
                “Standardize features” on, means and standard deviations are computed from the
                training observations only and the same transformation is applied to the test set —
                no test statistics leak into training. The coefficients shown are the standardized
                ones, which are the values the penalty actually sees.
              </p>
              {standardize && (
                <div className="overflow-x-auto rounded-md border border-border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-widest text-muted-foreground">
                        <th className="px-3 py-2 font-normal">Feature</th>
                        <th className="px-3 py-2 text-right font-normal">mean</th>
                        <th className="px-3 py-2 text-right font-normal">std</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60 font-mono">
                      {specs.map((s, i) => (
                        <tr key={s.key}>
                          <td className="px-3 py-1.5">{s.label}</td>
                          <td className="px-3 py-1.5 text-right">{fmt(scaling?.means[i] ?? 0, 3)}</td>
                          <td className="px-3 py-1.5 text-right">{fmt(scaling?.stds[i] ?? 1, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            <div className="space-y-4">
              <div className="rounded-md border border-dashed border-axis/60 bg-muted/40 p-4 text-sm">
                <h3 className="font-medium">Important</h3>
                <p className="mt-2 text-muted-foreground">
                  A coefficient becoming zero means Lasso excluded that feature from this fitted
                  model, under this regularization setup. It does not prove the feature is
                  unimportant. Which features survive depends on the dataset, the feature scaling,
                  λ, the correlations between features and the optimization setup.
                </p>
              </div>
              <div className="rounded-md border border-border p-4 text-sm">
                <h3 className="font-medium">What happens with correlated features?</h3>
                <p className="mt-2 text-muted-foreground">
                  Turn on “Correlated feature (x_twin)” at the top: it adds x + 0.05·sin(2x), almost
                  a copy of x. Then raise λ. When predictors are strongly correlated, Lasso may
                  select one and shrink the other toward zero. That does not necessarily mean the
                  discarded feature contains no useful information — the two simply carry much of
                  the same signal.
                </p>
                {withCorrelated && (
                  <p className="mt-3 font-mono text-xs">
                    x: {fmt(lasso.beta[1] ?? 0, 4)} · x_twin:{" "}
                    {fmt(lasso.beta[specs.length] ?? 0, 4)}
                  </p>
                )}
              </div>
            </div>
          </div>
        </Step>

        {/* 13 */}
        <Step n={13} title="Under the hood">
          <Accordion type="single" collapsible className="rounded-md border border-border px-4">
            <AccordionItem value="geometry">
              <AccordionTrigger>Advanced: why L1 and L2 behave differently</AccordionTrigger>
              <AccordionContent>
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`Ridge (L2)                 Lasso (L1)

      β₂                         β₂
       ↑                          ↑
     ╭───╮                        ◆
    │     │                      / \\
    │  ●  │                     ●   ●
     ╰───╯                       \\ /
       → β₁                       ◆
                                   → β₁`}</pre>
                <p className="mt-3 text-sm text-muted-foreground">
                  Thinking of regularization as a budget on coefficient size, the L2 budget is a
                  round ball and the L1 budget is a diamond with sharp corners sitting on the axes.
                  A solution that touches a corner has one coordinate equal to zero — a high-level
                  picture of why Lasso can produce exact zeros and Ridge usually does not.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="code">
              <AccordionTrigger>From mathematics to code</AccordionTrigger>
              <AccordionContent>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">
                      Lasso objective
                    </div>
                    <pre className="mt-2 overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed">{`MSE + λ Σ|βⱼ|

l1_penalty = regularization_strength * sum(
    abs(beta)
    for beta in coefficients[1:]  # skip the intercept
)

lasso_objective = mse + l1_penalty`}</pre>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">
                      Soft thresholding
                    </div>
                    <pre className="mt-2 overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed">{`S(z, γ) = sign(z)·max(|z| − γ, 0)

def soft_threshold(z, gamma):
    return copysign(1, z) * max(abs(z) - gamma, 0)`}</pre>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="algos">
              <AccordionTrigger>How each model in this lab is optimized</AccordionTrigger>
              <AccordionContent>
                <table className="w-full text-sm">
                  <tbody className="divide-y divide-border/60">
                    {[
                      ["Linear Regression", "OLS / gradient descent"],
                      ["Polynomial Regression", "Least squares (QR) / gradient descent"],
                      ["Ridge", "Closed form (augmented least squares) / gradient descent"],
                      ["Lasso", "Coordinate descent with soft thresholding"],
                    ].map(([model, how]) => (
                      <tr key={model}>
                        <td className="py-1.5">{model}</td>
                        <td className="py-1.5 text-right text-muted-foreground">{how}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-3 text-sm text-muted-foreground">
                  Lasso is commonly optimized with coordinate descent because the L1 penalty is not
                  differentiable at zero: one coefficient at a time, the update has an exact answer.
                </p>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="myths" className="border-none">
              <AccordionTrigger>Lasso Regression myths</AccordionTrigger>
              <AccordionContent>
                <ul className="space-y-3 text-sm">
                  {MYTHS.map((m) => (
                    <li key={m.q}>
                      <div className="font-medium">{m.q}</div>
                      <div className="text-muted-foreground">{m.a}</div>
                    </li>
                  ))}
                </ul>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Step>

        {/* 14 */}
        <Step n={14} title="Can you explain Lasso Regression?">
          <Accordion type="single" collapsible className="rounded-md border border-border px-4">
            {CHECKPOINTS.map((c, i) => (
              <AccordionItem key={c.q} value={`q${i}`} className={i === CHECKPOINTS.length - 1 ? "border-none" : ""}>
                <AccordionTrigger className="text-left">
                  {i + 1}. {c.q}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">{c.a}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </Step>

        {/* 15 */}
        <Step n={15} title="Lasso Regression in one picture">
          <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
            <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-4 text-xs leading-relaxed">{`            DATA
              ↓
      Feature Expansion
              ↓
            Model
              ↓
         Prediction
              ↓
      Prediction Error
              ↓
            MSE
              +
     L1 Regularization
              ↓
        Penalize |β|
              ↓
     Soft Thresholding
              ↓
   Some β can become 0
              ↓
        Sparse Model
              ↓
  Evaluate on Test Data`}</pre>
            <div className="space-y-4">
              <FlowDiagram
                steps={[
                  "Data",
                  "Features",
                  "Predictions",
                  "MSE",
                  "L1 penalty",
                  "Soft thresholding",
                  "Sparse model",
                  "Test error",
                ]}
                active
              />
              <p className="text-sm text-muted-foreground">
                Right now: {specs.length} features, {active} active, λ = {lambdaLabel(lambda)},
                training MSE {num(trainMse)}
                {showSplit ? `, test MSE ${num(testMse)}` : ""}, Lasso objective{" "}
                {num(lasso.objective)}.
              </p>
            </div>
          </div>
        </Step>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-6xl px-5 py-8 text-sm text-muted-foreground">
          Every coefficient, penalty, objective, zero and curve on this page comes from coordinate
          descent running in your browser on the dataset shown — nothing is hard-coded.
        </div>
      </footer>
    </div>
  );
}
