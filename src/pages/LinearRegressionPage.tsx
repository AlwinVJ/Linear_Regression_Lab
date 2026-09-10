import { useMemo } from "react";
import { Link } from "@tanstack/react-router";
import {
  buildBreakdown,
  calculateMSE,
  fitOLS,
  olsSteps,
} from "@/algorithms/linearRegression";
import { FormulaPanel } from "@/components/FormulaPanel";
import { RegressionChart } from "@/components/RegressionChart";
import { ResidualTable } from "@/components/ResidualTable";
import { SiteNav } from "@/components/SiteNav";
import { Term } from "@/components/Term";
import { TooltipProvider } from "@/components/ui/tooltip";
import { equationString, fmt } from "@/utils/calculations";
import { defaultDataset } from "@/utils/dataset";

export default function LinearRegressionPage() {
  const points = useMemo(() => defaultDataset(), []);
  const params = useMemo(() => fitOLS(points), [points]);
  const rows = useMemo(() => buildBreakdown(points, params), [points, params]);
  const mse = useMemo(() => calculateMSE(points, params), [points, params]);
  const steps = useMemo(() => olsSteps(points), [points]);

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Linear Regression</h1>
              <p className="mt-1 text-muted-foreground">
                The model, and the direct formula that solves it.
              </p>
            </div>
            <SiteNav />
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-12 px-5 py-10">
          <section className="grid gap-8 lg:grid-cols-[1fr_360px]">
            <div className="rounded-md border border-border bg-card p-4">
              <RegressionChart
                points={points}
                params={params}
                showResiduals
                selectedId={null}
                onSelect={() => {}}
              />
            </div>
            <div className="space-y-4 text-sm text-muted-foreground">
              <div className="rounded-md bg-muted p-5 font-mono text-xl text-foreground">
                {equationString(params.intercept, params.slope)}
              </div>
              <p>
                The <Term name="model" /> is a straight line. Its two{" "}
                <Term name="parameters" /> — β₀ and β₁ — are everything the model knows. A{" "}
                <Term name="prediction" /> is what the line says for a given input, the{" "}
                <Term name="error" /> is the gap to reality, and the <Term name="loss" /> summarises
                all those gaps in one number.
              </p>
              <p className="font-mono text-xs">
                β₀ = {fmt(params.intercept, 4)} · β₁ = {fmt(params.slope, 4)} · MSE = {fmt(mse, 4)}
              </p>
              <p>
                Want to move the line yourself?{" "}
                <Link to="/" className="underline underline-offset-4">
                  Open the playground
                </Link>
                . Want the computer to find it step by step?{" "}
                <Link to="/gradient-descent" className="underline underline-offset-4">
                  Try gradient descent
                </Link>
                .
              </p>
            </div>
          </section>

          <section>
            <h2 className="font-display text-2xl">Every prediction and error</h2>
            <div className="mt-4">
              <ResidualTable rows={rows} mse={mse} selectedId={null} onSelect={() => {}} />
            </div>
          </section>

          <section>
            <h2 className="font-display text-2xl">Ordinary Least Squares, worked out</h2>
            <div className="mt-4">
              <FormulaPanel steps={steps} />
            </div>
          </section>
        </main>
      </div>
    </TooltipProvider>
  );
}
