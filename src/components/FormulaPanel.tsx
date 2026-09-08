import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { OlsSteps } from "@/algorithms/linearRegression";
import { Button } from "@/components/ui/button";
import { fmt } from "@/utils/calculations";

function Section({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t border-border py-4">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="font-display text-lg">{title}</span>
        <ChevronDown
          className={`size-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && <div className="mt-4 animate-in fade-in slide-in-from-top-1 duration-200">{children}</div>}
    </div>
  );
}

export function FormulaPanel({ steps }: { steps: OlsSteps }) {
  const [showCalc, setShowCalc] = useState(false);

  return (
    <div>
      <Section title="How does the computer find the best line?" defaultOpen>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          For a line with a single input, the best-fitting line does not have to be searched for by
          trial and error. It can be calculated directly, using a method called{" "}
          <strong className="text-foreground">Ordinary Least Squares</strong>.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <pre className="rounded-md bg-muted p-4 font-mono text-xs leading-relaxed">{`β₁ =  Σ((xᵢ − x̄)(yᵢ − ȳ))
      ───────────────────
          Σ(xᵢ − x̄)²`}</pre>
          <pre className="rounded-md bg-muted p-4 font-mono text-xs leading-relaxed">{`β₀ = ȳ − β₁x̄`}</pre>
        </div>
        <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
          You do not need to memorise these. They simply calculate the slope and intercept that
          produce the least possible squared error.
        </p>
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          onClick={() => setShowCalc((s) => !s)}
        >
          {showCalc ? "Hide calculation" : "Show calculation"}
        </Button>

        {showCalc && (
          <ol className="mt-4 space-y-3 font-mono text-xs leading-relaxed animate-in fade-in duration-200">
            <li>
              <span className="font-sans text-sm">1. Mean of x</span>
              <div className="text-muted-foreground">x̄ = {fmt(steps.meanX, 4)}</div>
            </li>
            <li>
              <span className="font-sans text-sm">2. Mean of y</span>
              <div className="text-muted-foreground">ȳ = {fmt(steps.meanY, 4)}</div>
            </li>
            <li>
              <span className="font-sans text-sm">3. Deviations from the means</span>
              <div className="mt-1 max-h-56 overflow-auto rounded border border-border">
                <table className="w-full">
                  <thead className="bg-muted text-left">
                    <tr>
                      <th className="px-2 py-1">x</th>
                      <th className="px-2 py-1">y</th>
                      <th className="px-2 py-1">x − x̄</th>
                      <th className="px-2 py-1">y − ȳ</th>
                      <th className="px-2 py-1">(x−x̄)(y−ȳ)</th>
                      <th className="px-2 py-1">(x−x̄)²</th>
                    </tr>
                  </thead>
                  <tbody className="tabular-nums">
                    {steps.deviations.map((d, i) => (
                      <tr key={i} className="border-t border-border">
                        <td className="px-2 py-1">{fmt(d.x)}</td>
                        <td className="px-2 py-1">{fmt(d.y)}</td>
                        <td className="px-2 py-1">{fmt(d.dx)}</td>
                        <td className="px-2 py-1">{fmt(d.dy)}</td>
                        <td className="px-2 py-1">{fmt(d.dxdy)}</td>
                        <td className="px-2 py-1">{fmt(d.dx2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </li>
            <li>
              <span className="font-sans text-sm">4. Numerator</span>
              <div className="text-muted-foreground">Σ(x−x̄)(y−ȳ) = {fmt(steps.numerator, 4)}</div>
            </li>
            <li>
              <span className="font-sans text-sm">5. Denominator</span>
              <div className="text-muted-foreground">Σ(x−x̄)² = {fmt(steps.denominator, 4)}</div>
            </li>
            <li>
              <span className="font-sans text-sm">6. Slope</span>
              <div className="text-muted-foreground">
                β₁ = {fmt(steps.numerator, 4)} ÷ {fmt(steps.denominator, 4)} ={" "}
                {fmt(steps.slope, 4)}
              </div>
            </li>
            <li>
              <span className="font-sans text-sm">7. Intercept</span>
              <div className="text-muted-foreground">
                β₀ = {fmt(steps.meanY, 4)} − ({fmt(steps.slope, 4)} × {fmt(steps.meanX, 4)}) ={" "}
                {fmt(steps.intercept, 4)}
              </div>
            </li>
            <li>
              <span className="font-sans text-sm">8. Final equation</span>
              <div className="text-foreground">
                ŷ = {fmt(steps.intercept)} {steps.slope < 0 ? "−" : "+"}{" "}
                {fmt(Math.abs(steps.slope))}x
              </div>
            </li>
          </ol>
        )}
      </Section>

      <Section title="From mathematics to code">
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            { math: "ŷ = β₀ + β₁x", code: "const prediction = intercept + slope * x;" },
            { math: "error = y − ŷ", code: "const error = actual - prediction;" },
            { math: "MSE = average(error²)", code: "const squaredError = error ** 2;" },
          ].map((row) => (
            <div key={row.math} className="space-y-2">
              <div className="text-xs uppercase tracking-widest text-muted-foreground">
                Mathematics
              </div>
              <pre className="rounded-md bg-muted p-3 font-mono text-xs">{row.math}</pre>
              <div className="text-xs uppercase tracking-widest text-muted-foreground">
                JavaScript
              </div>
              <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs">
                {row.code}
              </pre>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
