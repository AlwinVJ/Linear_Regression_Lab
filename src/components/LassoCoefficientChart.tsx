import type { FeatureSpec } from "@/algorithms/featureRegression";
import { fmt } from "@/utils/calculations";
import { betaLabel } from "@/utils/polynomialFeatures";

/**
 * Coefficient bars for Lasso, optionally next to Ridge on the same features.
 * Exact zeros are labelled — that is the whole point of L1.
 */
export function LassoCoefficientChart({
  lasso,
  ridge,
  specs,
  selected,
  onSelect,
  title = "Watch the coefficients change",
}: {
  lasso: number[];
  ridge?: number[];
  specs: FeatureSpec[];
  selected?: number | null;
  onSelect?: (j: number | null) => void;
  title?: string;
}) {
  const values = [...lasso, ...(ridge ?? [])].slice(1).map(Math.abs).filter(Number.isFinite);
  const maxAbs = Math.max(...values, 1e-9);

  const bar = (value: number, tone: "lasso" | "ridge") => {
    const width = Math.min((Math.abs(value) / maxAbs) * 50, 50);
    return (
      <span className="relative flex h-2.5 flex-1 items-center">
        <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
        {value === 0 ? (
          <span className="absolute left-1/2 h-2.5 w-px bg-resid-neg" />
        ) : (
          <span
            className={`absolute h-2.5 rounded-sm ${value < 0 ? "right-1/2" : "left-1/2"} ${
              tone === "lasso" ? "bg-point" : "bg-axis/60"
            }`}
            style={{ width: `${Math.max(width, 0.4).toFixed(3)}%` }}
          />
        )}
      </span>
    );
  };

  return (
    <div className="rounded-md border border-border">
      <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
        {title}
      </h3>
      <ul className="divide-y divide-border/60">
        {lasso.map((b, j) => {
          const spec = specs[j - 1];
          const isSelected = selected === j;
          const ridgeValue = ridge?.[j];
          return (
            <li
              key={j}
              className={`px-4 py-2 ${onSelect ? "cursor-pointer" : ""} ${
                isSelected ? "bg-accent/20" : ""
              }`}
              onClick={onSelect ? () => onSelect(isSelected ? null : j) : undefined}
            >
              <div className="grid grid-cols-[6.5rem_1fr_6rem] items-center gap-3">
                <span className="font-mono text-sm">
                  {betaLabel(j)}
                  <span className="ml-1 text-xs text-muted-foreground">
                    {j === 0 ? "intercept" : (spec?.label ?? "")}
                  </span>
                </span>
                {j === 0 ? (
                  <span className="text-xs text-muted-foreground">not penalised</span>
                ) : (
                  bar(b, "lasso")
                )}
                <span className="text-right font-mono text-sm tabular-nums">
                  {j > 0 && b === 0 ? "0" : Math.abs(b) >= 1e6 ? b.toExponential(2) : fmt(b, 4)}
                </span>
              </div>
              {j > 0 && b === 0 && (
                <div className="mt-1 text-xs text-resid-neg">
                  {betaLabel(j)} = 0 · feature removed from this fitted model
                </div>
              )}
              {ridge && ridgeValue !== undefined && (
                <div className="mt-1 grid grid-cols-[6.5rem_1fr_6rem] items-center gap-3 opacity-70">
                  <span className="text-xs text-muted-foreground">Ridge</span>
                  {j === 0 ? <span /> : bar(ridgeValue, "ridge")}
                  <span className="text-right font-mono text-xs tabular-nums">
                    {Math.abs(ridgeValue) >= 1e6 ? ridgeValue.toExponential(2) : fmt(ridgeValue, 4)}
                  </span>
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
        Every bar is the actual fitted value. Raising λ shortens the penalised bars, and some can
        land exactly on zero. β₀ is never penalised.
      </p>
    </div>
  );
}
