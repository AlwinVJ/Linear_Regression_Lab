import { fmt } from "@/utils/calculations";
import { betaLabel, powerLabel } from "@/utils/polynomialFeatures";

/**
 * Coefficient shrinkage. Bars are drawn from real coefficient values; when OLS
 * values are supplied the two models are shown side by side.
 */
export function CoefficientChart({
  ridge,
  ols,
  selected,
  onSelect,
}: {
  ridge: number[];
  ols?: number[];
  selected?: number | null;
  onSelect?: (j: number | null) => void;
}) {
  const all = [...ridge, ...(ols ?? [])].slice(1).map(Math.abs).filter(Number.isFinite);
  const maxAbs = Math.max(...all, 1e-9);

  const bar = (value: number, tone: "ridge" | "ols") => {
    const width = Math.min((Math.abs(value) / maxAbs) * 50, 50);
    return (
      <span className="relative flex h-2.5 flex-1 items-center">
        <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
        <span
          className={`absolute h-2.5 rounded-sm ${
            value < 0 ? "right-1/2" : "left-1/2"
          } ${tone === "ridge" ? "bg-point" : "bg-axis/60"}`}
          style={{ width: `${Math.max(width, 0.4)}%` }}
        />
      </span>
    );
  };

  return (
    <div className="rounded-md border border-border">
      <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
        Coefficient shrinkage
      </h3>
      <ul className="divide-y divide-border/60">
        {ridge.map((b, j) => {
          const olsValue = ols?.[j];
          const isSelected = selected === j;
          return (
            <li
              key={j}
              className={`px-4 py-2 ${onSelect ? "cursor-pointer" : ""} ${
                isSelected ? "bg-accent/20" : ""
              }`}
              onClick={onSelect ? () => onSelect(isSelected ? null : j) : undefined}
            >
              <div className="grid grid-cols-[5.5rem_1fr_6rem] items-center gap-3">
                <span className="font-mono text-sm">
                  {betaLabel(j)}
                  <span className="ml-1 text-xs text-muted-foreground">
                    {j === 0 ? "intercept" : powerLabel(j)}
                  </span>
                </span>
                {j === 0 ? (
                  <span className="text-xs text-muted-foreground">not penalised</span>
                ) : (
                  bar(b, "ridge")
                )}
                <span className="text-right font-mono text-sm tabular-nums">
                  {Math.abs(b) >= 1e6 ? b.toExponential(2) : fmt(b, 4)}
                </span>
              </div>
              {ols && olsValue !== undefined && (
                <div className="mt-1 grid grid-cols-[5.5rem_1fr_6rem] items-center gap-3 opacity-70">
                  <span className="text-xs text-muted-foreground">OLS</span>
                  {j === 0 ? <span /> : bar(olsValue, "ols")}
                  <span className="text-right font-mono text-xs tabular-nums">
                    {Math.abs(olsValue) >= 1e6 ? olsValue.toExponential(2) : fmt(olsValue, 4)}
                  </span>
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
        Bars use the actual fitted values. Raising λ makes the penalised bars shorter — but usually
        not exactly zero. The intercept β₀ is never penalised.
      </p>
    </div>
  );
}
