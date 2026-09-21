import type { FeatureSpec } from "@/algorithms/featureRegression";
import { fmt } from "@/utils/calculations";
import { betaLabel } from "@/utils/polynomialFeatures";

/** Which features the fitted Lasso model still uses, and which were zeroed out. */
export function FeatureSelectionPanel({
  beta,
  specs,
}: {
  beta: number[];
  specs: FeatureSpec[];
}) {
  const active = beta.slice(1).filter((b) => b !== 0).length;
  return (
    <div className="rounded-md border border-border">
      <div className="flex items-baseline justify-between border-b border-border px-4 py-2">
        <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
          Features selected by Lasso
        </h3>
        <span className="font-mono text-sm tabular-nums">
          Active features: {active} / {specs.length}
        </span>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-widest text-muted-foreground">
            <th className="px-4 py-2 font-normal">Feature</th>
            <th className="px-4 py-2 text-right font-normal">Coefficient</th>
            <th className="px-4 py-2 text-right font-normal">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/60">
          {specs.map((s, i) => {
            const b = beta[i + 1] ?? 0;
            const removed = b === 0;
            return (
              <tr key={s.key}>
                <td className="px-4 py-2 font-mono">
                  {s.label}
                  <span className="ml-2 text-xs text-muted-foreground">
                    {betaLabel(i + 1)} · {s.description}
                  </span>
                </td>
                <td className="px-4 py-2 text-right font-mono tabular-nums">
                  {removed ? "0" : fmt(b, 4)}
                </td>
                <td
                  className={`px-4 py-2 text-right ${removed ? "text-resid-neg" : "text-foreground"}`}
                >
                  {removed ? "Removed" : "Selected"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
        A feature with a coefficient of exactly zero contributes nothing to this model's prediction.
        That is a statement about this fit — this dataset, this scaling, this λ — not proof that the
        feature is unimportant in the real world.
      </p>
    </div>
  );
}
