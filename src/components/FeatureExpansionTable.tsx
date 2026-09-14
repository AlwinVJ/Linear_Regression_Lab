import type { Point } from "@/algorithms/linearRegression";
import { expandVisibleRow, powerLabel } from "@/utils/polynomialFeatures";
import { fmt } from "@/utils/calculations";

export function FeatureExpansionTable({
  points,
  degree,
  limit = 8,
}: {
  points: Point[];
  degree: number;
  limit?: number;
}) {
  const rows = points.slice(0, limit);
  const cols = Array.from({ length: degree }, (_, i) => i + 1);
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full min-w-[420px] text-sm">
        <caption className="sr-only">Polynomial features generated from each input x</caption>
        <thead>
          <tr className="border-b border-border bg-muted/40 text-xs uppercase tracking-widest text-muted-foreground">
            <th scope="col" className="px-3 py-2 text-left">
              y
            </th>
            {cols.map((j) => (
              <th key={j} scope="col" className="px-3 py-2 text-right font-mono normal-case">
                {powerLabel(j)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((p) => {
            const features = expandVisibleRow(p.x, degree);
            return (
              <tr key={p.id} className="border-b border-border/60 last:border-0">
                <td className="px-3 py-1.5 text-left">{fmt(p.y)}</td>
                {features.map((v, i) => (
                  <td key={i} className="px-3 py-1.5 text-right">
                    {Math.abs(v) >= 1e6 ? v.toExponential(2) : fmt(v)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {points.length > limit && (
        <p className="px-3 py-2 text-xs text-muted-foreground">
          Showing the first {limit} of {points.length} rows.
        </p>
      )}
    </div>
  );
}
