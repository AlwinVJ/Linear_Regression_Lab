import type { RowBreakdown } from "@/algorithms/linearRegression";
import { fmt } from "@/utils/calculations";

export function ResidualTable({
  rows,
  mse,
  selectedId,
  onSelect,
}: {
  rows: RowBreakdown[];
  mse: number;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted text-left">
          <tr>
            <th className="px-3 py-2 font-medium">x</th>
            <th className="px-3 py-2 font-medium">Actual y</th>
            <th className="px-3 py-2 font-medium">Prediction ŷ</th>
            <th className="px-3 py-2 font-medium">Error (y − ŷ)</th>
            <th className="px-3 py-2 font-medium">Error²</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((r) => (
            <tr
              key={r.id}
              onMouseEnter={() => onSelect(r.id)}
              className={`border-t border-border ${selectedId === r.id ? "bg-accent/15" : ""}`}
            >
              <td className="px-3 py-1.5">{fmt(r.x)}</td>
              <td className="px-3 py-1.5">{fmt(r.actual)}</td>
              <td className="px-3 py-1.5">{fmt(r.prediction)}</td>
              <td className="px-3 py-1.5">
                {r.error >= 0 ? "+" : "−"}
                {fmt(Math.abs(r.error))}
                <span className="ml-1 text-xs text-muted-foreground">
                  {r.error >= 0 ? "(above line)" : "(below line)"}
                </span>
              </td>
              <td className="px-3 py-1.5">{fmt(r.squaredError)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-axis bg-muted/60">
            <td colSpan={4} className="px-3 py-2 text-right font-medium">
              MSE = average of the Error² column
            </td>
            <td className="px-3 py-2 font-mono text-base tabular-nums">{fmt(mse, 3)}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
