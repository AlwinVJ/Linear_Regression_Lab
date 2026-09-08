import { fmt } from "@/utils/calculations";
import { Term } from "@/components/Term";

export function MetricsPanel({
  slope,
  intercept,
  mse,
}: {
  slope: number;
  intercept: number;
  mse: number;
}) {
  const items = [
    { label: <Term name="parameters">Slope</Term>, sub: "β₁", value: fmt(slope, 3) },
    { label: <Term name="parameters">Intercept</Term>, sub: "β₀", value: fmt(intercept, 3) },
    { label: <Term name="loss">MSE</Term>, sub: "loss", value: fmt(mse, 3) },
  ];
  return (
    <div className="grid grid-cols-3 divide-x divide-border rounded-md border border-border">
      {items.map((it, i) => (
        <div key={i} className="px-4 py-3">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">{it.label}</div>
          <div className="mt-1 font-mono text-2xl tabular-nums transition-all">{it.value}</div>
          <div className="text-xs text-muted-foreground">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}
