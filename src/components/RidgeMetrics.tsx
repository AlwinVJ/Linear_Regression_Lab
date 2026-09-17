import { fmt } from "@/utils/calculations";
import { lambdaLabel } from "@/utils/regularization";

export function RidgeMetrics({
  lambda,
  trainMse,
  testMse,
  objective,
  energy,
  showSplit,
}: {
  lambda: number;
  trainMse: number;
  testMse: number | null;
  objective: number;
  energy: number;
  showSplit: boolean;
}) {
  const num = (v: number | null) =>
    v === null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 1e6 ? v.toExponential(2) : fmt(v, 4);
  const items = [
    { label: "λ", value: lambdaLabel(lambda), sub: "regularization strength" },
    { label: "Training MSE", value: num(trainMse), sub: "how wrong the predictions are" },
    {
      label: "Test MSE",
      value: showSplit ? num(testMse) : "—",
      sub: showSplit ? "unseen observations" : "turn on the train/test split",
    },
    { label: "Ridge objective", value: num(objective), sub: "MSE + λ Σβⱼ²" },
    { label: "Σβⱼ² ", value: num(energy), sub: "coefficient magnitude (no β₀)" },
  ];
  return (
    <div className="grid grid-cols-1 divide-y divide-border rounded-md border border-border sm:grid-cols-5 sm:divide-x sm:divide-y-0">
      {items.map((it) => (
        <div key={it.label} className="px-4 py-3">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">{it.label}</div>
          <div className="mt-1 font-mono text-xl tabular-nums">{it.value}</div>
          <div className="text-xs text-muted-foreground">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}
