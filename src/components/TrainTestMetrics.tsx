import { fmt } from "@/utils/calculations";

export function TrainTestMetrics({
  degree,
  trainMse,
  testMse,
  trainCount,
  testCount,
  showSplit,
}: {
  degree: number;
  trainMse: number;
  testMse: number | null;
  trainCount: number;
  testCount: number;
  showSplit: boolean;
}) {
  const items = [
    { label: "Degree", value: String(degree), sub: "highest power of x" },
    {
      label: "Training MSE",
      value: fmt(trainMse, 4),
      sub: `${trainCount} observation${trainCount === 1 ? "" : "s"} used to fit`,
    },
    showSplit
      ? {
          label: "Test MSE",
          value: testMse === null ? "—" : fmt(testMse, 4),
          sub: `${testCount} unseen observation${testCount === 1 ? "" : "s"}`,
        }
      : { label: "Test MSE", value: "—", sub: "turn on the train/test split" },
  ];
  return (
    <div className="grid grid-cols-1 divide-y divide-border rounded-md border border-border sm:grid-cols-3 sm:divide-x sm:divide-y-0">
      {items.map((it) => (
        <div key={it.label} className="px-4 py-3">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">{it.label}</div>
          <div className="mt-1 font-mono text-2xl tabular-nums">{it.value}</div>
          <div className="text-xs text-muted-foreground">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}
