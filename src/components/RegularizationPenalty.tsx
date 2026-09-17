import { fmt } from "@/utils/calculations";
import { lambdaLabel } from "@/utils/regularization";

export function RegularizationPenalty({
  mse,
  penalty,
  objective,
  lambda,
  energy,
}: {
  mse: number;
  penalty: number;
  objective: number;
  lambda: number;
  energy: number;
}) {
  const num = (v: number) =>
    !Number.isFinite(v) ? "—" : Math.abs(v) >= 1e6 ? v.toExponential(2) : fmt(v, 4);
  return (
    <div className="rounded-md border border-border">
      <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
        Where does the extra loss come from?
      </h3>
      <dl className="px-4 py-3 font-mono text-sm">
        <div className="flex justify-between py-1">
          <dt className="text-muted-foreground">MSE (prediction loss)</dt>
          <dd className="tabular-nums">{num(mse)}</dd>
        </div>
        <div className="flex justify-between py-1">
          <dt className="text-muted-foreground">L2 penalty = λ Σβⱼ²</dt>
          <dd className="tabular-nums">{num(penalty)}</dd>
        </div>
        <div className="my-1 border-t border-border" />
        <div className="flex justify-between py-1 text-base">
          <dt>Ridge objective</dt>
          <dd className="tabular-nums">{num(objective)}</dd>
        </div>
      </dl>
      <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
        With λ = {lambdaLabel(lambda)} and Σβⱼ² = {num(energy)} (intercept excluded), the penalty is{" "}
        {lambdaLabel(lambda)} × {num(energy)} = {num(penalty)}. Ridge does not only minimise
        prediction error; it also counts the size of the coefficients.
      </p>
    </div>
  );
}
