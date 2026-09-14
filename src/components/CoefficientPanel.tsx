import { betaLabel, powerLabel } from "@/utils/polynomialFeatures";
import { fmt } from "@/utils/calculations";

export function CoefficientPanel({ coefficients }: { coefficients: number[] }) {
  const maxAbs = Math.max(...coefficients.map((b) => Math.abs(b)), 1e-9);
  return (
    <div className="rounded-md border border-border">
      <h3 className="border-b border-border px-4 py-2 text-xs uppercase tracking-widest text-muted-foreground">
        Model coefficients
      </h3>
      <ul className="divide-y divide-border/60">
        {coefficients.map((b, j) => {
          const width = (Math.abs(b) / maxAbs) * 100;
          return (
            <li key={j} className="grid grid-cols-[4rem_1fr_7rem] items-center gap-3 px-4 py-2">
              <span className="font-mono text-sm">
                {betaLabel(j)}
                <span className="ml-1 text-xs text-muted-foreground">
                  {j === 0 ? "const" : powerLabel(j)}
                </span>
              </span>
              <span className="relative flex h-2 items-center">
                <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
                <span
                  className={`absolute h-2 rounded-sm ${b < 0 ? "bg-resid-neg right-1/2" : "bg-point left-1/2"}`}
                  style={{ width: `${Math.max(width / 2, 0.5)}%` }}
                />
              </span>
              <span className="text-right font-mono text-sm tabular-nums">
                {Math.abs(b) >= 1e6 ? b.toExponential(2) : fmt(b, 4)}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
        Each coefficient says how strongly its feature contributes to the prediction. Size alone
        does not mean importance: x, x² and x³ live on very different scales and move together.
      </p>
    </div>
  );
}
