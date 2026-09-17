import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { LAMBDA_STEPS, lambdaLabel, nearestLambdaIndex } from "@/utils/regularization";

const QUICK = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000];

export function RegularizationControl({
  lambda,
  onChange,
}: {
  lambda: number;
  onChange: (l: number) => void;
}) {
  const index = nearestLambdaIndex(lambda);
  return (
    <div className="rounded-md border border-border p-4">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs uppercase tracking-widest text-muted-foreground">
          Regularization strength λ
        </span>
        <span className="font-mono text-3xl tabular-nums">{lambdaLabel(lambda)}</span>
      </div>
      <Slider
        className="mt-4"
        min={0}
        max={LAMBDA_STEPS.length - 1}
        step={1}
        value={[index]}
        onValueChange={([i]) => onChange(LAMBDA_STEPS[i ?? 0] ?? 0)}
        aria-label="Regularization strength lambda"
      />
      <div className="mt-2 flex justify-between text-xs text-muted-foreground">
        <span>No regularization</span>
        <span>Very strong regularization</span>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {QUICK.map((l) => (
          <Button
            key={l}
            size="sm"
            variant={Math.abs(l - lambda) < 1e-12 ? "default" : "outline"}
            onClick={() => onChange(l)}
          >
            λ = {lambdaLabel(l)}
          </Button>
        ))}
      </div>
      <p className="mt-3 text-sm text-muted-foreground">
        {lambda === 0
          ? "λ is zero, so there is no penalty term at all: Ridge is exactly ordinary least squares here."
          : "Every coefficient except the intercept now costs the model something. Larger λ means a larger cost."}
      </p>
    </div>
  );
}
