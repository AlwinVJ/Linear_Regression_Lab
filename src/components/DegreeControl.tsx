import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

const QUICK = [1, 2, 3, 5, 10];

export function DegreeControl({
  degree,
  onChange,
  max = 10,
}: {
  degree: number;
  onChange: (d: number) => void;
  max?: number;
}) {
  return (
    <div className="rounded-md border border-border p-4">
      <div className="flex items-baseline justify-between">
        <label htmlFor="degree" className="text-xs uppercase tracking-widest text-muted-foreground">
          Polynomial degree
        </label>
        <span className="font-mono text-3xl tabular-nums">{degree}</span>
      </div>
      <Slider
        id="degree"
        className="mt-4"
        min={1}
        max={max}
        step={1}
        value={[degree]}
        onValueChange={([v]) => onChange(v ?? 1)}
      />
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Try:</span>
        {QUICK.filter((d) => d <= max).map((d) => (
          <Button
            key={d}
            size="sm"
            variant={degree === d ? "default" : "outline"}
            onClick={() => onChange(d)}
          >
            Degree {d}
          </Button>
        ))}
      </div>
      <p className="mt-3 text-sm text-muted-foreground">
        The degree is the highest power of x the model is allowed to use. Changing it regenerates
        the features, refits the model and recalculates every number on this page.
      </p>
    </div>
  );
}
