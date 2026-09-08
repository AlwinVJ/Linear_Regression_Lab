import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { fmt } from "@/utils/calculations";

interface Props {
  manual: boolean;
  onManualChange: (manual: boolean) => void;
  slope: number;
  intercept: number;
  onSlope: (v: number) => void;
  onIntercept: (v: number) => void;
  onFitBest: () => void;
  showResiduals: boolean;
  onShowResiduals: (v: boolean) => void;
}

export function ModelControls({
  manual,
  onManualChange,
  slope,
  intercept,
  onSlope,
  onIntercept,
  onFitBest,
  showResiduals,
  onShowResiduals,
}: Props) {
  return (
    <div className="space-y-4">
      <div className="flex rounded-md border border-border p-0.5 text-sm">
        <button
          onClick={() => onManualChange(false)}
          className={`flex-1 rounded-sm px-2 py-1.5 transition-colors ${!manual ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}
        >
          Automatic fit
        </button>
        <button
          onClick={() => onManualChange(true)}
          className={`flex-1 rounded-sm px-2 py-1.5 transition-colors ${manual ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}
        >
          Manual mode
        </button>
      </div>

      <div className={manual ? "space-y-4" : "space-y-4 opacity-50"}>
        <div>
          <div className="flex items-baseline justify-between text-sm">
            <label htmlFor="slope">Slope β₁</label>
            <span className="font-mono tabular-nums">{fmt(slope, 3)}</span>
          </div>
          <Slider
            id="slope"
            disabled={!manual}
            min={-5}
            max={5}
            step={0.01}
            value={[slope]}
            onValueChange={(v) => onSlope(v[0] ?? 0)}
            className="mt-2"
          />
        </div>
        <div>
          <div className="flex items-baseline justify-between text-sm">
            <label htmlFor="intercept">Intercept β₀</label>
            <span className="font-mono tabular-nums">{fmt(intercept, 3)}</span>
          </div>
          <Slider
            id="intercept"
            disabled={!manual}
            min={-20}
            max={20}
            step={0.01}
            value={[intercept]}
            onValueChange={(v) => onIntercept(v[0] ?? 0)}
            className="mt-2"
          />
        </div>
      </div>

      <Button variant="outline" size="sm" className="w-full" onClick={onFitBest}>
        <Sparkles className="size-3.5" /> Fit best line
      </Button>

      <div className="flex items-center justify-between border-t border-border pt-4 text-sm">
        <label htmlFor="residuals">Show residuals</label>
        <Switch id="residuals" checked={showResiduals} onCheckedChange={onShowResiduals} />
      </div>
    </div>
  );
}
