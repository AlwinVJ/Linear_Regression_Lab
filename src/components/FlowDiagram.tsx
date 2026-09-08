import { ArrowRight } from "lucide-react";

export function FlowDiagram({ steps, active }: { steps: string[]; active?: boolean }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-2 text-xs">
      {steps.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            className={`rounded-full border px-3 py-1 transition-colors duration-300 ${
              active
                ? "border-axis bg-accent/20 text-foreground"
                : "border-border text-muted-foreground"
            }`}
          >
            {s}
          </span>
          {i < steps.length - 1 && <ArrowRight className="size-3 text-muted-foreground" />}
        </li>
      ))}
    </ol>
  );
}
