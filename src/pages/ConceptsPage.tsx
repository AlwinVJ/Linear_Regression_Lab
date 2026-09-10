import { SiteNav } from "@/components/SiteNav";
import { GLOSSARY } from "@/components/Term";
import { FlowDiagram } from "@/components/FlowDiagram";
import { TooltipProvider } from "@/components/ui/tooltip";

const EXTRA = [
  {
    label: "Gradient",
    text: "How the loss changes when a parameter changes. It points in the direction that increases the loss.",
  },
  {
    label: "Gradient descent",
    text: "An optimisation algorithm: repeatedly subtract a small multiple of the gradient from each parameter.",
  },
  {
    label: "Learning rate (α)",
    text: "How big each update is. Too small and learning crawls; too large and it overshoots the minimum.",
  },
  {
    label: "Iteration",
    text: "One complete pass: predict, measure error, compute gradients, update parameters.",
  },
  {
    label: "Convergence",
    text: "The point where extra iterations barely change the loss or the parameters.",
  },
  {
    label: "Ordinary Least Squares",
    text: "A direct formula that gives the least-squares line in one shot, without iterating.",
  },
];

export default function ConceptsPage() {
  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
            <div>
              <h1 className="font-display text-4xl tracking-tight">Concepts</h1>
              <p className="mt-1 text-muted-foreground">
                The words used across the lab, kept carefully apart.
              </p>
            </div>
            <SiteNav />
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-10 px-5 py-10">
          <section className="rounded-md border border-border p-5">
            <h2 className="text-xs uppercase tracking-widest text-muted-foreground">
              The whole story in one row
            </h2>
            <div className="mt-3">
              <FlowDiagram
                active
                steps={[
                  "Data",
                  "Model",
                  "Prediction",
                  "Residual",
                  "Loss",
                  "Gradient",
                  "Update",
                  "Better model",
                ]}
              />
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            {[...Object.values(GLOSSARY), ...EXTRA].map((entry) => (
              <div key={entry.label} className="rounded-md border border-border p-4">
                <h3 className="font-medium">{entry.label}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{entry.text}</p>
              </div>
            ))}
          </section>

          <section className="rounded-md border border-dashed border-axis/50 bg-muted/40 p-5 text-sm text-muted-foreground">
            <p>
              One distinction is worth repeating: linear regression is the{" "}
              <strong className="text-foreground">model</strong>, mean squared error is the{" "}
              <strong className="text-foreground">loss</strong>, and gradient descent is the{" "}
              <strong className="text-foreground">algorithm</strong> that searches for good
              parameters. They are three different things that often appear together.
            </p>
          </section>
        </main>
      </div>
    </TooltipProvider>
  );
}
