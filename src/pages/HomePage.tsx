import { Link } from "@tanstack/react-router";

import { FlowDiagram } from "@/components/FlowDiagram";
import { SiteNav } from "@/components/SiteNav";
import { Button } from "@/components/ui/button";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-5 py-8">
          <div>
            <h1 className="font-display text-4xl tracking-tight">Regression Lab</h1>
            <p className="mt-1 text-muted-foreground">See how regression learns from data.</p>
          </div>
          <SiteNav />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-16">
        <section className="max-w-2xl">
          <h2 className="font-display text-3xl tracking-tight">Regression Lab</h2>
          <p className="mt-4 text-lg text-muted-foreground">
            An interactive visualization tool that helps beginners understand what happens inside
            regression models—from data and predictions to errors, loss, and optimization.
          </p>
          <div className="mt-8">
            <Button asChild size="lg">
              <Link to="/playground">Start Learning</Link>
            </Button>
          </div>
        </section>

        <section className="mt-16">
          <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Learning flow</h3>
          <div className="mt-4">
            <FlowDiagram
              active
              steps={["Data", "Model", "Predictions", "Errors", "Loss", "Learning"]}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
