import { Link } from "@tanstack/react-router";

const items = [
  { to: "/", label: "Home" },
  { to: "/playground", label: "Playground" },
  { to: "/linear-regression", label: "Linear Regression" },
  { to: "/gradient-descent", label: "Gradient Descent" },
  { to: "/polynomial-regression", label: "Polynomial Regression" },
  { to: "/concepts", label: "Concepts" },
] as const;

export function SiteNav() {
  return (
    <nav className="flex flex-wrap items-center gap-1 text-sm" aria-label="Main">
      {items.map((it) => (
        <Link
          key={it.to}
          to={it.to}
          activeOptions={{ exact: true }}
          className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground data-[status=active]:bg-accent/20 data-[status=active]:text-foreground"
        >
          {it.label}
        </Link>
      ))}
    </nav>
  );
}
