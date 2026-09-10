import { createFileRoute } from "@tanstack/react-router";
import LinearRegressionPage from "@/pages/LinearRegressionPage";

const title = "Linear Regression — the model and the least-squares formula";
const description =
  "The straight-line model ŷ = β₀ + β₁x explained with a live chart, every prediction and error, and the Ordinary Least Squares calculation worked out step by step.";

export const Route = createFileRoute("/linear-regression")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
    ],
  }),
  component: LinearRegressionPage,
});
