import { createFileRoute } from "@tanstack/react-router";
import GradientDescentLab from "@/pages/GradientDescentLab";

const title = "Gradient Descent — watch a regression line learn its parameters";
const description =
  "Step through real gradient descent iterations: gradients, learning rate, loss curve and parameter trajectory, all calculated live in your browser.";

export const Route = createFileRoute("/gradient-descent")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
    ],
  }),
  component: GradientDescentLab,
});
