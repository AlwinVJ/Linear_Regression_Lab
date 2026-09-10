import { createFileRoute } from "@tanstack/react-router";
import ConceptsPage from "@/pages/ConceptsPage";

const title = "Concepts — model, loss, gradient and gradient descent";
const description =
  "A plain-language glossary for Regression Lab: parameters, predictions, residuals, mean squared error, gradients, learning rate and convergence.";

export const Route = createFileRoute("/concepts")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
    ],
  }),
  component: ConceptsPage,
});
