import { createFileRoute } from "@tanstack/react-router";
import RegressionLab from "@/pages/RegressionLab";

const title = "Regression Lab — See how Linear Regression learns from data";
const description =
  "An interactive, browser-only lab for absolute beginners: edit data, move the line, and watch predictions, residuals and mean squared error update live.";

export const Route = createFileRoute("/playground")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
    ],
  }),
  component: RegressionLab,
});
