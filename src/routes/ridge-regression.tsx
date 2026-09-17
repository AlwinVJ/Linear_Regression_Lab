import { createFileRoute } from "@tanstack/react-router";
import RidgeRegressionLab from "@/pages/RidgeRegression";

const title = "Ridge Regression — watch coefficients shrink as λ grows";
const description =
  "See why regularization is needed: move λ from zero upward and watch real Ridge coefficients, penalty, objective, curve, training error and test error respond.";

export const Route = createFileRoute("/ridge-regression")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: RidgeRegressionLab,
});
