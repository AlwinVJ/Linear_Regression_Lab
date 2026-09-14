import { createFileRoute } from "@tanstack/react-router";
import PolynomialRegressionLab from "@/pages/PolynomialRegression";

const title = "Polynomial Regression — curves from a linear model";
const description =
  "Expand x into x, x², x³ … and fit a linear model to those features. Watch the curve, coefficients, training error and test error change with the polynomial degree.";

export const Route = createFileRoute("/polynomial-regression")({
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
  component: PolynomialRegressionLab,
});
