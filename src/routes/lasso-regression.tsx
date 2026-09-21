import { createFileRoute } from "@tanstack/react-router";
import LassoRegressionLab from "@/pages/LassoRegression";

const title = "Lasso Regression — watch coefficients hit exactly zero";
const description =
  "See how the L1 penalty works: raise λ and watch real Lasso coefficients shrink, some reach exactly zero, and features drop out — solved with coordinate descent and soft thresholding.";

export const Route = createFileRoute("/lasso-regression")({
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
  component: LassoRegressionLab,
});
