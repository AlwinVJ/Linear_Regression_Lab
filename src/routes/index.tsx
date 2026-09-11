import { createFileRoute } from "@tanstack/react-router";
import HomePage from "@/pages/HomePage";

const title = "Regression Lab — See how regression learns from data";
const description =
  "An interactive visualization tool that helps beginners understand what happens inside regression models—from data and predictions to errors, loss, and optimization.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
    ],
  }),
  component: HomePage,
});
