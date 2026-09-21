// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import fs from "node:fs";
import path from "node:path";

function prepareGitHubPagesOutput() {
  const root = process.cwd();
  const outputPublicDir = path.join(root, ".output", "public");

  if (!fs.existsSync(outputPublicDir)) return;

  const indexHtml = path.join(outputPublicDir, "index.html");
  const notFoundHtml = path.join(outputPublicDir, "404.html");
  if (fs.existsSync(indexHtml)) {
    fs.copyFileSync(indexHtml, notFoundHtml);
  }

  const noJekyll = path.join(outputPublicDir, ".nojekyll");
  fs.writeFileSync(noJekyll, "");
}

function githubPagesPlugin() {
  return {
    name: "github-pages-spa",
    apply: "build" as const,
    enforce: "post" as const,
    buildApp: {
      order: "post" as const,
      async handler() {
        prepareGitHubPagesOutput();
      },
    },
    closeBundle: {
      order: "post" as const,
      async handler() {
        prepareGitHubPagesOutput();
      },
    },
  };
}

export default defineConfig({
  vite: {
    base: "/Linear_Regression_Lab/",
    plugins: [githubPagesPlugin()],
  },

  tanstackStart: {
    server: {
      entry: "server",
    },

    spa: {
      enabled: true,

      prerender: {
        outputPath: "/index.html",
      },
    },
  },
});
