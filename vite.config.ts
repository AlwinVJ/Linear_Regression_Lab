// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import fs from "node:fs";
import path from "node:path";

function prepareDist() {
  const root = process.cwd();
  const outputPublicDir = path.join(root, ".output", "public");
  const distDir = path.join(root, "dist");

  if (!fs.existsSync(outputPublicDir)) return;

  fs.mkdirSync(distDir, { recursive: true });

  const copyRecursive = (src: string, dest: string) => {
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      if (entry.isDirectory()) {
        fs.mkdirSync(destPath, { recursive: true });
        copyRecursive(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  };

  copyRecursive(outputPublicDir, distDir);

  const serverDir = path.join(distDir, "server");
  if (fs.existsSync(serverDir)) {
    fs.rmSync(serverDir, { recursive: true, force: true });
  }

  const indexHtml = path.join(distDir, "index.html");
  const notFoundHtml = path.join(distDir, "404.html");
  if (fs.existsSync(indexHtml)) {
    fs.copyFileSync(indexHtml, notFoundHtml);
  }

  const noJekyll = path.join(distDir, ".nojekyll");
  fs.writeFileSync(noJekyll, "");
}

function githubPagesPlugin() {
  return {
    name: "github-pages-dist",
    apply: "build" as const,
    enforce: "post" as const,
    buildApp: {
      order: "post" as const,
      async handler() {
        prepareDist();
      },
    },
    closeBundle: {
      order: "post" as const,
      async handler() {
        prepareDist();
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
