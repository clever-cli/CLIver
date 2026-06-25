import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/admin/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;

          // Split large, independent libraries into separate chunks.
          // Benefits: finer cache granularity, smaller parallel downloads.
          //
          // React + ReactDOM must stay together — splitting them causes
          // circular dependency warnings because every UI lib imports React.

          // UI primitives — self-contained, React as peer dep (resolved from vendor)
          if (id.includes("node_modules/@radix-ui") ||
              id.includes("node_modules/@assistant-ui") ||
              id.includes("node_modules/lucide-react")) {
            return "ui";
          }

          // Data layer — Zod is pure JS, TanStack Query core is framework-agnostic
          if (id.includes("node_modules/@tanstack") ||
              id.includes("node_modules/zod")) {
            return "data";
          }

          // Everything else (React, ReactDOM, React Router, and other deps)
          return "vendor";
        },
      },
    },
    chunkSizeWarningLimit: 500,
  },
  server: {
    port: 5173,
    proxy: {
      "/admin/api": "http://localhost:8321",
      "/admin/static": "http://localhost:8321",
      "/admin/ws": { target: "ws://localhost:8321", ws: true },
    },
  },
});
