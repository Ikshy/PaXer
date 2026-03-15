import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI backend during development
      "/ingest": "http://localhost:8000",
      "/infer": "http://localhost:8000",
      "/queue": "http://localhost:8000",
      "/verify": "http://localhost:8000",
      "/audit": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
    },
  },
});
