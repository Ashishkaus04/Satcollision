import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards the API (and its
// WebSocket) to the FastAPI backend on :8000. `npm run build` writes dist/,
// which the backend serves itself, so production is a single process.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
      "/earth.jpg": "http://127.0.0.1:8000",
    },
  },
});
