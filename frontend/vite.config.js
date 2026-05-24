import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config. During local dev we proxy /api -> the FastAPI backend so the
// frontend can call the API without CORS friction. In production the API base
// URL comes from VITE_API_URL (see src/api.js).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Allow public tunnel hosts (ngrok / Cloudflare) to reach the dev server.
    // Without this, Vite rejects the tunnel's Host header ("Blocked request").
    // Leading dots match all subdomains. Safe to remove after the demo.
    allowedHosts: [".ngrok-free.app", ".ngrok.io", ".ngrok.app", ".trycloudflare.com"],
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
