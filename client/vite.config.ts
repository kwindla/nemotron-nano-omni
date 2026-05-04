import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

const botUrl = process.env.PIPECAT_BOT_URL ?? "http://127.0.0.1:7860";
const allowedHosts = Array.from(
  new Set([
    ".ngrok.app",
    ...(process.env.VITE_ALLOWED_HOSTS ?? "")
      .split(",")
      .map((host) => host.trim())
      .filter(Boolean),
  ]),
);

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    allowedHosts,
    proxy: {
      "/api": {
        target: botUrl,
        changeOrigin: true,
      },
      "/sessions": {
        target: botUrl,
        changeOrigin: true,
      },
      "/start": {
        target: botUrl,
        changeOrigin: true,
      },
    },
  },
});
