import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

const botUrl = process.env.PIPECAT_BOT_URL ?? "http://127.0.0.1:7860";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    allowedHosts: ["gb-client-khk.ngrok.app"],
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
