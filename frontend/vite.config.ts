import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(process.cwd(), "src"),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        // AGENTSEG_API points a dev server at another backend (e.g. a private one for UI work).
        target: process.env.AGENTSEG_API ?? 'http://127.0.0.1:5172',
        changeOrigin: true,
      },
    },
  },
})
