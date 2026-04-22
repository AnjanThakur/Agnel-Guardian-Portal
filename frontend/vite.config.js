import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/ocr': 'http://localhost:8000',
      '/analysis': 'http://localhost:8000',
      '/analytics': 'http://localhost:8000'
    }
  },
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
  }
})
