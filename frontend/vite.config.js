import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: {
    port: 5173,
    proxy: {
      '/upload': 'http://localhost:8000',
      '/transcribe': 'http://localhost:8000',
      '/separate': 'http://localhost:8000',
      '/segment': 'http://localhost:8000',
      '/transform': 'http://localhost:8000',
      '/previews': 'http://localhost:8000',
      '/score': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/cleanup': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/session-data': 'http://localhost:8000',
      '/progress-data': 'http://localhost:8000',
      '/voice-data': 'http://localhost:8000',
    }
  }
})
