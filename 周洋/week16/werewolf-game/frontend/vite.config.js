import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/games': 'http://localhost:8000',
      '/configs': 'http://localhost:8000',
    },
  },
})
