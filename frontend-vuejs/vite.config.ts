import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueJsx(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: {
      usePolling: true,
      interval: 1000,
    },
    hmr: {
      port: 5173,
      host: '0.0.0.0',
    }
  },
  optimizeDeps: {
    exclude: ['@rollup/rollup-linux-arm64-musl', '@rollup/rollup-linux-arm64-gnu']
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      external: ['@rollup/rollup-linux-arm64-musl', '@rollup/rollup-linux-arm64-gnu']
    }
  }
})
