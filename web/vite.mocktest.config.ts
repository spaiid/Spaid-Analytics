// TEMPORARY: local smoke-test config (mock API). Deleted after the run.
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5199,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8123', changeOrigin: true } },
  },
});
