import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: '/Users/jspaid/Spaid-Analytics/web',
  plugins: [react()],
  server: {
    port: 5311,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8447', changeOrigin: true } },
  },
});
