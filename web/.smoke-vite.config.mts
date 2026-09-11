import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: '/Users/jspaid/Spaid-Analytics/web',
  plugins: [react()],
  build: {
    outDir: '/private/tmp/claude-502/-Users-jspaid-Spaid-Analytics/c3ac357a-6777-44bd-952b-9eaf65bc93ed/scratchpad/dist-smoke',
    emptyOutDir: true,
    rollupOptions: { input: '/Users/jspaid/Spaid-Analytics/web/.smoke-entry.tsx' },
  },
});
