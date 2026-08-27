import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:18789',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Split slow-changing vendor code into its own long-cached chunks so
        // the app chunk stays small and a code change doesn't re-download React
        // or the terminal emulator.
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-xterm': ['@xterm/xterm', '@xterm/addon-fit'],
        },
      },
    },
  },
});
