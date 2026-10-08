import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Multi-page build: the interactive human-play entry and the headless harness
// are separate HTML entry points but share the exact same engine core modules.
export default defineConfig({
  // Relative base so the built bundle can be served from any path by the
  // headless static server (and would work from file:// for simple cases).
  base: './',
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        headless: resolve(__dirname, 'headless.html'),
        record: resolve(__dirname, 'record.html'),
      },
    },
  },
  server: {
    port: 5173,
  },
});
