export default {
  // These compiled modules are copied from public/ and loaded by the worker.
  build: { rollupOptions: { external: [/^\/vendor\//] } },
};
