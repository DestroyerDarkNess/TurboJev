let runtime, engine;
self.onmessage = async ({ data: { id, op, args } }) => {
  try {
    let value;
    if (op === 'load') {
      const manifestResponse = await fetch('./vendor/manifest.json', { cache: 'no-cache' });
      if (!manifestResponse.ok) throw new Error('Demo assets are missing. Run the website build.');
      const manifest = await manifestResponse.json();
      const runtimeModule = args.backend === 'llamacpp'
        ? await import('https://cdn.jsdelivr.net/npm/@turbojev/runtime-llamacpp-wasm@0.29.0/+esm')
        : await import(`./vendor/studio-runtime.js?v=${manifest.runtime}`);
      const { createRuntime } = runtimeModule;
      runtime = await createRuntime();
      const descriptor = await runtime.initialize({ ...args, progressCallback: (progress) => postMessage({ progress }) });
      const wasm = await import(`./vendor/wasm/turbojev_wasm.js?v=${manifest.wasm}`);
      await wasm.default({ module_or_path: new URL(`./vendor/wasm/turbojev_wasm_bg.wasm?v=${manifest.wasm}`, import.meta.url) });
      engine = new wasm.WasmEngine(runtime, JSON.stringify(descriptor), undefined, args.cyclicOrderRobustness === true);
      value = descriptor;
    } else if (!engine) throw new Error('Load a model first.');
    else if (op === 'evaluate') {
      runtime.setMedia?.(args.media);
      try { value = JSON.parse(await engine.evaluate_json(JSON.stringify(args.request))); }
      finally { runtime.setMedia?.(null); }
    } else throw new Error('Unknown operation');
    postMessage({ id, value });
  } catch (error) { postMessage({ id, error: error?.message || String(error) }); }
};
