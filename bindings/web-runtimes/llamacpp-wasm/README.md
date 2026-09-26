# TurboJev llama.cpp WASM runtime

Browser decision adapter for GGUF text models supported by llama.cpp, using the
Wllama WebAssembly build. It implements TurboJev's `BrowserDecisionRuntime`
contract and returns one score per candidate. The Rust/WASM engine validates
the scores and creates the shared `EvaluationResponse`.

The adapter sends the model's chat-formatted request to llama.cpp, constrains
the next answer to `A`–`T` with grammar, and maps returned log-probabilities
back to the original candidate order. It does not contain model-specific token
IDs. One inference call is made per decision task.

```js
import { TurboJevWeb } from '@turbojev/web';

const engine = await TurboJevWeb.load({
  model: 'Qwen3 0.6B Q4_K_M',
  runtimeModuleUrl: '/llamacpp-runtime.js',
  wasmModuleUrl: '/turbojev_wasm.js',
  runtimeOptions: {
    modelUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q4_K_M.gguf',
    device: 'wasm', // use 'webgpu' to request GPU execution
  },
});

const result = await engine.classify('My package arrived broken', [
  'billing', 'shipping', 'technical', 'other',
]);
await engine.close();
```

The host must serve this adapter, TurboJev's generated WASM files, and the page
over localhost or HTTPS. Model files are fetched from their URL and remain in
the browser. Wllama 3.6.1 and its WASM binary are loaded from jsDelivr.
