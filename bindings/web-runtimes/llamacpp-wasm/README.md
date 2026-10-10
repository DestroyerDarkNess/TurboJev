# TurboJev llama.cpp WASM runtime

Browser decision adapter for compatible llama.cpp GGUF models, using Wllama's
WebAssembly build. It implements TurboJev's `BrowserDecisionRuntime` contract
and returns one score per candidate. The Rust/WASM engine validates the scores
and creates the shared `EvaluationResponse`.

The adapter sends the model's chat-formatted request to llama.cpp, constrains
the next answer to `A`–`Z` with grammar, and maps returned log-probabilities
back to the original candidate order. It does not contain model-specific token
IDs. One inference call is made per decision task unless a permutation budget or order correction
is requested. Wllama 3.8.1 supplies
llama.cpp's WebAssembly multimodal path: set `mmprojUrl` to the projector paired
with the GGUF to enable supported image input. Video is passed as ordered
timestamped image frames; it does not guarantee temporal reasoning. When Wllama
reports an audio encoder, the adapter forwards audio bytes before the prompt
text. Audio needs cross-origin isolation and a prestarted worker pool for mtmd
preprocessing. The adapter reserves those workers independently of `nThreads`;
the requested inference thread count is preserved.

```js
import { TurboJevWeb } from '@turbojev/web';

const engine = await TurboJevWeb.load({
  model: 'Qwen3 0.6B Q4_K_M',
  runtimeModuleUrl: '/llamacpp/index.js',
  wasmModuleUrl: '/turbojev_wasm.js',
  runtimeOptions: {
    modelUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q4_K_M.gguf',
    device: 'wasm', // use 'webgpu' to request GPU execution
  },
});

const result = await engine.evaluate({
  state: { message: 'My package arrived broken' },
  questions: { department: {
    type: 'choice',
    instructions: 'Which department should handle this message?',
    criteria: { billing: 'Billing', shipping: 'Shipping', technical: 'Technical', other: 'Other' },
  } },
});
await engine.close();
```

For image input, provide the matching projector and declare the modality when
loading:

```js
const engine = await TurboJevWeb.load({
  model: 'LFM2-VL 450M Q4_0',
  runtimeModuleUrl: '/llamacpp/index.js',
  wasmModuleUrl: '/turbojev_wasm.js',
  runtimeOptions: {
    modelUrl: 'https://huggingface.co/runanywhere/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf',
    mmprojUrl: 'https://huggingface.co/runanywhere/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf',
    modality: 'image',
    device: 'wasm',
  },
});

import { mediaEvidence } from '@turbojev/web';

const evidence = await mediaEvidence(imageFile, 'image', 'Read the traffic sign.');
const result = await engine.evaluate({
  state: evidence,
  questions: { sign: {
    type: 'choice',
    instructions: 'Which sign is visible?',
    criteria: { stop: 'A stop sign', yield: 'A yield sign' },
  } },
});
```

For spoken-command decisions, load the matching LFM2 Audio GGUF and projector:

```js
const engine = await TurboJevWeb.load({
  model: 'LFM2-Audio-1.5B-Q8_0',
  runtimeModuleUrl: '/llamacpp/index.js',
  wasmModuleUrl: '/turbojev_wasm.js',
  cyclicOrderRobustness: true, // Optional: one evaluation per choice; false is one pass.
  runtimeOptions: {
    modelUrl: 'https://huggingface.co/ggml-org/LFM2-Audio-1.5B-GGUF/resolve/main/LFM2-Audio-1.5B-Q8_0.gguf',
    mmprojUrl: 'https://huggingface.co/ggml-org/LFM2-Audio-1.5B-GGUF/resolve/main/mmproj-LFM2-Audio-1.5B-Q8_0.gguf',
    modality: 'audio', device: 'wasm', nThreads: 4, jinja: true, warmup: false,
  },
});
const state = await mediaEvidence(audioFile, 'audio');
const result = await engine.evaluate({ state, questions: { command: {
  type: 'choice', instructions: 'Which command does the speaker say?',
  criteria: { on: 'Turn on the light', off: 'Turn off the light' },
} } });
await engine.close();
```

Serve with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. Audio fails before loading if
isolation is absent, rather than hanging during preprocessing. The language
model and projector download about 1.58 GB together; CPU/WASM inference is
expensive. This example does not imply that arbitrary audio models accept the
same template. The standalone real-WASM check is
`scripts/validate_multimodal_wasm_audio.mjs`; browser qualification is separate.

`modality` is checked against Wllama after loading. Image input is discovered
from the loaded model/projector; video requires image support. The reported
audio flag is only an encoder capability check; it does not prove that inference
completes. These flags do not guarantee that a model's chat template accepts
multimodal content blocks. Evaluation returns
`MODEL_PROMPT_FORMAT_UNSUPPORTED` when Wllama cannot format that model's media
prompt. Set `jinja` or pass `chatTemplate` when an image model requires a
different template. The repository's `py website/serve.py` development server
sets COOP/COEP headers for browser WebAssembly threading; they do not by
themselves qualify the audio path. The
`@turbojev/web` media helpers encode input bytes but do not decode compressed
video containers. Use `videoFrameEvidence` with ordered image frames for video.

Copy the entire `src/` directory to `/llamacpp/`, including
`media-worker-pool.js`, or bundle `src/index.js` as one ES module. The host must
serve this adapter, TurboJev's generated WASM files, and the page
over localhost or HTTPS. Model files are fetched from their URL and remain in
the browser. Wllama 3.8.1 and its WASM binary are loaded from jsDelivr.

## Structured SDK extension

Source version 0.32.0 adds `generate` objects, arrays and ordered levels. Open values
use Wllama JSON Schema grammar; the Rust core checks every decoded value and owns
conditions, dependency scheduling, array continuation and duplicate stopping.
`imageEvidence([page1, page2], text)` preserves image order. The selected GGUF and
projector must accept those inputs.

Finite fields up to 26 values use the verified letter codebook. For 27–512 values,
provide a real full-continuation scorer through `runtime.useChoiceScorer(scorer)`
or a module URL in `runtimeOptions.choiceScorerModuleUrl`. That module must export
`createChoiceScorer(options)` and return an object with `maxChoices` (27–512) and
`async score(task, {engine, model})`. The task contains the complete semantic value
set and original evidence. Return `logits` in that same order, with one finite
log-score for each complete value continuation. Do not return a selected `value`.
Optional `input_tokens` records measured usage; omit it when unavailable. Report
`generated_tokens` and `thinking` if the scorer produces model text. The adapter
rejects missing scores, non-finite scores and a plugin-selected answer.

There is no built-in large-value scorer in this Wllama adapter. Native GGUF and
Transformers.js implement full text continuation scoring. Unsupported paths fail
instead of substituting first-token scores. Browser `generate(request,
{timeoutMs, signal})` cancellation is cooperative between runtime operations;
queued operations on a loaded engine are serialized.
