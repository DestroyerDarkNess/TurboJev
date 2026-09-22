# TurboJev Browser / WASM binding

`@turbojev/web` is the browser host binding for TurboJev. Browser APIs, model
fetching/cache, workers and WebGPU/WASM runtime handles live here; the portable
Rust core remains free of browser and operating-system dependencies.

## Boundary

```text
Browser application
       |
       v
@turbojev/web
       | module Worker
       v
turbojev-wasm
       |
       v
AsyncRuntimeBackend<R,C,E>
       |
       +-- Rust PromptRenderer
       +-- JS AsyncPromptCodec ------ exact chat template + tokenizer
       `-- JS AsyncRuntimeExecutor -- already-tokenized candidate scoring
       |
       v
TurboJev core -> calibration -> Choice/Noul/Score
```

The JavaScript runtime plugin does **not** own semantic decisions. It must expose
only four model-facing primitives:

```ts
interface BrowserRuntime {
  initialize(options): Promise<BrowserRuntimeDescriptor>;
  encodePrompt(messagesJson: string): Promise<string> | string;
  encodePromptWithAnswer(messagesJson: string, answer: string): Promise<string> | string;
  decodeTokens(tokenIdsJson: string): Promise<string> | string;
  scoreRuntimeBatch(runtimeBatchJson: string, planJson: string): Promise<string> | string;
  close(): Promise<void> | void;
}
```

Token arrays and runtime output cross this binding as JSON text. TurboJev WASM
performs dynamic candidate-code search, exact answer-boundary verification,
round-trip checking, shared-token-prefix factoring and execution planning. This
prevents a browser runtime from hard-coding `A/B/C` token IDs or reimplementing
Choice/Noul/Score logic.

## Loading

```ts
import { TurboJevWeb } from '@turbojev/web';

const jev = await TurboJevWeb.load({
  model: 'https://example.com/model.gguf',
  wasmModuleUrl: '/turbojev_wasm.js',
  runtimeModuleUrl: '/my-runtime.js',
  execution: { policy: 'auto', maxParallelTasks: 4 },
});

const result = await jev.classify(
  'My package never arrived',
  ['billing', 'shipping', 'support'],
);
```

A runtime module exports `createRuntime(options)` and returns a `BrowserRuntime`.
Its descriptor declares token-level capabilities, codec profile, resource hints,
and a default prompt profile. Applications may override the prompt profile at
load time without changing the model/runtime implementation.

## Model cache and large responses

The web binding can use CacheStorage. A remote response is cloned for cache and
returned to the runtime immediately; TurboJev does not wait for the cache branch
to consume the entire model first. This is important for multi-hundred-MB and
multi-GB weights because awaiting one side of a `Response.clone()` tee while the
runtime has not started consuming the other side can create large transient
buffers.

Cache failure does not invalidate an otherwise usable model response. Model
resolution/network errors and inference/runtime errors remain distinct typed
errors.

## Runtime capabilities

Browser plugins describe the loaded model/runtime combination using:

```text
next_token_logits
sequence_scores
shared_prefix
parallel_suffixes
selective_output_head
local_inference
```

Claims are capability contracts, not marketing labels. A runtime that computes a
full vocabulary head and gathers candidates must not claim
`selective_output_head`. Multi-token codes require exact sequence scoring.

## WebGPU and third-party runtimes

TurboJev does not make wllama, WebLLM, ONNX Runtime Web, or any other browser
engine a dependency of the core. A concrete plugin may use one of them if it can
satisfy the low-level contract above. If a third-party runtime exposes only a
high-level text-generation API, it is not sufficient for TurboJev's direct path
unless an adapter can obtain exact candidate scores and exact tokenizer/chat
boundary behavior.

## Verification

The web package gate runs generated-schema checks, TypeScript typechecking,
worker lifecycle tests, model-cache tests and runtime-contract tests. The Rust
CI additionally checks `turbojev-wasm` for `wasm32-unknown-unknown` separately
from the portable core crate set.
