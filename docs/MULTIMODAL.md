# Model-independent multimodal input

TurboJev has no dependency on Jev-Omni or any specific checkpoint. Applications
select a model and runtime. Text, image, audio and video are input modalities;
actual codec and model support must be declared by each runtime.

## Shared contract

All bindings use `evaluate` with the same media evidence:

```json
{
  "state": {
    "contract": "turbojev.media.v1",
    "text": "Optional context",
    "modality": "image",
    "base64": "<base64-encoded-media>"
  },
  "questions": {
    "scene": {
      "type": "choice",
      "instructions": "What is shown?",
      "criteria": { "cat": "A cat", "dog": "A dog" }
    }
  }
}
```

For image or audio, send one base64 payload plus text, up to 32 MiB of base64
(24 MiB decoded). For video, send up to 32 ordered, timestamped still frames;
each frame has `timestamp_ms` and its own base64 image. The combined frame data
uses the same 24 MiB decoded limit. The host decodes video containers into
frames; TurboJev does not accept MP4 or invoke a media executable. Every
language binding sends this same JSON contract. All decision types, including
plain-text choices, use the typed `evaluate` request.

```json
{
  "state": {
    "contract": "turbojev.media.v1",
    "text": "Compare the start and end of the clip.",
    "modality": "video",
    "frames": [
      { "timestamp_ms": 0, "base64": "<first-frame>" },
      { "timestamp_ms": 1200, "base64": "<later-frame>" }
    ]
  },
  "questions": {
    "change": {
      "type": "choice",
      "instructions": "What changed?",
      "criteria": { "arrived": "An object arrived", "same": "The scene stayed the same" }
    }
  }
}
```

Rust owns validation, planning, normalization, calibration metadata and typed
responses. Runtime providers own media preprocessing and candidate scoring.

## Native applications

Rust runtimes implement `ScoringBackend` or `AsyncScoringBackend`, receiving the
original `DecisionBatch.state`. Node/TS, Python, C and .NET can also use the shared
native worker protocol with an application-selected executable:

```json
{
  "model": "application-selected-model",
  "worker": {
    "program": "/absolute/path/to/my-scorer",
    "args": ["--model", "/absolute/path/to/model"],
    "timeoutSeconds": 300
  }
}
```

This is a generic desktop/server process interface, not mobile subprocess
support. No shell is used. The executable is configured only at load time.
It starts once per session and first emits:

```json
{ "contract": "turbojev.scorer.v1", "modalities": ["text", "image"] }
```

It then receives one serialized `DecisionBatch` per input line, and returns one
line with `model` (string or null), `scores` and `metrics`, or an `error` string.
Diagnostics go to stderr. Closing the session terminates it; timeout invalidates
it. Unsupported modalities are rejected before dispatch.

## Local browser / WASM

`TurboJevWeb.load` uses an application-supplied `runtimeModuleUrl` and a WASM
module. A module exports `createRuntime()`. Its `initialize()` descriptor can
select the generic decision interface:

```javascript
{
  name: 'my-runtime', model: 'application-selected-model',
  scoringMode: 'decisions', modalities: ['text', 'image'],
  capabilities: {
    nextTokenLogits: false, sequenceScores: false, sharedPrefix: false,
    parallelSuffixes: false, selectiveOutputHead: false, localInference: true
  }
}
```

Implement `scoreDecisionBatch(batchJson, planJson)` and `close()`. Scoring may
return a promise. The batch contains original `state` and `tasks`, each with
`id`, `intent`, optional `instruction`, and ordered `candidates` containing
`key` and `description`. The runtime uses the chosen model's processor and
returns unnormalized logits or comparable log-scores, **not probabilities**:

```json
{
  "model": "application-selected-model",
  "scores": [{ "id": "scene", "logits": [0.4, 1.2] }],
  "metrics": {}
}
```

Rust checks task/candidate alignment and creates choice/noul/score responses.
This path conservatively uses serial scoring without token-prefix reuse or
fused execution. Declare only implemented modalities. Token runtimes default
to text-only and keep the existing tokenizer interface.

`mediaEvidence(file, 'image', context)` from `@turbojev/web` creates image or
audio input for local `TurboJevWeb.evaluate`. For video, use
`videoFrameEvidence([{ timestampMs, image }, ...], context)` with decoded image
frames. `TurboJevRemote` is an optional HTTP client; there is no automatic
network fallback.

The built-in `@turbojev/runtime-llamacpp-wasm` adapter loads Wllama 3.8.1,
including its WebAssembly `mtmd` path. Supply `mmprojUrl` with the matching GGUF
projector for image input. Video uses ordered, timestamped image blocks and
requires image support. The browser decodes video containers into frames. This
is separate from the native C ABI build and does not add a platform-specific
dependency to the Rust core or language bindings. The adapter now forwards audio
as the first user-content block when Wllama reports an audio encoder, matching
Wllama's own multimodal chat client. The adapter prestarts additional workers
for mtmd's audio preprocessing; these are separate from the requested inference
thread count. Audio requires cross-origin isolation and fails early without it.
`cyclicOrderRobustness` applies to media decision runtimes as well as text.
The standalone real-WASM runner exercises the adapter and compiled Rust core in
Node with LFM2 Audio; that is separate from browser qualification. The
repository's `py scripts/serve-browser-tests.py` sets COOP/COEP headers for WebAssembly
threading; those headers alone do not qualify audio inference.

## Native GGUF with llama.cpp

The native loader accepts an optional `mmproj` path alongside the main GGUF:

```json
{ "model": "/models/model.gguf", "mmproj": "/models/mmproj-model.gguf" }
```

`mmproj` is a normal load option in the Node and Python APIs; C, C#, Android,
Swift and Rust bindings pass the same option through their shared JSON loader.
The request state above is likewise unchanged across those bindings.

The native bindings use the same C ABI and media JSON contract; the runtime
libraries still have to match the host OS and CPU architecture. The Windows
publisher packages `mtmd.dll` with each x86, x64, and ARM64 llama.cpp runtime,
and the Linux E2E staging check requires `libmtmd`. A successful package build
does not by itself qualify inference on every target. macOS packaging and native
multimodal inference remain unverified. Browser inference is a separate WASM
build; it loads Wllama's `mtmd` implementation and uses the same media JSON
contract without adding native binaries to non-browser packages.

TurboJev builds and packages llama.cpp `mtmd` with its native runtime; image and
audio decoding stays in-process. Use a projector intended for the loaded text
model. The runtime's modality flags report which preprocessing paths are present
in `mtmd`; they do not certify that a model and projector were trained together
or that the model understands that modality. llama.cpp checks embedding
dimensions, but a same-width projector from another checkpoint can still load.
Validate decision quality separately. Unsupported modalities fail before
scoring. Video uses a timestamped sequence of decoded still images, passed to
`mtmd` as mergeable frames. The host must
decode compressed containers such as MP4; TurboJev does not bundle or launch
FFmpeg/ffprobe. Use a checkpoint trained for video when temporal understanding
matters. The runtime's video capability means it accepts ordered frames; it does
not promise that the selected checkpoint can reason about time. This frame
contract works through the same JSON request in all language bindings.

The GGUF path scores candidate logits without generating response text and
currently requires a single-token candidate codebook. The supported codecs and
architectures are those implemented by the pinned llama.cpp `mtmd` build, not
all files carrying a GGUF extension.

Candidate scores can still be order-sensitive for a particular checkpoint.
`OrderRobustness::Cyclic` evaluates every rotation of each choice and averages
the aligned logits; it is opt-in because it adds one model evaluation per
candidate. The default remains a single pass. Test the actual checkpoint and
prompt with permuted candidates before relying on the decision. The small native
LFM2 Audio probe and its reproducible runner are documented in
[`MULTIMODAL_VALIDATION.md`](MULTIMODAL_VALIDATION.md).

## Current implementation boundary

The generic native and browser interfaces are implemented. Built-in native
llama.cpp accepts image/audio input and timestamped video frames when opened
with a compatible `mmproj` and chat template. Browser GGUF supports text, image,
and video frames. Audio bytes are wired into the experimental WASM adapter, but
browser GGUF audio remains unqualified until a real model completes
`TurboJevWeb.evaluate` and returns the shared typed response.
Compressed video containers need host-side/browser decoding.
Transformers.js causal-LM adapters remain text-only.
Optional application workers are now provided in
[`samples/TurboJev-SmolVLM2`](../samples/TurboJev-SmolVLM2/README.md) and
[`samples/TurboJev-LFM-Audio`](../samples/TurboJev-LFM-Audio/README.md).
They run selected Transformers/liquid-audio architectures on CPU and preserve
Rust decision semantics. See [real validation status](MULTIMODAL_VALIDATION.md).
These interfaces do not by themselves make any GGUF, ONNX or safetensors model
multimodal. No claim of universal format or hardware support is made.

Protocol fixture tests cover contract negotiation, unchanged evidence, Rust responses, invalid
scores and process errors using protocol fixtures. Fixture results do not prove
real inference. Keep `RELEASE_ENABLED` unset until implementation, model,
platform and distribution release gates are complete.
