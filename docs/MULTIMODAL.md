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
    "scene": {"type":"choice", "instructions":"What is shown?",
              "criteria":{"cat":"A cat", "dog":"A dog"}}
  }
}
```

Use `audio` or `video` for those modalities. The current envelope carries one
media file plus text, up to 32 MiB of base64 (24 MiB decoded). The selected runtime
validates base64 and decodes bytes. The core imposes no model, Python, CUDA,
frame count, audio duration, download or server requirement. All decision types, including plain-text choices, use the typed `evaluate` request.

Rust owns validation, planning, normalization, calibration metadata and typed
responses. Runtime providers own media preprocessing and candidate scoring.

## Native applications

Rust runtimes implement `ScoringBackend` or `AsyncScoringBackend`, receiving the
original `DecisionBatch.state`. Node/TS, Python, C and .NET can also use the shared
native worker protocol with an application-selected executable:

```json
{
  "model":"application-selected-model",
  "worker":{"program":"/absolute/path/to/my-scorer",
            "args":["--model","/absolute/path/to/model"], "timeoutSeconds":300}
}
```

This is a generic desktop/server process interface, not mobile subprocess
support. No shell is used. The executable is configured only at load time.
It starts once per session and first emits:

```json
{"contract":"turbojev.scorer.v1","modalities":["text","image"]}
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
{"model":"application-selected-model","scores":[{"id":"scene","logits":[0.4,1.2]}],"metrics":{}}
```

Rust checks task/candidate alignment and creates choice/noul/score responses.
This path conservatively uses serial scoring without token-prefix reuse or
fused execution. Declare only implemented modalities. Token runtimes default
to text-only and keep the existing tokenizer interface.

`mediaEvidence(file, 'image', context)` from `@turbojev/web` creates input for
local `TurboJevWeb.evaluate`. `TurboJevRemote` is an optional HTTP client; there
is no automatic network fallback.

## Current implementation boundary

The generic native and browser interfaces are implemented. Built-in llama.cpp
and Transformers.js causal-LM adapters remain text-only and reject media.
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
