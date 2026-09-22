# TurboJev

Try the [five runnable sample projects](samples/README.md): vanilla browser JavaScript, TypeScript/Node, Python, C#/.NET and Rust. Shared build scripts prepare the compiled libraries for each host architecture.

TurboJev is a Rust-first, universal typed-decision layer for open-weight models.

TurboJev is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for
the full terms.

```text
state + typed questions -> candidate scores -> probabilities -> typed answers
```

The optimized path does not require natural-language answer generation. Model
runtimes return candidate logits/scores; TurboJev owns schema validation,
execution planning, calibration, and `choice` / `noul` / `score` construction.

> Status: **v0.20 specialized evaluation + external-run interchange milestone.** The portable core remains model-,
> runtime-, operating-system-, host-language-, prompt-template-, and async-executor-agnostic.
> Node/TypeScript, browser/WASM, Python, C, .NET, Android and iOS converge on the same Rust
> semantic contracts. v0.20 adds framework-neutral external evaluation runs, fair generic-vs-specialized
> comparison, robustness metrics and reproducible SHA-256 experiment bundles while keeping all trainers external.



## Specialized evaluation and external interop (v0.20)

`turbojev-specialized-eval` accepts framework-neutral external runs containing semantic candidate keys, raw logits and measurement provenance. It rejects mismatched benchmark/environment pairs, computes baseline accuracy/NLL/Brier/ECE, candidate-order/paraphrase invariance, latency/throughput/memory, and emits specialized-minus-generic deltas without collapsing them into a synthetic winner score. `turbojev-specialized-report` is the separate host CLI, and `turbojev.specialization-bundle.v1` pins the experiment evidence by SHA-256. See `docs/SPECIALIZED_EVALUATION.md`.


## Model-bound specialized artifacts (v0.19)

`turbojev-specialized-artifact` connects v0.18 materialized records to the exact `PromptRenderer` + `PromptCodec` used by an open-weight checkpoint. `BoundaryVerifiedCodeProvider` performs deterministic code discovery at the real assistant answer boundary, and `turbojev.model-bound-training-dataset.v1` exports rendered messages, prompt token ids, candidate token ids and the exact target token for framework-neutral external LoRA/QLoRA training.

TurboJev still does **not** train models itself and does not depend on PyTorch, TRL, Axolotl, MLX, CUDA or another trainer. v0.19 also adds specialized manifest construction, strict calibration-provenance handoff, and a fail-fast runtime compatibility gate. See `docs/SPECIALIZED_ARTIFACTS.md`.

## Specialized training data (v0.18)

`turbojev-training-data` adds versioned semantic datasets and deterministic materialization into candidate-logit training records. Training reuses the exact `EvaluationRequest -> DecisionBatch` compiler used by inference, preserves typed `choice`/`noul`/`score` targets, expands candidate-order and candidate-code permutations reproducibly, and conserves source-decision weights across augmentation.

Candidate token ids are still never guessed by the portable core: a host-side `CandidateCodeProvider` supplies codes verified against the real checkpoint renderer/tokenizer boundary. See `docs/TRAINING_DATA.md`.

## Specialized model contracts (v0.17)

Phase 10 begins without coupling TurboJev to a training framework. `turbojev-specialized` defines reproducible LoRA/QLoRA recipes, direct candidate-logit cross-entropy, candidate/order permutation policy, multilingual data requirements, explicit train/validation/calibration/benchmark splits and generic-vs-specialized evaluation plans.

`ModelSpec.decision_contract` is optional: generic open-weight models keep the exact same API and execution path, while specialized checkpoints can pin tokenizer/prompt/renderer/candidate-encoding/context/calibration compatibility.

See `docs/SPECIALIZED_MODELS.md`.

## Calibration tooling (v0.16)

TurboJev treats raw softmax probabilities as conditional scores unless a fitted profile says otherwise. v0.16 adds portable deterministic temperature fitting and auditable metrics:

```rust
use turbojev::calibration::{
    fit_temperature_profile, CalibrationDataset, TemperatureFitConfig,
    TemperatureCalibrator,
};

let profile = fit_temperature_profile(
    &dataset,
    "support-routing-v3",
    TemperatureFitConfig::default(),
)?;
let calibrator = TemperatureCalibrator::from_profile(&profile)?;
```

The companion host tool can fit and report profiles without adding filesystem or process APIs to the portable core:

```text
turbojev-calibrate report holdout.json --temperature 1.0
turbojev-calibrate fit holdout.json profile.json --id support-routing-v3
```

Profiles carry holdout identity plus optional model/tokenizer/prompt/renderer provenance and report before/after NLL, multiclass Brier, top-label ECE and accuracy.

## Non-negotiable portability rule

Portable TurboJev crates do **not** directly depend on:

- Windows, Linux, macOS, Android, iOS, or browser APIs;
- filesystem/network/process/environment APIs;
- CUDA, Metal, Vulkan, WebGPU, or another hardware API;
- llama.cpp, Candle, wllama, or another concrete inference runtime;
- a hard-coded list of model architectures.

Those concerns live in optional runtime adapters and host bindings.

## What “universal model support” means

TurboJev core contains no `Qwen`, `Llama`, `Gemma`, `MiniCPM`, or similar
allowlist. Any open-weight model can be integrated when a runtime adapter can:

1. render TurboJev's semantic prompt IR into the model's expected format;
2. tokenize with the real tokenizer;
3. score the allowed candidate outputs exactly;
4. return ordered candidate logits/scores.

`turbojev-backend-custom` already lets applications attach arbitrary model
runtimes without changing the core.

## Simple Rust API

```rust
use turbojev::{backend_mock::FixtureBackend, TurboJev};

let backend = FixtureBackend::builder()
    .with_logits("classification", vec![0.2, 3.7, -1.0])
    .build();

let jev = TurboJev::new(backend);
let result = jev.classify(
    "My package never arrived",
    ["billing", "shipping", "other"],
)?;

assert_eq!(result.choice, "shipping");
# Ok::<(), turbojev::Error>(())
```

A real runtime backend uses the exact same `TurboJev` API.

## Full typed API

```rust
use indexmap::indexmap;
use turbojev::{
    backend_mock::FixtureBackend,
    schema::{EvaluationRequest, Question},
    TurboJev,
};

let backend = FixtureBackend::builder()
    .with_logits("route", vec![0.2, 3.7, -1.0])
    .with_logits("urgent", vec![2.0, 0.1])
    .build();

let jev = TurboJev::new(backend);
let result = jev.evaluate(EvaluationRequest {
    model: None,
    state: serde_json::json!({"message": "My package never arrived"}),
    questions: indexmap! {
        "route".into() => Question::choice(
            "Which queue should handle this request?",
            indexmap! {
                "billing".into() => "Billing or payment issue".into(),
                "shipping".into() => "Delivery or logistics issue".into(),
                "other".into() => "None of the above".into(),
            },
        ),
        "urgent".into() => Question::noul(
            "Does this require urgent human intervention?"
        ),
    },
})?;

println!("{:#?}", result.answers);
# Ok::<(), turbojev::Error>(())
```


## Node.js / TypeScript

```ts
import { TurboJev } from "@turbojev/node";

const jev = await TurboJev.load({
  model: "./Qwen3-0.6B-Q8_0.gguf",
  execution: { policy: "auto", maxParallelTasks: 4 },
});

const result = await jev.classify(
  "My package never arrived",
  ["billing", "shipping", "technical", "other"],
);

await jev.close();
```

The package also accepts `file://`, direct HTTP(S), and
`hf://owner/repository/file.gguf` sources. Model I/O/cache logic lives only in the
Node host binding. Native inference runs in a dedicated `worker_threads` worker so
the main event loop is not blocked. See `docs/NODE_BINDING.md`.

## Browser / WebAssembly

```ts
import { TurboJevWeb } from "@turbojev/web";

const jev = await TurboJevWeb.load({
  model: "<transformersjs-compatible-model>",
  modelResolution: "runtime",
  wasmModuleUrl: "/turbojev_wasm.js",
  runtimeModuleUrl: "/runtime-transformersjs.js",
  runtimeOptions: { device: "webgpu", dtype: "q4" },
});

const result = await jev.classify(
  "My package never arrived",
  ["billing", "shipping", "support"],
);

await jev.close();
```

`@turbojev/web` owns Worker/browser concerns while `turbojev-wasm` keeps candidate
discovery, planning, calibration and typed answers in Rust. The optional
`@turbojev/runtime-transformersjs` adapter owns only checkpoint chat-template,
tokenization and causal-LM next-token scoring. It currently advertises the
conservative serial full-vocabulary reference path; it does not pretend that
full-vocabulary + gather is candidate-only projection. See `docs/WEB_BINDING.md`
and `docs/TRANSFORMERSJS_RUNTIME.md`.

## Python / C / .NET

Python:

```python
from turbojev import TurboJev

with TurboJev.load("./model.gguf") as jev:
    result = jev.classify(
        "My package never arrived",
        ["billing", "shipping", "technical", "other"],
    )
```

.NET:

```csharp
await using var jev = await TurboJevEngine.LoadAsync(
    new LoadOptions("./model.gguf"));

var result = await jev.ClassifyAsync(
    "My package never arrived",
    new[] { "billing", "shipping", "technical", "other" });
```

The C ABI exposes only opaque engine handles and UTF-8 JSON. Python uses a thin
PyO3 wrapper over the shared native host layer; .NET uses the same C ABI from a
dedicated managed thread so thread-affine runtimes remain valid. See
`docs/C_ABI.md`, `docs/PYTHON_BINDING.md`, and `docs/DOTNET_BINDING.md`.

## Android / iOS

Android uses the `turbojev-jni` host adapter and a Kotlin `HandlerThread`; Swift uses the stable C ABI from a dedicated Foundation `Thread`. Both reuse `turbojev-host-native` and keep runtime/device selection outside the portable core. See `docs/MOBILE_BINDINGS.md`.

## Arbitrary runtime integration

```rust
use turbojev::{
    backend::{BackendCapabilities, BackendOutput, BackendMetrics, TaskScores},
    backend_custom::CustomBackend,
};

let runtime = CustomBackend::new(
    BackendCapabilities::default(),
    |batch, _plan| {
        // Call any open-weight model runtime here and return its candidate scores.
        Ok(BackendOutput {
            model: Some("my-open-model".into()),
            scores: batch.tasks.iter().map(|task| TaskScores {
                id: task.id.clone(),
                logits: vec![0.0; task.candidates.len()],
            }).collect(),
            metrics: BackendMetrics::default(),
        })
    },
);
```

## Workspace

```text
crates/
  turbojev/                    facade
  turbojev-schema/             canonical typed API
  turbojev-calibration/        softmax / calibration
  turbojev-tokenizer/          single + multi-token code discovery
  turbojev-prompt/             language-neutral semantic prompt IR
  turbojev-model/              opaque model descriptors
  turbojev-backend/            runtime SPI + capability contract
  turbojev-engine/             execution planner
  turbojev-core/               orchestration + typed answers
  turbojev-wire/               canonical JSON boundary for bindings
  turbojev-backend-mock/       deterministic test backend
  turbojev-backend-custom/     arbitrary application runtime adapter
  turbojev-runtime/            universal token-level runtime bridge
  turbojev-sequence/           exact prefix-sharing multi-token scoring
  turbojev-runtime-conformance/ adapter acceptance/conformance kit
  turbojev-compat/              Jev/TypeSafe/AI SDK wire profiles
  turbojev-conformance/         canonical response parity
  turbojev-runtime-llamacpp/    optional GGUF/llama.cpp adapter
  turbojev-projection/           candidate-only projection math/parity
  turbojev-runtime-projection/   portable dense selective executor
  turbojev-benchmark/            reproducible benchmark protocol
  turbojev-host-native/          shared native binding/session layer
  turbojev-node/                 thin Node-API binding (host-only)
  turbojev-wasm/                 browser/WASM host binding (host-only)
  turbojev-capi/                 stable opaque-handle C ABI
  turbojev-python/               PyO3 extension binding
  turbojev-jni/                  Android JNI host adapter

bindings/
  node/                          @turbojev/node TypeScript package
  web/                           @turbojev/web worker/WASM package
  web-runtimes/transformersjs/  optional Transformers.js/WebGPU adapter
  c/                             public turbojev.h ABI header
  python/                        Python package
  dotnet/                        .NET P/Invoke package
  android/                       Kotlin/Android host wrapper
  swift/                         Swift/iOS host wrapper
```

## Execution planning

Backends advertise semantic capabilities:

```text
shared_prefix
parallel_suffixes
selective_output_head
multi_token_candidates
local_inference
```

The planner chooses among independent, shared-prefix, serial/parallel, and
full-vocabulary/candidate-only paths without branching on OS or hardware brand.

## Multilingual behavior

TurboJev preserves caller text exactly. The core does not translate or impose a
mandatory English system prompt. `turbojev-prompt` creates semantic prompt IR;
the model adapter chooses the rendering/profile expected by the checkpoint.

## Cross-language strategy

`turbojev-wire` is the canonical JSON boundary. Node/TypeScript, browser/WASM, Python, C,
.NET, Android/Kotlin and iOS/Swift bindings are implemented. All hosts delegate to the same
Rust contracts instead of reimplementing decision logic.

See:

- `SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/PORTABILITY.md`


## v0.3 — compatibility and conformance

TurboJev now keeps three concerns explicitly separate:

```text
external wire profile -> canonical schema -> decision core -> runtime adapter
```

`turbojev-compat` provides pure-data compatibility profiles for TurboJev native JSON, TypeSafe System One (`noul`) and AI SDK evaluation (`boolean`). `turbojev-conformance` compares reference and optimized paths with explicit probability/score tolerances so shared-prefix, batching, quantization and selective-head optimizations can be measured rather than assumed equivalent.

No compatibility crate opens a socket or knows an operating system.


## Universal Runtime Bridge

TurboJev v0.4 introduces `turbojev-runtime`. A concrete model integration only needs to provide a model-facing `PromptRenderer`, a `PromptCodec` that applies the checkpoint's real chat template/tokenizer, and a `RuntimeExecutor` that returns candidate logits/log-scores. The bridge verifies opaque candidate codes at the exact answer boundary, prefers single-token scoring, supports exact multi-token scoring when advertised, and factors a shared token prefix only when the execution plan requests it.

This is deliberately independent of GGUF, safetensors, llama.cpp, Candle, WebGPU, MLX, ONNX, CUDA, Metal, Vulkan, Android, iOS, Windows, Linux and macOS. Those belong in optional outer adapters.

## Exact multi-token candidate scoring (v0.15)

Single-token candidate codes remain the preferred fast path. When a tokenizer cannot provide
enough exact one-token codes at the model's real answer boundary, `turbojev-sequence` scores
prefix-free token sequences with exact conditional log-likelihood:

```text
score(c_1..c_n) = Σ_i [ logit(c_i) - logsumexp(full_vocab_logits_i) ]
```

Candidate sequences are organized as a trie, so a shared prefix such as `AA` / `AB` is
evaluated once. Runtime metrics expose candidate-token work, unique prefix evaluations, and
saved token steps. Because exact multi-token probabilities need the full-vocabulary
normalizer at every continuation state, v0.15 automatically disables candidate-only output
projection for this fallback unless a future backend can provide an equivalent exact `logZ`.
The actual downgraded execution plan is reported through `BackendMetrics.effective_plan`; the
core never claims an optimization that the runtime did not execute. See
`docs/MULTI_TOKEN_SCORING.md`.


## Verification

Portable release gates are documented in `docs/TESTING.md`. Structural/portability checks are runnable without Cargo; release correctness still requires `cargo fmt`, `cargo check`, `cargo clippy -D warnings`, the full test suite, the `wasm32-unknown-unknown` check, and the Linux/Windows/macOS CI matrix.

## Async without an async framework

TurboJev does not make Tokio, async-std, a browser runtime, or an operating
system part of its core contract. Synchronous runtimes use `TurboJev`; naturally
asynchronous runtimes use `TurboJevAsync`. Both share the same schema, planner,
calibration, and typed answer construction.

```text
TurboJev      -> ScoringBackend      -> RuntimeExecutor
TurboJevAsync -> AsyncScoringBackend -> AsyncRuntimeExecutor
```

The returned futures intentionally do not require `Send`, which keeps
thread-affine WebGPU/JavaScript/mobile handles valid. See `docs/ASYNC.md`.

## Runtime acceptance gate

`turbojev-runtime-conformance` verifies exact answer-boundary tokenization,
candidate-code round trips, capability truthfulness, output shapes, finite
scores, and optimized-vs-reference logit parity. Concrete adapters provide the
model fixtures; the invariant machinery stays model- and platform-neutral. See
`docs/RUNTIME_CONFORMANCE.md`.

## v0.6–v0.9 — GGUF baselines plus candidate-only projection

The first concrete runtime remains an optional outer adapter. v0.6 established
the independent serial/full-vocabulary reference path, v0.7 added deterministic
shared-prefix reuse, and v0.8 added bounded isolated parallel suffix batches.
v0.9 adds a runtime-neutral candidate-only output-head contract, portable dense
reference math, and conformance that verifies both numerical parity and actual
candidate-row accounting.

```text
reference:
  full prompt A -> decode -> logits
  full prompt B -> decode -> logits

shared serial:
  exact common token prefix -> prefill once as sequence 0
                               +-> copy -> suffix A -> logits -> delete branch
                               +-> copy -> suffix B -> logits -> delete branch

parallel:
  exact common token prefix -> prefill once as sequence 0
                               +-> copy -> seq 1 --\
                               +-> copy -> seq 2 ----> one bounded decode -> rows
                               +-> copy -> seq N --/
                               -> delete all ephemeral branches
```

`ExecutionBudget` constrains automatic parallel scheduling without exposing any
platform API. The planner combines the caller task cap/memory budget with
model-runtime `BackendResourceHints` and emits a concrete `parallel_task_limit`.
If the safe limit is less than two tasks, automatic execution falls back to
serial shared-prefix reuse.

The stock llama.cpp FFI path remains conservative and reports
`selective_output_head = false` unless a low-level integration can genuinely
bypass the full vocabulary head. `turbojev-runtime-projection` demonstrates a
true candidate-only path for runtimes that can expose final hidden states and
requested output rows; the llama.cpp adapter contract also supports a future or
custom parity-verified selective implementation without changing TurboJev core.

Use `PlanningPolicy::Reference` for the independent correctness baseline,
`PlanningPolicy::DeterministicReuse` for cached serial execution, or the default
`PlanningPolicy::Auto` for bounded parallel execution when the runtime advertises
it.


See `docs/PARALLEL_EXECUTION.md` for scheduling/cache isolation and
`docs/CANDIDATE_PROJECTION.md` for the v0.9 selective-head contract.

## Final real-model validation (v0.28)

Phase 11 separates protocol, cross-request execution, host transport and concrete runtime capability truth. v0.25 added the llama.cpp **multi-root** path; v0.26 made activation evidence-bound; v0.27 introduced the cross-language harness; and v0.28 turns that harness into the final reproducible release gate. The same small open-weight GGUF and canonical request must produce semantically equivalent complete `EvaluationResponse` objects through Node/TypeScript, Python, C and .NET/C#, while real llama.cpp independent-vs-multi-root qualification independently controls whether fused execution may be enabled. Fixture/stub tests are never treated as real-model release evidence. Patch v0.28.1 additionally freezes per-run Cargo/npm dependency resolution, verifies the planned host toolchain versions and records dependency-lock provenance in the final report. Patch v0.28.2 hardens the final .NET path: the E2E matrix reuses the prebuilt Release binary and the wrapper tears down its dedicated affinity thread if native model opening fails. See `docs/SERVER.md`, `docs/RUNTIME_QUALIFICATION.md`, `docs/E2E_LANGUAGE_MATRIX.md`, and `.github/workflows/real-e2e.yml`.
- v0.28.2 also guarantees the .NET affinity worker is torn down if native close fails during disposal.
