# TurboJev architecture

## Goal

TurboJev is a universal typed-decision layer over open-weight models. Universal
means the semantic core is independent of model architecture, model format,
operating system, hardware runtime, browser/native execution, and host language.

It does **not** mean every checkpoint can execute without an inference runtime.
A model is usable when an adapter can render its prompt, tokenize it, score the
allowed candidate outputs, and return those scores through TurboJev's SPI.

## Dependency direction

```text
                          host application
                                │
                 ┌──────────────┴───────────────┐
                 │                              │
          language binding               runtime adapter
                 │                              │
                 ▼                              ▼
           turbojev-wire                 ScoringBackend
                 │                              ▲
                 ▼                              │
         turbojev-schema                       │
                 │                              │
                 ▼                              │
          turbojev-core ──► turbojev-engine ───┘
             │      │
             │      └────► turbojev-calibration
             │
             ├───────────► turbojev-prompt (semantic IR)
             ├───────────► turbojev-tokenizer (code discovery)
             └───────────► turbojev-model (opaque descriptors)
```

There is no arrow from the portable core to an operating-system or GPU runtime.

## Decision pipeline

```text
EvaluationRequest
      │
      ▼
Schema validation
      │
      ▼
DecisionBatch
      │
      ├──── BackendCapabilities
      │            │
      │            ▼
      │     ExecutionPlanner
      │            │
      └────────────┤
                   ▼
             ExecutionPlan
                   │
                   ▼
            ScoringBackend
                   │
             candidate logits
                   │
                   ▼
               Calibrator
                   │
                   ▼
        typed Choice/Noul/Score
```

## Universal model adapter path

```text
PromptDocument (semantic IR)
        │
        ▼
model-specific PromptRenderer
        │
        ▼
model-specific tokenizer
        │
        ▼
CandidateCodebook
        │
        ├─ single-token fast path
        └─ fixed-width multi-token fallback
        │
        ▼
open-weight runtime
        │
        ├─ full vocabulary + gather
        └─ candidate-only projection
        │
        ▼
TaskScores
```

The runtime adapter can wrap a first-party TurboJev runtime or application-owned
inference code. `turbojev-backend-custom` already provides a portable integration
point for arbitrary host runtimes.

## Execution planning

The planner operates only on semantic capabilities:

- `shared_prefix`
- `parallel_suffixes`
- `selective_output_head`
- `multi_token_candidates`

It never asks whether the host is Windows, Linux, macOS, Android, iOS, WASM, or
anything else.

`Reference` execution remains deliberately slow and simple so optimized cached
or parallel paths always have a semantic baseline to compare against.

## Prompting

Prompt semantics and prompt rendering are separate.

The core preserves caller text verbatim and compiles a language-neutral
`PromptDocument`. The runtime/model adapter owns actual instruction wording and
chat-template formatting. This is required both for multilingual checkpoints and
for models with non-chat or specialized decision templates.

## Platform boundaries

Portable crates contain no filesystem, networking, environment, process, browser,
or OS APIs. Optional future adapters may contain such code, but no adapter is a
mandatory dependency of `turbojev-core`.

This lets one application choose, for example:

```text
Desktop A -> GGUF runtime adapter
Desktop B -> Rust-native runtime adapter
Browser   -> WebGPU/WASM adapter
Mobile    -> host-native adapter
Server    -> application-owned model executor
```

while preserving exactly the same request, calibration, and typed response
semantics.


## Universal runtime bridge (v0.4)

`turbojev-runtime` is the only portable layer allowed to compose semantic prompts with token-level execution. It defines `PromptCodec`, `RuntimeExecutor`, `RuntimeCapabilities`, `RuntimeBatch`, and `RuntimeBackend`. Concrete runtimes remain separate adapters.

```text
TurboJev Core
    |
ScoringBackend
    |
RuntimeBackend<R,C,E>
    |-- PromptRenderer ---- semantic IR -> model-facing messages
    |-- PromptCodec ------- chat template/tokenizer -> exact tokens
    `-- RuntimeExecutor --- candidate token sequences -> raw logits
```

No runtime adapter should duplicate Choice/Noul/Score semantics or calibration.

## v0.6–v0.9 concrete runtimes and projection layer

The first concrete runtime is intentionally outside the portable dependency
cone:

```text
turbojev-core -> turbojev-backend -> turbojev-runtime
                                      ^
                                      |
                         turbojev-runtime-llamacpp
                                      |
                         turbojev_llama_shim ABI
                                      |
                                  llama.cpp
```

The arrow from the concrete adapter points *toward* the portable contracts; no
portable crate points back to the adapter. This dependency direction is a
release invariant.

v0.6 froze independent/serial/full-vocabulary scoring as the reference path.
v0.7 added deterministic shared-prefix serial reuse and v0.8 added bounded
parallel suffix branches. v0.9 adds `turbojev-projection` and
`turbojev-runtime-projection` so a runtime can project only requested output
rows while the frozen full-vocabulary path remains the correctness baseline.
The stock llama.cpp FFI adapter does not claim this capability until its native
path can truly skip the ordinary vocabulary head.

## Specialized artifact and evaluation dependency path (v0.17–v0.20)

Specialization remains adjacent to inference rather than embedded inside it:

```text
TrainingRecipe ───────────────► turbojev-specialized
                                      │
SemanticTrainingDataset              │
        │                             │
        ▼                             ▼
 turbojev-training-data ─────► policy/compatibility validation
        │
        ├──► turbojev-core::compile_decision_batch
        │          (same choice/noul/score normalization as inference)
        │
        ▼
MaterializedTrainingDataset
        │
        ▼
 turbojev-specialized-artifact
        │
        ├──► PromptRenderer
        ├──► PromptCodec
        ├──► exact answer-boundary verification
        │
        ▼
ModelBoundTrainingDataset
        │
        ▼
external trainer chosen by the user
        │
        ▼
open-weight specialized checkpoint
        │
        ▼
external evaluation harness
        │
        ▼
turbojev.specialization-run.v1
        │
        ▼
turbojev-specialized-eval
        │
        ├──► turbojev-calibration metrics
        └──► generic-vs-specialized report/bundle
```

`turbojev-training-data`, `turbojev-specialized-artifact`, and `turbojev-specialized-eval` are portable. Neither contains filesystem, model download, GPU, Python, CUDA, Metal, browser, OS, optimizer or training-framework dependencies. The latter reuses the same renderer/tokenizer abstractions as inference and exports exact tokenized records for external optimization.

The v1 `candidate-logit-cross-entropy` objective explicitly requires single-token candidate codes. TurboJev's exact multi-token inference path is unchanged; a future multi-token training objective must be a new explicit contract instead of silently changing the meaning of v1.

This dependency direction prevents specialized training from creating a second interpretation of the request schema or tokenizer boundary while keeping trainer frameworks out of TurboJev.



v0.20 keeps `turbojev-specialized-report` outside the portable graph. It owns filesystem/CLI concerns for consuming external run JSON and writing canonical reports. Pair evaluation requires matching prompt/tokenizer contracts plus backend/host/quantization/execution-path provenance before performance deltas are accepted.

## Server execution boundary (v0.21-v0.28)

Server compatibility is split so HTTP concerns cannot leak into the semantic core:

```text
turbojev-server                 host-only sockets / threads / clocks
        │
        ├────► turbojev-server-protocol
        │          portable route/profile/batch semantics
        │
        └────► turbojev-server-execution
                   portable cross-request capabilities / planning / parity
```

`turbojev-server-execution` does not own a queue or a clock. The host supplies cancellation/deadline observation and decides how jobs are admitted. A backend may advertise fused cross-request execution only if its `execute_fused` path preserves ordered `EvaluationResponse` parity with serial reference execution. Since v0.23, `turbojev-host-native` is the source of truth for the loaded session's cross-request capabilities; the HTTP server maps those capabilities instead of inferring them from runtime names. v0.24 adds `FusedScoringBackend` at the backend layer and `FusedRuntimeExecutor` at the runtime layer. The deterministic fixture activates the fused backend path for conformance. v0.25 additionally implements a llama.cpp multi-root native operation using independent sequence IDs in one decode, but native-host exposure is disabled by default and becomes visible only after an explicit target-specific conformance opt-in. Per-shape eligibility can still downgrade a proposed fused turn to serial.

The host lifecycle is also explicit: `ready -> draining -> stopped`. Liveness and readiness are separate, request IDs remain transport metadata, and reverse-proxy deployment stays loopback-bound so TLS/auth/public exposure remain outside the semantic runtime.


## Evidence-bound runtime qualification (v0.26)

`turbojev-runtime-qualification` is portable and contains only identity/evidence contracts. Host layers may enable a risky optimization only when a passing manifest matches the exact current fingerprint. This keeps deployment probing and filesystem hashing outside the semantic core while preventing a capability validated on one model/backend/hardware target from silently migrating to another.

## Cross-language real-model gate (v0.27-v0.28)

The release harness treats Node/TypeScript, Python, C and .NET/C# as four surfaces over the same native Rust semantics. v0.28 upgrades the gate from per-process validity to full canonical-response parity: every runner emits the complete `EvaluationResponse`, the harness validates choice/noul/score invariants and zero generation, then compares all fields with explicit floating-point tolerance. Real multi-root qualification remains a separate evidence stream bound to the exact model/runtime/backend/hardware fingerprint. The final release report passes only when both evidence streams pass.
