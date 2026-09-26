# TurboJev Core Specification v0.5

## Scope

This specification defines the portable contract shared by TurboJev SDKs,
`turbojev-core`, model/runtime adapters, and compatibility layers.

The contract is intentionally independent of:

- operating system;
- CPU/GPU vendor;
- browser vs native execution;
- model family;
- weight/container format;
- concrete inference library;
- human language.

A concrete runtime can use GGUF, safetensors, ONNX-like artifacts, a browser
engine, a Rust-native executor, or another future open-weight representation.
The core only requires exact candidate scores.

## Request

```json
{
  "model": "optional-model-id",
  "state": "or any non-empty JSON object/array",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which queue should handle this?",
      "criteria": {
        "billing": "Billing or payment issue",
        "support": "Technical support issue"
      }
    },
    "requires_review": {
      "type": "noul",
      "instructions": "Does this require human review?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the incident?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

`boolean` is accepted as a wire alias for `noul`.

## Human-language rule

TurboJev does not translate `state`, `instructions`, choice descriptions, or
score labels. They are preserved verbatim. The portable prompt layer produces a
semantic `PromptDocument`; a model/runtime adapter renders that document using
the instruction/chat format suitable for the selected checkpoint.

This means language support is a property of the selected model and prompt
profile, not a hard-coded assumption in TurboJev core.

## State rules

Portable v0.3 accepts:

- non-empty string;
- non-empty JSON object;
- non-empty JSON array.

`null`, numbers, booleans, and empty state containers are rejected by the native
schema.

## Choice

A choice has 2..256 criteria. Keys are stable application-level values.
Descriptions are semantic text supplied by the application.

```json
{
  "type": "choice",
  "choice": "support",
  "confidence": 0.91,
  "probabilities": {
    "billing": 0.09,
    "support": 0.91
  }
}
```

Probabilities are conditional on the supplied candidates. Unless
`_turbojev.calibrated` is true, they are not claimed to be calibrated correctness
probabilities. The generic `confidence` field is currently the selected candidate
probability, and `_turbojev.confidence_method` names that derivation explicitly.
It must not be confused with a proprietary provider-specific confidence statistic.

## Noul / Boolean

Canonical candidate order is:

```text
false
true
```

Output:

```json
{
  "type": "noul",
  "noul": 0.83
}
```

`noul` is P(true) after the configured calibration layer.

## Score

Score levels are ordered from zero. `score` is the probability-weighted expected
level index.

```json
{
  "type": "score",
  "score": 1.72,
  "confidence": 0.66,
  "probabilities": {
    "0": 0.04,
    "1": 0.20,
    "2": 0.66,
    "3": 0.10
  },
  "legend": {
    "0": "none",
    "1": "low",
    "2": "high",
    "3": "critical"
  }
}
```

## Backend contract

A backend receives:

```text
DecisionBatch
  shared state
  ordered DecisionTask[]

ExecutionPlan
  prefix reuse
  task scheduling
  candidate projection
```

and returns:

```text
BackendOutput
  model identity
  ordered TaskScores[]
  optional metrics
```

Each `TaskScores.logits` vector MUST match the input candidate order.

A backend MUST NOT perform public answer construction or silently alter the
schema. Probability normalization/calibration and typed result construction
belong to `turbojev-core`.

## Capability negotiation

Backends advertise semantic capabilities rather than hardware names:

```text
shared_prefix
parallel_suffixes
selective_output_head
multi_token_candidates
local_inference
```

`turbojev-engine` chooses an execution plan from workload shape and these flags.
The planner never branches on OS, CPU architecture, GPU vendor, or model family.

Current planning policies:

- `Auto`
- `Reference`
- `DeterministicReuse`

The `Reference` path recomputes tasks independently and is retained as a
correctness oracle for optimized paths.

## Candidate scoring invariants

- No natural-language answer generation is required on the optimized path.
- A backend returns exactly one score row per task.
- Candidate score count exactly matches candidate count.
- Candidate logits must be finite before calibration.
- Exact probability ties resolve to the earliest candidate.
- `generated_tokens` is zero for direct scoring.

## Candidate code contract

Model-specific token IDs are forbidden in generic TurboJev logic.

`turbojev-tokenizer` dynamically verifies candidate codes at the exact answer
boundary. It first seeks one-token codes. When there are too few it may select
fixed-width sequences, and runtimes with exact sequence scoring may also accept
variable-length codes when the final selected codebook is unique and prefix-free.

A valid code suffix must satisfy:

```text
encode(prompt_prefix + symbol)
    = encode(prompt_prefix) + suffix_tokens

decode(suffix_tokens)
    = symbol
```

Every multi-token codebook MUST be prefix-free so no candidate is an incomplete
prefix of another candidate. A backend must advertise exact sequence scoring
before such a codebook may be executed. Single-token codes remain the preferred
fast path.

## Prompt contract

`turbojev-prompt` no longer contains a mandatory English system prompt.

The portable layer compiles:

```text
PromptDocument
  evidence
  criterion
  option { code, key, description }[]
  contract = turbojev.prompt-ir.v2
```

A `PromptRenderer` owned by the model/runtime adapter converts that semantic IR
into model-facing messages. Specialized checkpoints may pin an exact renderer
profile.

## Model contract

`turbojev-model` stores portable descriptors only. `ModelSource` is opaque and
never triggers I/O in the core.

```text
ModelSpec
  id
  source
  architecture?       free-form
  format?             free-form
  tokenizer?
  prompt_profile?
  revision?
```

TurboJev core has no architecture allowlist. A new model family requires only a
runtime adapter capable of satisfying the scoring contract.

## Calibration contract

Raw backend logits pass through one core-owned `Calibrator`.

Current implementations:

- `SoftmaxCalibrator`
- `TemperatureCalibrator`

Training a decision model and calibrating its probability scale remain separate
operations.

## Wire contract

`turbojev-wire` is the canonical UTF-8 JSON boundary for future Node,
TypeScript/browser, Python, .NET, Swift, Kotlin, JVM, Dart, C ABI, and other host
bindings. Bindings must not reimplement schema validation or answer semantics.

## Compatibility layers

Compatibility is layered above the canonical core and implemented by
`turbojev-compat`. Current profiles are:

1. `turbojev` — canonical native contract;
2. `typesafe-systemone-v1` — `choice` / `noul` / `score`;
3. `ai-sdk-evaluation-v4` — `choice` / `boolean` / `score`, with boolean answers
   exposed as `{ probability }`.

Choice/Score confidence can be mirrored into AI SDK provider metadata for source
compatibility. TurboJev simultaneously emits provenance metadata describing how
that value was derived. A compatibility adapter may rename or omit metadata fields
where required, but must never change the underlying semantic decision silently.

OpenRouter Decisions remains a future wire profile, not a dependency of the core.

## Model specialization contract

Two model classes share one API:

- **generic model**: frozen open-weight model scored directly;
- **decision-specialized model**: base checkpoint plus candidate-logit training.

A specialized model contract will pin at least:

```text
base model + revision
prompt IR + renderer profile
tokenizer revision
candidate encoding policy
training objective
maximum context
calibration profile
```

A runtime must reject incompatible model/prompt/tokenizer combinations instead
of silently falling back.

## Portability contract

Portable crates do not directly use filesystems, sockets, subprocesses,
environment variables, OS-specific APIs, browser APIs, or hardware APIs.
Concrete runtimes and language bindings live behind adapters. See
`docs/PORTABILITY.md` and ADR 0004.


## Conformance contract

`turbojev-conformance` compares complete canonical responses rather than backend
internals. This allows any runtime and platform to participate. A parity report
separates:

- discrete decision agreement;
- maximum candidate-probability delta;
- expected-score delta.

Optimized paths such as shared-prefix reuse, batched suffixes, quantization or a
selective output head must be benchmarked against a reference path with explicit
tolerances instead of assuming bitwise equality.


## Universal Runtime Bridge contract (v0.4)

TurboJev's portable semantic core MUST NOT depend on a model container, accelerator API, operating system, browser API, or inference framework. Concrete runtimes integrate through `turbojev-runtime`.

A runtime integration is decomposed into three independently testable components:

1. `PromptRenderer`: semantic `PromptDocument` -> model-facing messages.
2. `PromptCodec`: messages -> exact chat-template/tokenizer token stream, including an API to encode the same prompt with a complete candidate answer.
3. `RuntimeExecutor`: prepared token prefix/suffixes + candidate token sequences -> raw candidate logits/log-scores.

`RuntimeBackend` owns the portable glue: candidate-code search, exact answer-boundary verification, single-token preference, multi-token fallback, prefix-free validation, exact common-token-prefix factoring, and conversion back to the stable `ScoringBackend` output.

### Runtime neutrality

The contract has no enum of known model architectures and no enum of known weight formats. An adapter MAY execute GGUF, safetensors, ONNX, MLX artifacts, a memory-owned checkpoint, a browser-resident model, a mobile NPU graph, or any future format. The adapter is responsible only for satisfying the token-level contract.

### Candidate scoring

Single-token candidate codes are preferred. If no compatible single-token code set exists, multi-token candidate sequences MAY be used only when `RuntimeCapabilities.sequence_scores` is true. Every selected code MUST round-trip through the runtime tokenizer at the exact model answer boundary and candidate token sequences MUST be unique and prefix-free.

### Shared-state reuse

When the execution plan requests shared prefix reuse, TurboJev factors only the literal longest common token prefix of the fully rendered model prompts. A runtime MUST NOT assume a string-level common prefix or silently alter tokenization.


### Threading model

Portable backend, renderer, codec and runtime traits MUST NOT require `Send` or `Sync`. Thread-affine browser/mobile runtimes are valid TurboJev integrations. Hosts that need cross-thread sharing MAY require the `ThreadSafeScoringBackend` marker.

## Async execution contract (v0.5)

Portable TurboJev MUST support synchronous and asynchronous runtimes without
requiring a particular scheduler. `ScoringBackend`/`RuntimeExecutor` define the
synchronous path. `AsyncScoringBackend`/`AsyncRuntimeExecutor` define the async
path. The canonical schema, planner, calibration, answer construction, and wire
compatibility MUST remain identical between both paths.

Portable async futures MUST NOT require `Send`. Outer server/binding crates MAY
impose stronger thread-safety requirements. Portable crates MUST NOT depend on
Tokio, async-std, browser APIs, JNI, Objective-C, or an OS scheduler merely to
provide async execution.

## Runtime conformance contract (v0.5)

A concrete runtime adapter is conforming only when its adapter-owned test suite
uses `turbojev-runtime-conformance` (or equivalent stricter checks) to establish:

1. exact prompt answer-boundary extension for every candidate code;
2. decode round-trip, uniqueness, and prefix-free multi-token codebooks;
3. execution-plan compatibility with declared capabilities;
4. one finite score per candidate and exactly one score row per task;
5. explicit numerical parity bounds against a reference execution path for any
   optimized shared-prefix, batched, quantized, or selective-head path.

No conformance test may special-case a model family in `turbojev-core`.

## Deterministic shared-prefix contract (v0.7)

A concrete runtime MAY advertise `shared_prefix = true` only after it provides
an execution path that preserves the semantic result of the independent
reference path within an explicit conformance tolerance.

For the llama.cpp adapter, shared-prefix execution is defined as follows:

1. TurboJev factors the literal longest common **token** prefix after full prompt
   rendering and tokenization.
2. The runtime prefills that prefix exactly once into an immutable base sequence.
3. Every task suffix is evaluated from an isolated branch copied from that base.
4. The branch is discarded after scoring and MUST NOT mutate the base or another
   task branch.
5. A zero-length suffix is valid and MUST use the logits at the final prefetched
   position rather than appending a synthetic token.
6. If no non-empty token prefix exists, correctness takes priority and execution
   falls back to the independent path.
7. Candidate probabilities remain core-owned; the optimized runtime returns raw
   logits in the same candidate order as the reference path.

`PlanningPolicy::Reference` MUST select `PrefixReuse::None`, serial scheduling,
and full-vocabulary projection even when a backend advertises additional
optimizations. This makes the reference path stable as later phases add parallel
suffixes and selective output projection.

## Bounded parallel suffix contract (v0.8)

A concrete runtime MAY advertise `parallel_suffixes = true` only when parallel
branches are semantically independent and are checked against the reference path.
TurboJev treats parallelism as a bounded execution property, never as an
unbounded request to the accelerator.

1. Parallel suffix execution MUST use the exact same rendered/tokenized tasks as
   serial shared-prefix execution.
2. The immutable shared prefix MUST NOT be mutated by any task branch.
3. Every simultaneously evaluated suffix MUST have an isolated runtime branch or
   sequence identity.
4. Branch state MUST be discarded before the next bounded batch is started.
5. Result rows MUST preserve request task order regardless of native batch order.
6. Empty suffixes MUST use the retained prefix logits and MUST NOT insert a
   synthetic token.
7. `ExecutionPlan::parallel_task_limit` MUST be at least two for parallel plans;
   serial plans MUST set it to one.
8. Automatic scheduling MUST bound the parallel limit by the task count, caller
   `ExecutionBudget`, backend `max_parallel_tasks`, and—when both sides provide
   values—the caller memory budget divided by the backend's conservative
   per-branch memory estimate.
9. If the resulting safe limit is less than two, Auto MUST fall back to serial
   execution without disabling valid shared-prefix reuse.
10. Portable planning MUST NOT query or name an operating system, accelerator
    vendor/API, device family, or model architecture.
11. Parallel output MUST satisfy the same explicit numerical parity tolerance as
    cached serial output before the runtime advertises the capability.

The native response contract is `turbojev.core.v4`; execution metadata includes
`parallel_task_limit` so callers can audit the actual bounded plan selected by
TurboJev.


## Candidate-only projection contract (v0.9)

`CandidateProjection::CandidatesOnly` has a strict computational meaning: the
runtime projects only the output-head rows required by the declared candidate
tokens. Computing the complete vocabulary head and gathering afterward is
semantically valid for the reference path but is not selective projection.

A runtime advertising `selective_output_head = true` MUST:

- produce candidate logits in exactly request order;
- preserve the same model output scale/bias semantics as the ordinary LM head;
- report `BackendMetrics.candidate_rows_projected`;
- report exactly one projected row per single-token candidate occurrence;
- pass reference/full-vocabulary parity at an explicitly declared tolerance;
- never treat a multi-token candidate code as one output-head row.

`PlanningPolicy::Reference` MUST continue to request `FullVocabulary` even if
the backend advertises candidate-only support. Automatic planning MAY select
`CandidatesOnly` only from a truthful backend capability.

The portable reference math is defined by `turbojev-projection`; concrete GPU,
WebGPU, NPU or other accelerators MAY fuse this operation but MUST preserve the
same observable logits and conformance contract.


## Exact multi-token sequence scoring contract (v0.15)

A multi-token candidate score is the exact conditional log-likelihood of the
opaque candidate token sequence at the model answer boundary. For candidate
`c = [c_1, ..., c_n]`:

```text
score(c) = Σ_i ( logit_i(c_i) - logsumexp(logits_i[full vocabulary]) )
```

The following are normative:

1. Raw next-token logits MUST NOT be summed directly as sequence scores.
2. Normalization MUST use the complete model vocabulary at every continuation
   state, or a mathematically equivalent exact normalizer supplied by the runtime.
3. Candidate codes MUST be non-empty, unique and prefix-free.
4. A trie MAY reuse candidate-prefix inference, but each unique prefix state must
   be semantically identical to independent continuation from the same prompt.
5. Single-token execution remains the fast path and MUST preserve its existing
   logits/probability distribution. For a single-token codebook, sequence-log-score
   differences must equal raw candidate-logit differences because the common
   vocabulary normalizer cancels.
6. A plan requesting `CandidatesOnly` MUST be downgraded to `FullVocabulary` for
   exact multi-token scoring unless the runtime explicitly exposes an exact
   full-vocabulary normalizer (`logZ`) for every continuation state.
7. v0.15 serializes task-level multi-token traversal while retaining valid
   shared-state prefix reuse. A runtime MUST report the actual plan through
   `BackendMetrics.effective_plan` when this differs from the requested plan.
8. `BackendMetrics.sequence_candidate_tokens`,
   `sequence_prefix_evaluations`, and `sequence_token_steps_saved` report actual
   sequence-scoring work and MUST NOT be fabricated by a single-token path.

The portable reference implementation lives in `turbojev-sequence`.


## Node host binding contract (v0.10)

The Node binding is an outer host adapter, not a portable-core dependency.
`@turbojev/node` exposes a Promise-based API while the Node-API addon exposes a
small synchronous JSON boundary. One dedicated worker owns each native session.

Native exports:

```text
openEngine(optionsJson) -> sessionId
closeEngine(sessionId)
evaluateJson(sessionId, requestJson) -> canonical response JSON
evaluateJson(sessionId, requestJson) -> evaluation JSON
evaluateCompatJson(sessionId, profile, requestJson) -> profile response JSON
engineInfoJson(sessionId) -> backend/capability JSON
```

The JavaScript layer MUST NOT perform schema validation, calibration, candidate
selection, or typed answer construction. It MAY own host concerns such as
filesystem paths, remote downloads, cache directories, worker lifecycle, and
native-addon discovery.

Public errors are normalized to `TurboJevError { code, message }`; raw N-API
status codes are not part of the compatibility contract.


## Browser/WASM host binding contract (v0.11)

The browser binding is an outer host adapter. Browser APIs, workers, CacheStorage,
networking and concrete WebGPU/runtime handles MUST NOT enter portable Rust
crates.

`@turbojev/web` communicates with `turbojev-wasm` in a module Worker. The WASM
binding composes the same `AsyncRuntimeBackend` used by other asynchronous hosts.
A JavaScript runtime module is therefore model-facing, not semantic.

Every browser runtime MUST expose:

```text
initialize(init) -> BrowserRuntimeDescriptor
encodePrompt(renderedMessagesJson) -> token-id JSON
encodePromptWithAnswer(renderedMessagesJson, answer) -> token-id JSON
decodeTokens(token-id JSON) -> text
scoreRuntimeBatch(runtimeBatchJson, executionPlanJson) -> RuntimeOutput JSON
close()
```

TurboJev WASM MUST retain ownership of candidate-code discovery, exact
answer-boundary validation, prefix factoring, execution planning, calibration,
compatibility profiles and typed answer construction. A runtime MUST NOT require
hard-coded candidate token IDs in generic mode.

The model resolver has two legitimate ownership modes:

- `browser-cache`: `@turbojev/web` resolves/fetches the resource and may cache a
  streamed response in CacheStorage;
- `runtime`: the model string is preserved as an opaque reference for runtimes
  that own repository/file/cache resolution themselves.

Cache population MUST NOT delay consumption of an otherwise usable remote model
response. Cache errors are non-fatal when the runtime still has a valid response.


## Transformers.js browser runtime contract (v0.12)

`@turbojev/runtime-transformersjs` is an optional concrete implementation of the
browser runtime contract. It is not a dependency of `turbojev-core`,
`turbojev-runtime` or any other portable crate.

The v0.12 adapter MUST:

- require `modelResolution = runtime` and pass the opaque model reference to
  Transformers.js;
- use the checkpoint tokenizer/chat template for the assistant generation
  boundary;
- tokenize `boundary + candidate` and rely on Rust's prefix/round-trip checks
  rather than model-specific candidate IDs;
- run a causal-LM forward pass and return raw candidate logits;
- release temporary input/logit tensors when the runtime exposes disposal;
- release model resources on `close()`;
- reject execution plans or candidate codebooks it does not implement.

Its truthful v0.12 capability set is:

```text
next_token_logits      = true
sequence_scores        = false
shared_prefix          = false
parallel_suffixes      = false
selective_output_head  = false
local_inference        = true
```

An implementation MAY request only the final position's vocabulary logits when
the loaded model supports that optimization. This does not satisfy
`selective_output_head`: candidate-only projection means skipping the complete
vocabulary projection, not merely reducing the number of sequence positions for
which that vocabulary projection is produced.

Model-format/device compatibility is an adapter concern. A Transformers.js
checkpoint, GGUF runtime, native safetensors runtime or mobile NPU runtime can
all coexist behind the same TurboJev semantic contract without becoming a
portable-core dependency.


## Shared native host + C/Python/.NET contract (v0.13)

Native host-language bindings MUST reuse `turbojev-host-native` rather than copy
model-open, prompt, planning, execution-budget or compatibility logic into each
language adapter.

`NativeSession` is thread-affine and exposes only canonical JSON operations plus
engine metadata. A build without a concrete native runtime MUST retain the same
host API and fail engine creation with `RUNTIME_UNAVAILABLE` rather than changing
the binding contract.

### C ABI

`turbojev-capi` exports an opaque `tj_engine*`. No Rust struct layout, enum layout
or allocator-owned buffer is part of the ABI. Inputs are NUL-terminated UTF-8;
JSON outputs are TurboJev-owned strings released only by `tj_string_free`.

All calls on one engine, including close, MUST occur on the same native thread.
The ABI reports this with `tj_engine_is_thread_affine()`. Language bindings whose
schedulers can move continuations between threads MUST introduce a host-owned
dedicated thread/worker rather than requiring `Send + Sync` from the runtime.

Every exported C operation MUST catch unwinding panics at the ABI boundary.
Failures return a compact integer status while stable semantic error code/message
strings are exposed through thread-local `tj_last_error_code()` and
`tj_last_error_message()` borrowed pointers.

### Python

The PyO3 class MUST be unsendable/thread-affine and delegate every inference
operation to `NativeSession`. The pure Python layer MAY serialize ordinary
Python mappings/sequences, normalize errors and own context-manager lifetime; it
MUST NOT implement candidate scoring, probability calibration, schema rules or
compatibility semantics.

### .NET

The .NET package consumes the stable C ABI through P/Invoke. The managed engine
MUST schedule open/evaluate/info/close onto one dedicated managed
thread. Public Task continuations MAY resume anywhere; native handle calls may
not. The managed layer owns only JSON transport, P/Invoke lifetime and error
mapping.


## Calibration profile contract (v0.16)

TurboJev distinguishes a conditional candidate distribution from calibrated confidence. `SoftmaxCalibrator` at T=1 remains unfitted. A fitted temperature profile must use the `turbojev.calibration-profile.v1` contract and a strictly positive finite scalar temperature.

Temperature fitting is performed only on a separately identified holdout dataset using weighted multiclass NLL. The profile must record sample count, ECE bin count, before/after NLL, multiclass Brier, top-label ECE and accuracy. Model/tokenizer/prompt/renderer provenance should be pinned whenever available.

Because every candidate score is divided by the same `T > 0`, temperature scaling must not change candidate ordering or typed answers; it changes only the reported probability distribution. Filesystem and CLI behavior are forbidden from the portable `turbojev-calibration` crate and live in `turbojev-calibrate`.


## Specialized model contracts (v0.17)

Decision-specialized checkpoints use `turbojev.specialized-model.v1` and are trained from `turbojev.specialized-training-recipe.v1`. The training objective is direct candidate-logit cross-entropy; no generated JSON/prose target is required. Recipes pin the base-model revision, tokenizer revision, prompt IR, renderer profile, candidate encoding, context limit, LoRA/QLoRA configuration, deterministic debiasing policy and data split identities.

The four split ids `train`, `validation`, `calibration` and `benchmark` must be pairwise distinct. Calibration remains a separate held-out stage. A specialized manifest embeds `turbojev.decision-model.v1`; `ModelSpec.decision_contract` remains optional so Generic Mode is unchanged. Generic-vs-specialized evaluation uses one named benchmark suite and explicitly records accuracy, NLL, Brier, ECE, latency, peak memory, decisions/sec, candidate-order robustness and paraphrase robustness.
