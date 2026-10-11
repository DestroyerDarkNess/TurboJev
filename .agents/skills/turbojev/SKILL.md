---
name: turbojev
description: Use when installing, integrating, designing decisions for, or troubleshooting TurboJev in Rust, Node.js/TypeScript, Python, C, .NET, browser/WASM, Android, or Swift applications.
---

# TurboJev

Help build applications with TurboJev's typed decision and structured workflow APIs. This guide describes the 0.34.0 published SDK: `evaluate`, `generate`, balanced choice ordering, field dependencies, conditional omission, probability wrappers and bounded reasoning. Use the repository's current source and docs as the authority: package versions, platform assets, and runtime capabilities change. Do not repeat old README version numbers or claim validation that has not been run.

This skill is self-contained: installation, contracts, options and limitations are included below. No external Markdown is required to follow it.

## First establish the integration

Identify the host language, target OS/architecture, model format, and intended inference runtime. TurboJev has a portable decision core and separate runtime adapters. The core does not load every model by itself. A checkpoint is usable only when the selected runtime supports its architecture, tokenizer, prompt template, and required modality.

## Install by language

Native 0.34.0 packages cover Windows and Linux x86/x64/ARM64, plus macOS x64/ARM64. Linux requires glibc 2.28+ (Alpine/musl requires a source build); macOS requires 11+. Browser WASM is architecture independent. NuGet bundles all eight runtime directories; pip and npm select the running interpreter architecture. macOS artifacts are cross-compiled and link-checked, without claiming Mac execution. Android/iOS remain source integrations. These packages expose TurboJev's typed-decision API; they do not all include the same model runtime.

| Host                            | Install                                                                                               | What it provides                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Rust                            | `cargo add turbojev`                                                                                  | Portable typed-decision API. For GGUF, add `cargo add turbojev-runtime-llamacpp --features ffi`. |
| Node.js / TypeScript            | `npm install @turbojev/node`                                                                          | Worker-backed native binding.                                                                    |
| Python                          | `python -m pip install turbojev`                                                                      | Context-managed native binding.                                                                  |
| C# / .NET                       | `dotnet add package TurboJev`                                                                         | .NET binding over the stable C ABI.                                                              |
| C                               | Build native artifacts, include `bindings/c/include/turbojev.h`, and link the matching C ABI library. | Stable UTF-8 JSON ABI.                                                                           |
| Browser JavaScript / TypeScript | `npm install @turbojev/web @turbojev/wasm @turbojev/runtime-llamacpp-wasm`                            | Browser host, worker, and WASM integration.                                                      |
| Android / Kotlin                | Include the Android binding source/module and package matching native libraries.                      | JNI wrapper over the C ABI.                                                                      |
| Swift / iOS / macOS             | Add the local Swift package at `bindings/swift` and package the matching C ABI library.               | Swift wrapper over the C ABI.                                                                    |

For native samples, build artifacts for the same architecture as the host compiler/interpreter. The samples builder takes one architecture per run (for example, `--arch x64`); it does not mean “all architectures.” C consumers can start with `cargo build -p turbojev-capi --release`; enable the `llamacpp` feature for a real GGUF C ABI build (`cargo build -p turbojev-capi --release --features llamacpp`). A fixture-only library cannot load a real model.

## Binding compatibility requirements

- **Python:** CPython 3.9+ with the normal GIL-enabled stable ABI (`cp39-abi3`).
  Wheels have separate Windows/Linux x86/x64/ARM64 and macOS x64/ARM64 tags; use the interpreter's
  architecture. PyPy and free-threaded interpreters are not qualified. Version 0.33.1 fixed the 0.33.0 import failure on Python 3.9 in the benchmark helper and
  makes public type annotations introspectable there. The corrected x64 wheel
  was tested on CPython 3.9.13, 3.12.7 and 3.14.0; the published x86 binary
  passed contract checks on CPython 3.12.10. ARM64 DLL architecture was checked,
  but ARM64 execution was not tested.
- **Node:** Node 18+ for the full wrapper, including URL model downloads that
  use `fetch` and Web Streams. The native addon uses Node-API 8 rather than a
  Node-major-specific ABI. The same x64 addon passed contract checks on Node
  18.20.8 and 24.18.0; the x86 addon passed on Node 22.18.0. The package is ESM:
  use `import`, `.mjs`, or dynamic `import()` from CommonJS. Native ARM64
  execution remains untested.
- **TypeScript:** the generated Node/browser declarations were checked with
  TypeScript 4.7.4 and the repository uses 5.9.2 to build. A modern TypeScript
  compiler is a build tool requirement, not a different inference ABI. NodeNext
  consumers should use ESM configuration or `.mts`.
- **C/C++:** the header uses fixed-width C99 types, opaque pointers and C linkage.
  C99/C11/C17 and C++11/C++17 consumers passed native calls on Windows x64;
  MSVC C11 also passed on x86. Build or obtain native libraries matching the
  application's architecture. The matching NuGet runtime directory supplies desktop C ABI libraries and
  their dependencies for Windows, Linux and macOS. Android/iOS need dedicated builds.
- **Rust:** the workspace declares Rust 1.82 / edition 2021. The locked native
  binding crates passed `cargo check` with Rust 1.82. GGUF remains an optional
  runtime feature and requires the shim and its dependencies when linking.
- **Browser/WASM:** the host needs modern ESM, module workers and WebAssembly.
  Multithreaded and audio GGUF paths also need cross-origin isolation and
  `SharedArrayBuffer`; GPU requires WebGPU. Device/backend errors must remain
  explicit. Browser adapter contract tests do not qualify every browser or GPU.
- **Android/Kotlin:** the source wrapper now declares API 24+ / Java 8 bytecode;
  `CompletableFuture` sets that runtime floor. Kotlin 2.0.21 is the repository's
  build compiler. Supply JNI libraries and model-runtime dependencies for each
  Android ABI. The wrapper compiled against the Android SDK, but full Gradle
  validation was blocked by a local JVM loopback error; no device execution was
  performed. There is no bundled published AAR.
- **Swift:** the source manifest now requires tools 5.7+, iOS 12+ and macOS
  10.13+. Swift Package Manager can resolve the v0.34.0 Git tag. These are
  wrapper deployment settings, not promises about the separately built model
  runtime or Metal backend. Supply a matching Apple C ABI library; no bundled
  XCFramework is published.
  Native errors must be read on the thread that performed the call. Compilation
  and execution of the lower Apple deployment targets remain pending.

Compatibility fixes above are included in published 0.34.0. The v0.34.0 Git tag
provides the source package for Swift Package Manager and the Android module.
Windows ARM64 binaries were cross-compiled and checked, not executed on ARM64.

## Choose the operation

- `evaluate`: `state` plus typed `choice`, `noul` and ordered `score` questions; read `response.answers`.
- `generate`: `context` plus typed `questions` or an object `schema`; read `response.result`, `skipped` and `usage`. Finite enums score candidates; open strings/numbers generate bounded tokens.
- `evaluateCompatible`: an explicit compatibility profile and evaluation request.

A numeric enum is unordered; an ordered score is the expected zero-based index of the original criteria list. Do not interchange these contracts.

## Request and decision model

Use `evaluate` for choices, boolean gates (`noul`), ordered scores, and multiple typed questions over one state. Do not use the removed `classify` operation. A normal request has `state` and `questions`; each question specifies `type`, task-specific `instructions`, and, for `choice`, a `criteria` map from stable answer keys to meaningful descriptions.

```json
{
  "state": "A customer says the package arrived broken and asks for a replacement.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices, or refunds",
        "shipping": "Delivery, damage, or replacement",
        "technical": "Product or application problems"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this require immediate human intervention?"
    },
    "quality": {
      "type": "score",
      "instructions": "How complete is the proposed resolution?",
      "criteria": ["Incomplete", "Partly complete", "Complete"]
    }
  }
}
```

Response probabilities are relative candidate scores, not calibrated confidence by default. Only describe them as calibrated when a matching calibration profile was fitted and applied. Keep the model's choice, host policy, and calibration provenance distinct.

The built-in candidate-logit adapters request a structured answer and insert the literal `{"answer": "` before scoring its option code. This prefill is supplied prompt text, not generated JSON or a chosen option. Codebook probes must include the identical prefill and verify the candidate tokens at that continuation point. Reading option-letter logits at the beginning of an unconstrained assistant message can introduce a strong first-letter bias. Custom codecs must honor `RenderedMessages.assistant_prefix`; its legacy default is empty.

Do not confuse removing positional bias with solving model reasoning. In a real Windows Qwen3-0.6B Q4_K_M audit, five simpler tasks passed all four candidate rotations (20/20), while a Snake wall state expressed in raw coordinates failed all four (0/4), selecting `up` regardless of its position. The same GGUF and Wllama 3.8.1 binary passed four replacement-routing rotations in Node CPU WASM. These are specific measurements, not guarantees for other models, Snake navigation, browsers, or WebGPU.

## Design decisions models can answer

- State the relevant relationship in plain language when a small model would otherwise need to infer it from raw JSON, coordinates, or arithmetic. Structured state is valid; make its decision-relevant meaning explicit too.
- Keep prompts compact and avoid repeating the same facts in the state, instructions, and criteria. More prompt tokens can increase prefill latency; remove only redundant text, then remeasure decision quality and latency.
- Ask for one bounded decision at a time. Do not expect a small checkpoint to plan a long sequence of actions just because the API accepts a complex state.
- Give candidates meaningful keys and descriptions that expose the distinction being scored. Opaque keys such as `A/B/C` add a mapping the model must infer.
- Production applications may constrain available actions in host code, but disclose that assistance. A demo claiming model-controlled actions must execute the returned choice directly; do not replace it with a heuristic, probability fallback, tie-break, or forced action and count it as a model decision.
- Test for order sensitivity by permuting candidates. A model that repeatedly chooses the first candidate is order-biased, even when one fixed ordering appears accurate. `orderRobustness: "balanced"` uses canonical layouts and averages probabilities; `cyclic` uses rotations and averages logits. Both cost real model evaluations and neither guarantees correctness. Keep single-pass mode by default unless reproducible tests justify the added latency.
- Validate with the exact model, quantization, runtime, and prompt path intended for deployment. Use reproducible cases and a simple reference policy where possible; report both decision quality and latency.

For model-controlled games, disclose whether the state is raw coordinates or derived sensor observations. Sensor facts can simplify perception without choosing the action. Keep the returned choice separate from host legality policy; never substitute a path, heuristic move or forced action and count it as an AI decision. Log the complete request, response, selected/applied action and outcome. Small successful games do not establish general navigation reliability.

## Choose the right runtime

- **Native text inference:** use the llama.cpp adapter for GGUF models it can load. CPU is the normal default. GPU is opt-in; an unsupported GPU request should surface the runtime error, not silently change devices.
- **Published GPU assets:** Windows x64 includes Vulkan; macOS x64/ARM64 include Metal with embedded kernels. Linux and Windows x86/ARM64 packages use CPU inference. Explicit GPU selection does not silently fall back to CPU. Architecture coverage does not imply GPU coverage on every target.
- **Node/TypeScript, Python, C, and .NET:** these bindings expose the shared native host. Their package or binary must match the target OS, CPU architecture, and language ABI. Check actual release assets before promising x86, x64, or ARM64 support.
- **Rust:** compose the core crate with the runtime adapter the application selects; adding the core crate alone does not provide GGUF inference.
- **Browser/WASM:** choose a supported browser runtime, such as Transformers.js with ONNX or the llama.cpp WASM path with compatible GGUF. Browser hardware, memory, model-format, tokenizer, and template constraints apply.
- **Android and Swift:** these bindings call native code through JNI/C ABI and need the corresponding host libraries packaged for the app. Do not present them as registry packages unless the current repository/release explicitly provides one.
- **Images, audio, and video:** the media envelope is a transport contract, not an inference model. Select a multimodal checkpoint and runtime that implement preprocessing and candidate scoring for that modality.

Browser runtime choices are separate integrations: Transformers.js for supported ONNX models, llama.cpp WebAssembly for GGUF models supported by that WASM build, or a custom implementation of the browser runtime interface. WebGPU must be requested explicitly; expose CPU/WASM where available and report unsupported-device errors honestly. Browser memory, hardware, model format, tokenizer, and template support can differ from the native host.

TurboJev's media envelope only standardizes how evidence enters `evaluate`; it does not include a vision/audio model, processor, or inference runtime. Verify that both the selected checkpoint and adapter support the modality and task. A GGUF filename alone does not guarantee compatibility.

TurboJev supports routing, typed gates, ordered rubric scoring, scalar extraction and conditional field workflows. Open scalar generation is supported by `generate`; this does not promise unrestricted chat, path planning or correctness for every model/runtime combination.

## Candidate ordering in evaluate

Native loads accept `execution={"orderRobustness":"balanced"}`; browser loads
accept `orderRobustness: "balanced"`. Rust uses `OrderRobustness::Balanced`.
The default is still `none`, one pass. Balanced mode builds canonical Williams
layouts from stable candidate keys, keeps the whole set visible, calibrates each
layout once and averages aligned probabilities. It costs K passes for even K,
2K for odd K, with at most 32 candidates. `cyclic` retains cyclic raw-logit
averaging. Both leave ordered `score` scales and nouls untouched. Exact choice
ties use lexicographic semantic keys; enum ties use their JSON value encoding.
Unequal probabilities are never treated as ties. Robust layouts cannot guarantee
correct model reasoning or calibrated confidence. Integer permutation sampling
in `generate` still includes the original listing; use `auto` for canonical
balanced layouts. Fit calibration against the actual production policy.

Score training data retains the original scale and only varies output codes;
regenerate previously materialized score data before additional fine-tuning.

### Host configuration

```python
with TurboJev.load("model.gguf", execution={"orderRobustness": "balanced"}) as jev:
    response = jev.evaluate(request)
```

```typescript
const jev = await TurboJev.load({
  model: "model.gguf",
  execution: { orderRobustness: "balanced" },
});
// Browser: TurboJevWeb.load({...options, orderRobustness: "balanced"})
```

```csharp
await using var jev = await TurboJevEngine.LoadAsync(new {
    model = "model.gguf", execution = new { orderRobustness = "balanced" }
});
```

Rust uses `engine.with_order_robustness(turbojev::OrderRobustness::Balanced)`. C, Android and Swift use native load JSON: `{"model":"model.gguf","execution":{"orderRobustness":"balanced"}}`. Browser `orderRobustness` overrides the legacy `cyclicOrderRobustness` flag. The corrected core limit is 32 choices; browser finite scoring has a control-label limit of 26. Four balanced choices cost four layouts; three cost six. Fit calibration to the selected aggregation policy. In default mode `order_evaluations` remains legacy zero, not a count of actual calls.

## Troubleshooting

1. Reproduce with the same request, model file, quantization, runtime, and device.
2. Separate model-load errors from evaluation errors; preserve the complete native/runtime error and check model architecture, tokenizer/template compatibility, file path, and platform assets.
3. Inspect candidate descriptions, candidate order, and the returned distribution. A high score on one candidate is not a guarantee of a correct semantic answer.
4. Compare none, cyclic and balanced order scoring only when order bias is suspected, and include the extra latency in the result.
5. Change the smallest responsible layer: request design for ambiguous context, host code for legality, runtime adapter for execution support, or TurboJev core only for a demonstrated shared semantic defect.

Do not claim all models are supported merely because they use GGUF, or all hardware/architectures are available because a language binding exists. State the tested model, runtime, platform, and limitations.

## Optional native model reasoning and measured limitations

Native load JSON accepts `thinking: false` by default. Set `thinking: true` and
`maxThinkingTokens: 512` to generate model reasoning before the typed decision.
Python uses `TurboJev.load(model, thinking=True, max_thinking_tokens=512)`;
Node uses `TurboJev.load({model, thinking: true, maxThinkingTokens: 512})`;
.NET passes those camelCase fields in the object given to
`TurboJevEngine.LoadAsync` and evaluates with `EvaluateJsonAsync`.
C, Android and Swift pass the same load JSON. Rust uses
`FfiLlamaCpp::open(model, config)?.with_thinking(true, 512)` with suitable context
and sequence capacity. These options are available in the current release; verify the installed version when troubleshooting older applications.

The native adapter uses the actual model Jinja reasoning template and its
closing delimiters, then verifies candidate tokens after the literal JSON
answer prefill. It does not invent a thought or choose a direction in the host.
`usage.generated_tokens` counts actual model generation, summed across questions
and corrected order evaluations; direct scoring stays zero. Shared/parallel/fused scoring
is disabled for thinking, and the native host limits the context to at most four
sequences. Increase `nCtx` or reduce `nSeqMax` to make room for longer prompts and
thoughts. `THINKING_UNSUPPORTED`, `THINKING_CONTEXT_LIMIT` and
`THINKING_BUDGET_EXHAUSTED` are real failures; do not hide them or fabricate an
answer. Current media, worker and browser adapters reject this mode explicitly.
The current native shim ABI is 10; rebuild native dependents together.

Measured reasoning is not a guarantee of task quality. The recorded small-model probes include raw-coordinate Snake failures and unfinished reasoning at a 512-token budget. A trained decision head, adapted weights or a benchmark-specific checkpoint is a different model; loading a generic base checkpoint does not reproduce its reported results. Keep measured evidence separate from capability claims.

## JSON-typed fields and workflows

`generate` adds JSON-typed fields to TurboJev's portable core. `evaluate` remains the direct `choice`, `noul` and ordered `score` API. Finite fields score model candidates; unrestricted strings and numbers generate a bounded value. Extraction therefore generates tokens and can take longer than a single candidate decision.

### Request and response

```python
from turbojev import TurboJev

with TurboJev.load("invoice.gguf", device="cpu", n_ctx=8192, n_seq_max=1) as client:
    response = client.generate(
        context="Invoice total: 324.50 GBP. Payment is overdue.",
        questions={
            "total": {"type": "number", "instructions": "What is the total in GBP?"},
            "overdue": {"type": "boolean", "instructions": "Is payment overdue?"},
            "reminder": {"type": "string", "when": {"overdue": True},
                         "depends_on": ["total"], "instructions": "Write one short payment reminder."},
        },
        thinking=False,
        text_max_tokens=128,
        numeric_max_tokens=64,
    )
    print(response["result"])
    print(response["skipped"], response["usage"])
```

The same JSON request is accepted by every binding. Supply exactly one of `questions` (a field map) or `schema` (an object with `properties`, optional `required`, and optional `additionalProperties: false`). All declared properties are requested; `required` validates property names, not a separate omission policy. Conditions control omissions. The schema subset supports scalars, nested objects and top-level arrays; nested arrays and arrays inside objects are rejected. `state` is an input alias for `context`.

| Field option                   | Behavior                                                                          |
| ------------------------------ | --------------------------------------------------------------------------------- |
| `type`                         | `string`, `integer`, `number`, `boolean`, or `[type, "null"]`; retains JSON types |
| `enum`                         | 1–512 unique values of that type; a singleton is a constant with no model scoring |
| `instructions` / `description` | Question; description is the fallback                                             |
| `depends_on`                   | Names of earlier results to include as plain values; forward references allowed   |
| `when`                         | AND of tests on prior fields; adds implicit dependencies                          |
| `return_probabilities`         | Boolean, finite fields only; returns a value/confidence/distribution wrapper      |
| `permutations`                 | Finite booleans/enums: positive integer, `auto`, or `all`                         |
| `thinking`                     | Per-field override: false, true, or `auto`                                        |
| `thinking_effort`              | `none`, `low` (512 tokens), `medium` (2048), `high` (4096)                        |
| `thinking_budget`              | Positive token budget, up to 32768; alternative to effort                         |

Token limits are request-level `text_max_tokens` (default 256) and `numeric_max_tokens` (default 64), each 1–32768. Booleans default to `[true, false]`, plus actual null for nullable booleans. Numeric equality distinguishes booleans from integers and does not round distinct 64-bit integer values. Unsupported keywords, including bounds, string length/format constraints and unsupported container combinations, are rejected before inference rather than ignored. Invalid generated output and exhausted budgets are errors.

### Dependencies, conditions and null

Dependencies form a validated acyclic graph. A dependent prompt receives only its named previous answers, without probability wrappers. The media envelope and its bytes stay intact. Fields currently execute serially in dependency order; this API does not promise shared-prefix batching or parallel field generation. Response fields return in caller order.

```json
{
  "context": "The account has seven overdue invoices.",
  "questions": {
    "count": { "type": "integer", "instructions": "How many invoices?" },
    "escalate": {
      "type": "boolean",
      "when": { "count": { "gte": 5 } },
      "instructions": "Should support escalate this account?"
    },
    "reference": {
      "type": ["string", "null"],
      "instructions": "Return the payment reference, or null when it is absent."
    }
  }
}
```

Tests support equality, list membership, and objects with `in`, `not_in`, `ne`, `gt`, `gte`, `lt`, `lte`. Comparison operands are type-checked. Open string fields may be tested for null, but not arbitrary string equality. A false condition skips the field and all its dependents without inference, including automatic effort selection. Skipped fields are absent from `result` and named in `skipped`; they are not invented null answers. Actual null is present in `result`. The string `"null"` stays a string.

### Probabilities and permutation budgets

```json
{
  "type": "string",
  "enum": ["heads", "tails"],
  "instructions": "Which side of a fair coin will appear?",
  "permutations": "all",
  "return_probabilities": true
}
```

The result wrapper contains `value`, `confidence`, selected raw `probability`, `probabilities` and `distribution`. `distribution` is an array of `{value, probability}` retaining actual JSON types. Map keys normally use string values or JSON spellings for other scalars. In nullable string enums, the literal string `"null"` uses the quoted map key `"\"null\""` to distinguish it from actual null's `"null"` key. Prefer the typed array if keys matter.

Each ordering receives a real model evaluation. Scores are normalized separately, remapped to original values, then **probabilities** are averaged. `auto` uses canonical ordering and a Williams design: N evaluations for an even N, 2N for an odd N greater than one. Integer budgets sample distinct orderings reproducibly using request `seed`; the identity is included. `all` visits every permutation. A hard cap of 720 prevents accidental factorial workloads. A six-sided die with `all` costs 720 evaluations; opt in deliberately. Averaging may reduce position bias but does not guarantee correct reasoning or calibrated confidence. This policy is separate from `evaluate`'s load-time `none`/`cyclic`/`balanced` policy. Positive integer budgets include the caller listing and are not generally order invariant. Use `auto` for canonical balanced layouts. Numeric enums are unordered, unlike ordinal `score`. Exact enum ties compare JSON value encodings.

### Thinking and runtime capabilities

Default `generate` thinking is false, independently of the load-time policy used by `evaluate`. A field can override the request. Budget-only field overrides inherit the request's on/off setting. Effort selects the setting explicitly. Contradictory settings fail validation. `auto` asks the model to choose none/low/medium/high as an additional hidden finite decision, exposes that selection in `thinking_effort`, and then performs the actual field request. It cannot also specify a fixed effort or budget.

| Runtime                        | Finite fields               | Text/numbers                                                    | Reasoning                                                      |
| ------------------------------ | --------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| Native llama.cpp/GGUF          | Candidate logits            | JSON grammar plus Rust type checks                              | Bounded text-model reasoning using the model's actual template |
| Browser llama.cpp/WASM         | Candidate log probabilities | JSON Schema grammar plus Rust type checks                       | Explicit unsupported error                                     |
| Transformers.js text           | Token logits                | Bounded greedy generation, validated afterward; no grammar mask | Explicit unsupported error                                     |
| Existing custom scorer/fixture | Finite scoring              | Requires implementing `infer_field` / `inferField`              | Requires that hook                                             |

Native and browser GGUF field generation accept the existing image/audio/video-frame envelope when the loaded model and matching projector support the modality. That is not a promise of receipt-reading accuracy. Native media reasoning is explicitly unsupported. Video inputs are decoded, timestamped image frames; TurboJev does not decode video containers or invoke external media executables. Native binaries cover Windows/Linux x86/x64/ARM64 and macOS x64/ARM64. Architecture checks and cross-compilation are not execution tests; macOS GPU execution remains unverified.

`usage.decisions` counts completed output fields, not hidden effort queries or permutation evaluations. `input_tokens` sums known actual model counts; `generated_tokens` counts actual generated reasoning/value tokens. `thinking` contains actual model reasoning where supported. Increase context and reduce `nSeqMax` for longer reasoning. Unsupported modes, context overflow and unfinished reasoning return errors; no fallback answer is fabricated.

### Binding calls

| Language        | Method                                                                                        |
| --------------- | --------------------------------------------------------------------------------------------- |
| Rust            | `engine.generate(GenerateRequest)`; types in `turbojev::schema::fields`                       |
| Python          | `client.generate(request)` or `client.generate(context=..., questions=...)`                   |
| Node/TypeScript | `await client.generate(request)`; exported `GenerateRequest` / `GenerateResponse`             |
| Browser         | `await client.generate(request)` or `await wasmEngine.generate_json(JSON.stringify(request))` |
| .NET            | `await engine.GenerateJsonAsync(JsonSerializer.Serialize(request))`                           |
| C               | `tj_engine_generate_json(handle, json, &output)` then `tj_string_free(output)`                |
| Android         | `engine.generate(JSONObject(request))`                                                        |
| Swift           | `try engine.generate(json)` on its managed affinity thread                                    |

The native ABI is shim version 10; rebuild native dependents together. Runtime options are scoped to each field and restored after errors. Existing `evaluate` requests and responses remain supported.

With native `nSeqMax=1`, multi-question `evaluate` uses independent reference
scoring. Shared-prefix reuse requires at least two sequences; parallel suffix
reuse requires at least three. The runtime negotiates these capabilities from
the loaded context. Reducing sequence capacity can save memory while repeating
prefill per question. It does not switch an explicit CPU/GPU device selection.

Wllama 3.8.1 exposes top-N vocabulary scores and its Jinja autoparser does not apply custom GBNF on this path. The adapter resolves letter labels through the loaded model tokenizer using textual logit-bias keys, applies the same additive offset to every permitted label, and requests post-sampling probabilities with truncation disabled. A common offset preserves candidate log-odds and the conditional distribution; it does not favor the first option or select a winner. This uses one model call for 2–24 candidates. Missing or invalid scores are errors, never invented probabilities. The browser adapter does not claim a selective output head.

## Samples, demos and verification

Repository samples may build local artifacts; standalone demos install published packages or CDN assets. Each Python demo must own its code and data and must not import or launch another demo. Prefer meaningful task examples: routing with `evaluate`, invoice fields with `generate`, enum distributions, and image receipt extraction with the matching projector.

Before shipping an integration, check lifecycle closure, valid/invalid requests, omitted versus null fields, actual device selection and model/runtime failures. For suspected position bias, compare keyed distributions under reordered candidates. Record model/quantization/runtime/platform and decision latency separately from downloads and loading. Contract/fixture tests verify wire semantics, not model intelligence.

Run relevant commands from the repository root when developing TurboJev itself:

```powershell
cargo test -p turbojev-core -p turbojev-schema --locked
py evals/run.py --help
node scripts/validate-real-option-order.mjs C:/models/model.gguf
```

The recorded 0.31.0 Windows x64 Qwen3-0.6B Q4_K_M CPU probe passed 18/18 routing decisions across three cases, three orders and none/balanced modes. Balanced keyed drift was zero in that sample; none reached 13.7 percentage points. These are limited routing results, not a general accuracy, Snake or speed guarantee.

## Structured SDK features carried forward from 0.33.0

These features remain included in published 0.34.0. Install matching package versions across
the host, runtime and WASM core. Standalone demos must use registry/CDN packages.

### Nested objects and scoped dependencies

A field can be an object with `properties` containing scalar or nested-object fields.
Rust flattens decisions and reconstructs the original output shape. Within an object, local
names such as `age` resolve in that object's scope; complete paths such as `person.age`
are visible in compiled metadata. Parent instructions, dependencies and conditions are inherited.
An object dependency passes the available plain object; it skips only when all leaves were skipped.
Arrays inside objects and nested arrays are unsupported and rejected.

### Described choices, ordinal levels and confidence gates

```python
request = {
    "context": "The customer says the service is down and they are furious.",
    "questions": {
        "team": {
            "type": "string",
            "choices": [
                {"value": "outage", "description": "Service unavailable"},
                {"value": "billing", "description": "Payments and refunds"},
            ],
            "instructions": "Which team should handle this?",
            "return_probabilities": True,
            "permutations": 1,
        },
        "tone": {
            "type": "number",
            "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}],
            "return_probabilities": True,
        },
        "reply": {
            "type": "string",
            "instructions": "Write a short acknowledgement.",
            "when": {"team": {"confidence": {"gte": 0.8}}},
        },
    },
}
```

`choices` replaces enum with typed value/description records. Descriptions remain attached
through permutations. `levels` needs a non-nullable number field with 2–26 ordered labels.
Its value is the expected zero-based index, not the modal label or a rounded level.
It cannot be combined with enum or permutations. Numeric enums remain unordered.

Finite probability wrappers contain value, confidence, selected probability, probabilities and
an ordered typed distribution. Choice confidence measures the top mass above uniform; ordinal
confidence measures distance from the modal level relative to an even scale. Neither is fitted
calibration. Confidence gates require probability output on the parent and numeric bounds 0–1.

Probability requests now default to `permutations: "auto"` for small unordered fields;
set `permutations: 1` explicitly for latency. Large enums and ordinal levels default to one pass.
The existing evaluate contract and its load-time ordering policy remain separate.

### Arrays, continuation and dense images

```python
from pathlib import Path
from turbojev import TurboJev

rows = []
with TurboJev.load("vision.gguf", mmproj="mmproj.gguf", device="cpu", n_ctx=8192) as client:
    for _ in range(10):
        response = client.generate(
            context="Extract all invoice rows in printed order; never repeat existing rows.",
            images=[Path("invoice.png")],
            questions={
                "rows": {
                    "type": "array",
                    "instructions": "Return the next rows; stop when none remain.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                    },
                    "continue_from": rows,
                },
                "has_next": {
                    "type": "boolean",
                    "depends_on": ["rows"],
                    "instructions": "Are any printed rows still missing from rows?",
                },
            },
        )
        next_rows = response["result"]["rows"]
        advanced = len(next_rows) > len(rows)
        rows = next_rows
        if not response["result"]["has_next"] or not advanced:
            break
```

Arrays are top-level fields with scalar items or object items containing scalar/nested-object
leaves. Object items are one grammar-constrained generation each. Property-level conditions,
permutations, probabilities, levels and thinking are not allowed inside object-array items.
`minItems` defaults to 0. Explicit maxItems is 1–50 and limits the total including the prefix;
without maxItems, a call can append at most 50 new items. Continue_from is validated and preserved.
Non-nullable scalar arrays use null to stop after their minimum; nullable items retain real null
as data and use a separate append decision. Duplicates stop deterministic extraction; sampling
allows two retries. Duplicate output before minItems is an error, never an invented replacement.

The model supplies every extracted row and has_next decision. Host bounds are not model answers.
Images are optional request evidence: 1–32 base64/data-URI strings in caller order, combined limit
24 MiB decoded. Python encodes bytes, paths and PIL-like images. URLs must be fetched explicitly.
Browser imageEvidence(blobs, text) creates the same envelope. A compatible model and projector
remain mandatory; do not claim image accuracy from schema/adapter fixtures.

### Large enums and runtime limits

Schemas accept up to 512 unique typed enum values. Up to 26 use runtime-verified control codes.
For larger text sets, native GGUF and Transformers.js score entire JSON value continuations,
including the closing delimiter, with the full option set in the prompt. Shared first tokens must
not collapse different answers. This can be much slower than one-token scoring.

GGUF/WASM requires a real value-scorer plugin via useChoiceScorer or runtime option
choiceScorerModuleUrl (exports createChoiceScorer). The scorer declares maxChoices (27–512)
and returns one finite log-score per supplied semantic value, in order. It must not select a
value itself or invent scores. A media-aware scorer is required for large multimodal enums.
Unsupported built-in paths return errors. TypeLLM's private/hosted scorer weights are not bundled.

### SDK classes and helpers

Python exports Choice, Decision, ArrayField, Generation, Usage, SchemaError,
compile_json_schema, candidate_softmax, run_schema, example_schema, encode_image and
benchmark_prefix_cache. Compile_json_schema validates in Rust without loading a model and
returns typed semantic metadata, not tokenizer IDs. Generate returns a dictionary;
generate_typed returns attribute access through Generation.result, .thinking, .thinking_effort,
.skipped and .usage. Run_schema loads/generates/closes once; reuse a client for repeated work.

```python
from turbojev import Choice, compile_json_schema, candidate_softmax

schema = {
    "type": "object",
    "properties": {
        "team": Choice("Which team?", {"billing": "Payments", "outage": "Service down"}).to_schema()
    },
}
decisions = compile_json_schema(schema)
probabilities = candidate_softmax({"billing": -3.0, "outage": -0.1})
```

Node exports model-free compileJsonSchema and candidateSoftmax plus typed GenerateRequest,
GenerateResponse and CompiledDecision. Browser WASM exports compile_schema_json and
candidate_softmax_json. C exposes tj_compile_schema_json and tj_candidate_softmax_json;
free their owned UTF-8 output with tj_string_free. C# adds CompileSchema, CandidateSoftmax,
GenerateAsync and typed Generation/usage. Android/Swift use the same structured JSON and
model-free compiler. Existing JSON APIs are still valid.

The Python prefix benchmark compares reference and actual negotiated shared-prefix evaluation
in separate sessions after warm-up. It requires multiple evaluate questions and a runtime that
uses shared-prefix reuse. It reports actual times, input counts and answer equality. Unknown
cached-token counters remain None; it does not pretend to measure SGLang's radix cache.
SGLang transports, server pooling and cache flush operations are vendor-specific optional
integrations, not mandatory native dependencies or renamed bundled services.

### Sampling, cancellation and concurrency

Request temperature defaults to 0; a positive value enables seeded finite sampling. Native
GGUF also supports seeded open-value sampling; ONNX rejects that mode explicitly. GGUF/WASM
uses Wllama completion options. Seed controls reproducibility within the selected runtime;
do not claim identical random sequences across different runtimes.

Python client.generate(request, timeout=30) or a GenerationControl(timeout=30) supports a
cooperative deadline. Another thread may call control.cancel(); loaded engines themselves
remain thread-affine. GenerationCancelled and GenerationTimeout are distinct exceptions.
The native extension releases the GIL during controlled inference while guarding creator-thread
access. Cancellation is checked before/after a complete model operation and cannot interrupt
an active prefill or decode. No partial result or fabricated null is returned.

Node/browser generate(request, {timeoutMs: 30000, signal}) accepts an AbortSignal. C# accepts
TimeSpan and CancellationToken. C callers create tj_generation_control_open, cancel from any
thread, generate with tj_engine_generate_json_controlled and free control after all calls end.
Keep the engine on its creator thread. Node control IDs are private to the loaded addon and
hold shared cancellation state; they never move the model between threads. Browser workers
serialize operations on a loaded runtime; cancellation messages bypass that queue. The engine
can be reused once the cancelled workflow returns. Do not retry side effects automatically.

For finite sampling, temperature changes each layout's probability distribution before
averaging. Sampling draws from that returned distribution. Ordered levels return its
expected index rather than a sampled label. Greedy temperature=0 preserves the normal
scoring distribution and selects its strongest value.

## Interpreting prefix-cache benchmarks

`benchmark_prefix_cache` compares independently loaded reference and shared-prefix
sessions, warms each up, and returns their timings, answer payloads and negotiated
execution plans. `cached_tokens: None` means the runtime did not report cache hits.
Inspect `parity.qualified`, not just the speed ratio. A changed probability can
change downstream confidence gates even if the selected category is unchanged.
The 0.32.0 Windows x64 Qwen3-0.6B CPU check failed numerical parity; it is not a
qualified shared-prefix speed improvement. Select `execution={"policy": "reference"}`
in load options when exact reference probability behavior matters.

## Dice, sampling, and numeric limits

`Dice` is a convenience schema for the six-face fair-die example. It calls normal
`generate` inference; it does not simulate a roll, force a uniform distribution,
or provide cryptographic randomness. Compare the returned distribution with
1/6 per face to measure model bias. `permutations: "auto"` averages six layouts;
8 uses eight distinct layouts; `"all"` evaluates all 720 layouts and is expensive.
Choose 1 explicitly for a single model pass.

```python
from turbojev import Dice, TurboJev

with TurboJev.load("Qwen3-0.6B-Q4_K_M.gguf", device="cpu") as client:
    response = client.generate(
        context="A single roll of a fair die.",
        questions={"roll": Dice(permutations="auto").to_schema()},
        mode="sample",
        temperature=1,
        seed=42,
    )
    print(response["result"]["roll"])
```

```typescript
import { Dice, TurboJev } from "@turbojev/node";

const client = await TurboJev.load({ model: "Qwen3-0.6B-Q4_K_M.gguf" });
try {
  const response = await client.generate({
    context: "A single roll of a fair die.",
    questions: { roll: new Dice("auto").toSchema() },
    mode: "sample",
    temperature: 1,
    seed: 42,
  });
  console.log(response.result.roll);
} finally {
  await client.close();
}
```

Browser JavaScript imports `Dice` from `@turbojev/web` and uses the same request
with `TurboJevWeb`. C# uses `new Dice("auto").ToSchema()` in its questions map.
Rust uses `turbojev::schema::fields::Dice::default().to_schema()`. C and mobile
bindings send the equivalent six-value enum through their existing JSON API.

The default is greedy: no mode/temperature override selects `argmax` at zero.
Positive temperature alone selects `sample`; `mode="sample"` without temperature
uses 1. `mode="argmax"` with positive temperature still chooses the most likely
value, while returning the tempered probabilities. Sample mode with temperature
zero is invalid. Finite sampling draws once from the averaged distribution.
Ordinal `levels` always return the expected zero-based index, not a sampled label;
their probability maps use the actual level names.

Explicit seeds reproduce draws for a fixed model/runtime and request. Native and
browser hosts supply fresh entropy for sample calls without a seed. The portable
Rust core has no entropy source and defaults its omitted seed to zero; Rust hosts
must supply their own seed for a changing stream. Permutations are additional
model work and do not guarantee calibration or improved accuracy.

`numeric_max_digits` defaults to 32 and accepts 1–4096. It counts digits in the
integer and fractional parts of open numeric values; a minus sign and decimal
point do not count. Native GGUF constrains plain-decimal decoding with a grammar;
all adapters validate generated numeric leaves, including object-array items.
Numeric enum constants are exempt. The native grammar excludes scientific notation. Other adapters
validate the digit count of the normalized decimal value. Transformers.js validates after decoding; Wllama's Jinja processing can
override custom grammar, so its returned values are checked by the Rust core.
This is a contract guarantee, not a promise that every model can emit a valid answer.

```python
response = client.generate(
    context="Reference 12345, amount 12.34.",
    numeric_max_digits=5,
    questions={
        "reference": {"type": "integer", "instructions": "Extract the reference."},
        "amount": {"type": "number", "instructions": "Extract the amount."},
    },
)
```

A dependent field receives its direct parents and their ancestors, excluding
unrelated branches. Integer conditions treat `1` and `1.0` as equivalent operands;
integer outputs still require integer JSON values.

Generation errors retain known counts in Python `error.usage`, Node/browser
`error.usage`, .NET `exception.Usage` and Rust `BackendError::usage()`. C reads the
`TURBOJEV_USAGE` JSON trailer in `tj_last_error_message()`. Counts describe completed
operations; a backend that fails before reporting tokens cannot supply those counts.
Array-child token usage is included without counting an unfinished array as a
completed decision. Cancellation remains cooperative between model operations.

Use `print_final_prompt=True` for native GGUF diagnostics. Python prints the actual
runtime-formatted input and every binding returns captured `prompts` when available.
This flag is opt-in and can expose input data in logs. Native GGUF and the ONNX
adapter capture their actual formatted text; GGUF/WASM rejects this flag explicitly
because Wllama does not expose its final formatted prompt. Singleton enums need no
model inference and therefore have no model prompt.

Release 0.33.0 was checked against the public npm, PyPI, NuGet and crates.io
registries. Windows x86/x64/ARM64 assets were verified; real-model execution
was checked on Windows x64. Cross-compilation is not execution on ARM64 or x86.

## Complete application SDK cookbook

This section covers the application-facing APIs, rather than every internal Rust
trait, adapter-author primitive or training utility. All examples and request
definitions are inline. No external documentation or shared demo module is needed.

### Installation and model prerequisites

Install the latest release when starting an application; lock the resolved version
for reproducibility. The examples below target published API version 0.34.0.

```sh
# Python
python -m pip install --upgrade turbojev

# JavaScript or TypeScript on Node; use ES modules (.mjs or type: module).
npm install @turbojev/node@latest
# TypeScript additionally needs its compiler and Node types.
npm install --save-dev typescript @types/node

# .NET; choose the RuntimeIdentifier matching the running process.
dotnet add package TurboJev
# Windows: win-x86, win-x64, win-arm64.
# Linux (glibc 2.28+): linux-x86, linux-x64, linux-arm64.
# macOS 11+: osx-x64, osx-arm64. Host runtime support still applies.

# Rust: core plus optional concrete GGUF runtime.
cargo add turbojev turbojev-runtime-llamacpp --features turbojev-runtime-llamacpp/ffi
cargo add serde_json --features preserve_order

# Browser packages, when using a bundler.
npm install @turbojev/web@latest @turbojev/wasm@latest @turbojev/runtime-llamacpp-wasm@latest
# ONNX is a separate, optional runtime.
npm install @turbojev/runtime-transformersjs@latest @huggingface/transformers
```

Rust's `ffi` feature needs the matching `turbojev_llama_shim` library at final
linking. Set `TURBOJEV_LLAMA_SHIM_LIB_DIR` to its library directory and deploy its
runtime dependencies together. `cargo check` alone does not prove final linking.
Published Windows, Linux and macOS native assets can be obtained from the matching NuGet package;
they are not downloaded automatically by `cargo add`.

C has no separate package-manager installer: include `turbojev.h`, link
`turbojev_capi`, and deploy the matching native dependency set. For a source build,
build the llama.cpp shim first and enable `llamacpp` in `turbojev-capi`. Desktop
consumers can use the matching published NuGet native directory instead. On
Linux deploy all sibling `.so`/SONAME aliases; on macOS deploy all sibling
`.dylib` files. Their `$ORIGIN`/`@loader_path` paths preserve relocation.

Android is a source Gradle module, not a published Maven dependency. Include it
in `settings.gradle.kts` and depend on it in the app:

```kotlin
include(":turbojev")
project(":turbojev").projectDir = file("vendor/TurboJev/bindings/android")
// In app/build.gradle.kts:
dependencies {
    implementation(project(":turbojev"))
}
```

Build/package `libturbojev_jni.so` and all native dependencies under
`src/main/jniLibs/<ABI>/` for each supported Android target, such as
`arm64-v8a` or `x86_64`. Windows DLLs cannot be reused on Android. Copy downloaded
GGUF/projector assets to an application-readable file path before loading.

Swift uses the source Swift package, not a published binary package:

```swift
// Add to the consuming Package.swift dependencies:
.package(path: "vendor/TurboJev/bindings/swift")
// Add to the consuming target dependencies:
.product(name: "TurboJev", package: "TurboJev")
```

Provide a matching `turbojev_capi` library and dependencies for each Apple target.
The current system-library package is not an XCFramework and does not bundle
Apple binaries. Configure library search/linking and app embedding accordingly.

Default text demos use `qjev35-v19c-q8_0.gguf` from
`ZY-Lee4/QJev3.5-0.8B-GGUF`, revision
`008c7c7a5c35007b8ba6b06afba50f882f000667`: 811843104 bytes, SHA256
`7a5a50953df2fb866a2f39bc4ac8906084cac4e3a5fa8f04adc519058ceedf87`.
Native examples assume the model is already downloaded to `model.gguf`.
This checkpoint is decision-trained; use a suitable instruction/reasoning model
for open extraction or thinking. Its upstream trained prompt and benchmark scores
are not automatically reproduced by TurboJev's normal JSON-prefill path.

OCR examples use a vision model plus its matching projector, currently
`unsloth/Qwen3.5-2B-GGUF` revision
`f6d5376be1edb4d416d56da11e5397a961aca8ae`, files
`Qwen3.5-2B-Q4_K_M.gguf` and `mmproj-F16.gguf`. Do not use text-only QJev for OCR.
Audio requires a separate audio-capable model/projector pair. Keep downloads,
model loading and decision timing separate; show download progress in demos.

### Public API coverage and real differences

| Operation                | Python                                  | Node JS / TS                         | Browser JS / TS           | .NET                                 | Rust                             | C                           | Android                 | Swift                         |
| ------------------------ | --------------------------------------- | ------------------------------------ | ------------------------- | ------------------------------------ | -------------------------------- | --------------------------- | ----------------------- | ----------------------------- |
| Load / release           | `load` / `close`, context manager       | `load` / `close`                     | `load` / `close`          | `LoadAsync` / `DisposeAsync`         | Construct runtime / drop         | `open_json` / `close`       | `load` / `close`        | initializer / `close`         |
| Evaluate                 | `evaluate`                              | `evaluate`                           | `evaluate`                | `EvaluateJsonAsync`                  | `evaluate`                       | `evaluate_json`             | `evaluate`              | `evaluate`                    |
| Generate                 | `generate`, `generate_typed`            | `generate`                           | `generate`                | `GenerateJsonAsync`, `GenerateAsync` | `generate`                       | `generate_json`             | `generate`              | `generate`                    |
| Compatibility            | `evaluate_compatible`                   | `evaluateCompatible`                 | `evaluateCompatible`      | `EvaluateCompatibleJsonAsync`        | `compat` encode/decode           | `evaluate_compatible_json`  | `evaluateCompatible`    | `evaluateCompatible`          |
| Compile schema           | `compile_json_schema`, `compile_schema` | `compileJsonSchema`, `compileSchema` | `compileSchema`, raw WASM | `CompileSchema`, `CompileSchemaJson` | `schema::fields` functions       | `tj_compile_schema_json`    | static `compileSchema`  | static `compileSchema`        |
| Candidate softmax        | `candidate_softmax`                     | static export and client method      | client method, raw WASM   | `CandidateSoftmax`                   | `calibration::candidate_softmax` | `tj_candidate_softmax_json` | Not exported by wrapper | C ABI via `CTurboJev`         |
| Controlled generation    | `timeout`, `GenerationControl`          | timeout + AbortSignal                | timeout + AbortSignal     | timeout + CancellationToken          | `generate_controlled` callback   | generation control handle   | Not exported by wrapper | C ABI only, preserve affinity |
| Runtime info             | `info`                                  | `info`                               | `info`                    | No public info wrapper               | `backend()` capabilities         | `tj_engine_info_json`       | No public info wrapper  | `info`                        |
| `Dice` builder           | `Dice.to_schema`                        | `Dice.toSchema`                      | `Dice.toSchema`           | `Dice.ToSchema`                      | `fields::Dice::to_schema`        | JSON enum                   | JSON enum               | JSON enum                     |
| Described-choice builder | `Choice.to_schema`                      | JSON field                           | JSON field                | JSON field                           | JSON field                       | JSON field                  | JSON field              | JSON field                    |

In the C columns, engine operation names are abbreviated: prefix them with
`tj_engine_`. Never invent missing mobile methods to make the matrix symmetrical.
All bindings accept the same supported field schemas through `generate`; helper
classes are conveniences, not different inference capabilities.

### One inline request catalogue, usable in every language

Copy this JSON into a string named `casesJson` (Python: `cases_json`; Rust:
`cases_json`; C: `cases_json`) in the chosen example below. Select one case by
name. This is inline request data, not an imported module or an external file.
Each case is a complete request; no inferred answers or host-selected choices
are supplied. Models can still return an incorrect semantic answer.

```json
{
  "evaluate": {
    "state": "Everything is down and we have a client demo at noon.",
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {
          "outage": "Service unavailable",
          "billing": "Charges and refunds",
          "feature": "Requests and how-to"
        }
      },
      "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"]
      }
    }
  },
  "choices": {
    "context": "The package arrived broken. Please replace it.",
    "questions": {
      "team": {
        "type": "string",
        "choices": [
          { "value": "shipping", "description": "Delivery damage or replacement" },
          { "value": "billing", "description": "Payments or refunds" }
        ],
        "instructions": "Which team should handle this?",
        "permutations": 1,
        "return_probabilities": true
      }
    }
  },
  "scalars": {
    "context": "Invoice 12345 totals 12.34 GBP and is overdue. No payment reference.",
    "text_max_tokens": 64,
    "numeric_max_tokens": 32,
    "numeric_max_digits": 5,
    "questions": {
      "currency": { "type": "string", "instructions": "Extract the currency code." },
      "invoice": { "type": "integer", "instructions": "Extract the invoice number." },
      "amount": { "type": "number", "instructions": "Extract the total amount." },
      "overdue": { "type": "boolean", "instructions": "Is payment overdue?" },
      "reference": {
        "type": ["string", "null"],
        "instructions": "Extract the payment reference, or null if absent."
      }
    }
  },
  "typed_enums": {
    "context": "This is a high-priority request with no assigned owner.",
    "questions": {
      "priority": {
        "type": "integer",
        "enum": [1, 2, 3],
        "instructions": "Select 1 for low, 2 for medium, or 3 for high priority.",
        "permutations": 1,
        "return_probabilities": true
      },
      "owner": {
        "type": ["string", "null"],
        "enum": ["Alex", "null", null],
        "instructions": "Who owns the request? Use actual null if unassigned.",
        "permutations": 1,
        "return_probabilities": true
      }
    }
  },
  "large_enum": {
    "context": "The requested reference is code-27.",
    "questions": {
      "reference": {
        "type": "string",
        "instructions": "Select the exact requested reference.",
        "enum": [
          "code-01",
          "code-02",
          "code-03",
          "code-04",
          "code-05",
          "code-06",
          "code-07",
          "code-08",
          "code-09",
          "code-10",
          "code-11",
          "code-12",
          "code-13",
          "code-14",
          "code-15",
          "code-16",
          "code-17",
          "code-18",
          "code-19",
          "code-20",
          "code-21",
          "code-22",
          "code-23",
          "code-24",
          "code-25",
          "code-26",
          "code-27"
        ],
        "permutations": 1,
        "return_probabilities": true
      }
    }
  },
  "levels": {
    "context": "The customer says: I am furious about this repeated outage.",
    "questions": {
      "tone": {
        "type": "number",
        "instructions": "How upset is the customer?",
        "levels": [
          { "label": "calm", "description": "Neutral and patient" },
          { "label": "annoyed", "description": "Frustrated but composed" },
          { "label": "furious", "description": "Extremely angry" }
        ],
        "return_probabilities": true
      }
    }
  },
  "workflow": {
    "context": "The service is unavailable. Please reply immediately.",
    "questions": {
      "team": {
        "type": "string",
        "enum": ["outage", "billing"],
        "instructions": "Which team should handle this?",
        "return_probabilities": true,
        "permutations": 1
      },
      "acknowledge": {
        "type": "boolean",
        "depends_on": ["team"],
        "when": { "team": { "in": ["outage"], "confidence": { "gte": 0.8 } } },
        "instructions": "Should the outage team acknowledge immediately?"
      },
      "reply": {
        "type": "string",
        "depends_on": ["team"],
        "when": { "acknowledge": true },
        "instructions": "Write one short acknowledgement."
      }
    }
  },
  "nested": {
    "context": "Alex is 21 years old.",
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["person"],
      "properties": {
        "person": {
          "type": "object",
          "properties": {
            "name": { "type": "string", "instructions": "Extract the person's name." },
            "age": { "type": "integer", "instructions": "Extract their age." },
            "adult": {
              "type": "boolean",
              "when": { "age": { "gte": 18 } },
              "instructions": "Is this person an adult?"
            }
          }
        }
      }
    }
  },
  "array": {
    "context": "The agenda covers deployment, security, and billing.",
    "questions": {
      "topics": {
        "type": "array",
        "instructions": "List the remaining agenda topics in their printed order.",
        "items": { "type": "string" },
        "minItems": 1,
        "maxItems": 3,
        "continue_from": ["deployment"]
      }
    }
  },
  "object_array": {
    "context": "Invoice rows: Keyboard, quantity 2. Mouse, quantity 1.",
    "questions": {
      "rows": {
        "type": "array",
        "instructions": "Extract invoice rows in printed order.",
        "maxItems": 2,
        "items": {
          "type": "object",
          "properties": {
            "description": { "type": "string" },
            "quantity": { "type": "integer" }
          }
        }
      }
    }
  },
  "dice": {
    "context": "A single roll of a fair six-sided die.",
    "mode": "sample",
    "temperature": 1,
    "seed": 42,
    "questions": {
      "roll": {
        "type": "string",
        "enum": ["one", "two", "three", "four", "five", "six"],
        "instructions": "What number comes up?",
        "return_probabilities": true,
        "permutations": "auto"
      }
    }
  },
  "thinking": {
    "context": "How many letter r occurrences are in strawberry?",
    "thinking": true,
    "thinking_budget": 512,
    "questions": {
      "count": {
        "type": "integer",
        "enum": [1, 2, 3, 4],
        "instructions": "Count the letter r occurrences.",
        "permutations": 1
      }
    }
  },
  "automatic_thinking": {
    "context": "Are 2026-01-02 and 2026-02-01 in chronological order?",
    "thinking": "auto",
    "questions": {
      "ordered": { "type": "boolean", "instructions": "Is the first date earlier?" }
    }
  },
  "diagnostics": {
    "context": "The service is down.",
    "print_final_prompt": true,
    "questions": {
      "outage": { "type": "boolean", "instructions": "Is service unavailable?" }
    }
  },
  "image": {
    "context": "Read the attached receipt.",
    "questions": {
      "total": { "type": "number", "instructions": "What is the printed total?" }
    }
  }
}
```

`evaluate` is the only catalogue entry sent to `evaluate`; all others go to
`generate`. `large_enum` needs sequence scoring; GGUF/WASM requires the explicit
value-scorer plugin described above. Attach actual image bytes to `image` as shown
below. Thinking requires
native text inference; diagnostics requires a runtime exposing the final prompt.
Do not run the whole catalogue with a text-only decision checkpoint and claim
it has demonstrated extraction, OCR or reasoning.

For condition variants, replace the workflow's gate with `{"team":"outage"}`,
`{"team":["outage","billing"]}`, `{"team":{"not_in":["billing"]}}` or
`{"team":{"ne":"billing"}}`. Numeric parents accept `gt/gte/lt/lte` with
type-compatible bounds. Confidence gates require a probability-wrapped parent.
For permutation variants use 1, a positive integer, `"auto"`, or `"all"`, observing
the 720-evaluation cap. Request and field thinking overrides must be consistent;
use an effort or a budget, not both. Generate schemas are a documented subset of
JSON Schema; do not assume arbitrary validation keywords are implemented.

### Python: every public application helper

After assigning the catalogue string to `cases_json`:

```python
import json
from pathlib import Path
from turbojev import (
    ArrayField, Choice, Decision, Dice, Generation, GenerationControl,
    GenerationCancelled, GenerationTimeout, LoadOptions, SchemaError,
    TurboJev, TurboJevError, Usage, benchmark_prefix_cache,
    candidate_softmax, compile_json_schema, encode_image, example_schema,
    run_schema,
)

cases = json.loads(cases_json)
request = cases["choices"]  # Any supported generate case from the catalogue.
schema = {"type": "object", "properties": request["questions"]}
try:
    compiled = compile_json_schema(schema)  # Model-free Rust validation.
    for field in compiled:
        if isinstance(field, ArrayField):
            print(field.name, field.items, field.continue_from)
        elif isinstance(field, Decision):
            print(field.name, field.path, field.choices, field.depends_on)
except SchemaError as error:
    print("Invalid schema:", error)
    raise

print(candidate_softmax({"shipping": -0.1, "billing": -3.0}, temperature=1))
choice = Choice("Which team?", {"shipping": "Delivery", "billing": "Payments"})
print(choice.to_schema(), Dice(permutations=1).to_schema())
print(example_schema())  # Built-in support schema; no inference here.

options = LoadOptions(
    model="model.gguf", device="cpu", n_ctx=8192, n_seq_max=1,
    execution={"orderRobustness": "none"},
)
with TurboJev.load(options) as client:
    print(client.info())
    print(client.evaluate(cases["evaluate"])["answers"])
    print(client.evaluate_compatible("typesafe-systemone-v1", cases["evaluate"]))
    print(client.compile_schema(schema))
    response = client.generate(request)
    typed = Generation.from_response(response)
    typed_again = client.generate_typed(request)
    print(typed.result, typed.skipped, typed.thinking, typed.thinking_effort)
    print(typed.usage.decisions, typed.usage.input_tokens, typed.usage.generated_tokens)
    print(typed.prompts, typed_again.result, Usage())
    try:
        controlled = client.generate(request, timeout=30)
        control = GenerationControl(timeout=30)
        # Another thread may call control.cancel(); inference stays on this thread.
        controlled = client.generate(request, control=control)
    except (GenerationCancelled, GenerationTimeout, TurboJevError) as error:
        print(error, error.usage)

# One-shot convenience: loads, generates, and closes its own engine.
one_shot = run_schema(
    context=request["context"], questions=request["questions"],
    model="model.gguf", load_options={"device": "cpu"}, timeout=30,
)
print(one_shot.result)

# Optional benchmark: multiple questions and enough sequences for prefix reuse.
# report = benchmark_prefix_cache(
#     "model.gguf", cases["evaluate"], repeats=3,
#     load_options={"device": "cpu", "n_seq_max": 4},
# )
# print(report["parity"], report["speedup"])

# Separate vision session; never run this with the text-only model.
with TurboJev.load("vision.gguf", mmproj="mmproj.gguf", n_ctx=8192) as vision:
    image_request = {**cases["image"], "images": [encode_image(Path("receipt.png"))]}
    print(vision.generate(image_request)["result"])
```

`Generation.result` is a mapping, not an invented nested attribute object.
`Decision`, `ArrayField`, `Generation` and `Usage` are data classes returned by
compilation/generation, not model sessions. `MAX_ENUM_CHOICES` and
`MAX_ARRAY_ITEMS` are exported constants, 512 and 50. Context managers close
automatically; use `client.close()` in `finally` when not using `with`.

### Node JavaScript

After assigning the catalogue string to `casesJson`, run as an ES module:

```javascript
import { readFile } from "node:fs/promises";
import { TurboJev, TurboJevError, Dice, compileJsonSchema, candidateSoftmax } from "@turbojev/node";

const cases = JSON.parse(casesJson);
const request = cases.choices; // Select scalars, levels, workflow, arrays, etc.
const schema = { type: "object", properties: request.questions };
console.log(compileJsonSchema(schema));
console.log(candidateSoftmax({ shipping: -0.1, billing: -3.0 }, 1));
console.log(new Dice(1).toSchema());

const client = await TurboJev.load({
  model: "model.gguf",
  device: "cpu",
  nCtx: 8192,
  nSeqMax: 1,
  execution: { orderRobustness: "none" },
});
try {
  console.log(await client.info());
  console.log((await client.evaluate(cases.evaluate)).answers);
  console.log(await client.evaluateCompatible("typesafe-systemone-v1", cases.evaluate));
  console.log(await client.compileSchema(schema));
  console.log(await client.candidateSoftmax({ shipping: -0.1, billing: -3.0 }));
  const controller = new AbortController();
  // A UI event can call controller.abort().
  const response = await client.generate(request, {
    timeoutMs: 30000,
    signal: controller.signal,
  });
  console.log(response.result, response.skipped, response.thinking_effort, response.usage);
} catch (error) {
  if (error instanceof TurboJevError) console.error(error.code, error.message, error.usage);
  throw error;
} finally {
  await client.close();
}

const vision = await TurboJev.load({ model: "vision.gguf", mmproj: "mmproj.gguf" });
try {
  const image = (await readFile("receipt.png")).toString("base64");
  console.log(await vision.generate({ ...cases.image, images: [image] }));
} finally {
  await vision.close();
}
```

### Node TypeScript

The JavaScript example also works in TypeScript. These additional exported types
check the same requests and responses; they do not create new SDK functions:

```typescript
import {
  TurboJev,
  Dice,
  type GenerateRequest,
  type GenerateResponse,
  type FieldSchema,
  type CompiledDecision,
  type ProbabilityAnswer,
  type LoadOptions,
  type EvaluationRequest,
  type EvaluationResponse,
} from "@turbojev/node";

const cases: Record<string, GenerateRequest> = JSON.parse(casesJson);
const field: FieldSchema = new Dice(1).toSchema();
const request: GenerateRequest = {
  ...cases.dice,
  questions: { roll: field },
};
const options: LoadOptions = { model: "model.gguf", device: "cpu" };
const client = await TurboJev.load(options);
try {
  const decisions: CompiledDecision[] = await client.compileSchema({
    type: "object",
    properties: request.questions,
  });
  const response: GenerateResponse = await client.generate(request);
  const roll = response.result.roll as ProbabilityAnswer;
  console.log(decisions, roll.value, roll.distribution, response.usage);
  const evaluation: EvaluationRequest = JSON.parse(casesJson).evaluate;
  const answer: EvaluationResponse = await client.evaluate(evaluation);
  console.log(answer.answers);
} finally {
  await client.close();
}
```

### Browser JavaScript and TypeScript, including CDN installation

For vanilla HTML, create a same-origin `worker-bootstrap.js` containing only:

```javascript
import "https://cdn.jsdelivr.net/npm/@turbojev/web@0.34.0/worker.js";
```

Import the SDK in a `<script type="module">`. The bootstrap is application-owned
worker setup; all TurboJev code still comes from npm/CDN. Do not point the Worker
constructor directly at a cross-origin CDN URL. Serve via HTTPS or localhost,
with COOP/COEP for threading/audio. Allow required modules, workers and WASM in CSP.
After assigning the catalogue to `casesJson`:

```javascript
import {
  TurboJevWeb,
  TurboJevRemote,
  Dice,
  imageEvidence,
  mediaEvidence,
  videoFrameEvidence,
} from "https://cdn.jsdelivr.net/npm/@turbojev/web@0.34.0/dist/index.js";

const cases = JSON.parse(casesJson);
const cdn = "https://cdn.jsdelivr.net/npm/";
const modelUrl =
  "https://huggingface.co/ZY-Lee4/QJev3.5-0.8B-GGUF/resolve/008c7c7a5c35007b8ba6b06afba50f882f000667/qjev35-v19c-q8_0.gguf";
const client = await TurboJevWeb.load({
  model: "QJev3.5 0.8B v19c",
  workerUrl: new URL("./worker-bootstrap.js", location.href).href,
  runtimeModuleUrl: cdn + "@turbojev/runtime-llamacpp-wasm@0.34.0/src/index.js",
  wasmModuleUrl: cdn + "@turbojev/wasm@0.34.0/turbojev_wasm.js",
  orderRobustness: "none",
  runtimeOptions: { modelUrl, device: "wasm", nThreads: 4 },
});
try {
  console.log(await client.info());
  console.log(await client.evaluate(cases.evaluate));
  console.log(await client.evaluateCompatible("typesafe-systemone-v1", cases.evaluate));
  console.log(await client.compileSchema({ type: "object", properties: cases.choices.questions }));
  console.log(await client.candidateSoftmax({ shipping: -0.1, billing: -3.0 }));
  console.log(new Dice(1).toSchema());
  const response = await client.generate(cases.choices, {
    timeoutMs: 30000,
    signal: new AbortController().signal,
  });
  console.log(response.result, response.skipped, response.usage);
} finally {
  await client.close();
}

// Helpers only encode evidence; choose a matching multimodal session separately.
// imageFile and audioFile are actual Files selected by the application user.
const images = await imageEvidence([imageFile], "Read this receipt.");
const sound = await mediaEvidence(audioFile, "audio", "Identify the spoken command.");
const video = await videoFrameEvidence(
  [{ timestampMs: 0, image: imageFile }],
  "Describe the scene.",
);
console.log(images, sound, video);

// Explicitly selected server transport: this sends input to that server.
// const remote = new TurboJevRemote("https://your-server.example");
// console.log(await remote.evaluate(cases.evaluate, new AbortController().signal));
```

Provide user-selected File inputs before inference. For vision, reload using the
vision GGUF URL and matching `mmprojUrl`,
`modality: "image"`; pass `images.images` as `generate.images` or the complete
envelope as `evaluate.state`. Audio uses `modality: "audio"` and its own paired
checkpoint; decoded video uses `modality: "video"` and an image-capable projector.

For ONNX, change `model` to a supported ONNX repository, `runtimeModuleUrl` to
`https://esm.sh/@turbojev/runtime-transformersjs@0.34.0?deps=@huggingface/transformers@4.3.0`,
and `runtimeOptions` to `{device: "wasm", dtype: "q4"}`. This CDN resolves the
peer dependency for a vanilla module worker; a bundler can use the installed
runtime package and resolve its Transformers dependency instead.
QJev's GGUF is not an ONNX model. Select `webgpu` explicitly to request browser GPU.

Browser TypeScript imports the same symbols from `@turbojev/web`; annotate using
its exported `GenerateRequest`, `GenerateResponse`, `CompiledDecision`,
`FieldSchema`, `ProbabilityAnswer`, `MediaEvidence`, `VideoFrameInput` and
`WebLoadOptions`. Its `TurboJevWebError` carries known `usage` on generation failure.
For example, after copying the same catalogue into `casesJson` and supplying
application `modelUrl` and a same-origin worker bootstrap:

```typescript
import {
  TurboJevWeb,
  Dice,
  type WebLoadOptions,
  type GenerateRequest,
  type GenerateResponse,
  type CompiledDecision,
  type ProbabilityAnswer,
} from "@turbojev/web";

const cases: Record<string, GenerateRequest> = JSON.parse(casesJson);
const options: WebLoadOptions = {
  model: "Selected GGUF",
  workerUrl: "/worker-bootstrap.js",
  runtimeModuleUrl:
    "https://cdn.jsdelivr.net/npm/@turbojev/runtime-llamacpp-wasm@0.34.0/src/index.js",
  wasmModuleUrl: "https://cdn.jsdelivr.net/npm/@turbojev/wasm@0.34.0/turbojev_wasm.js",
  runtimeOptions: { modelUrl, device: "wasm" },
};
const client = await TurboJevWeb.load(options);
try {
  const request: GenerateRequest = { ...cases.dice, questions: { roll: new Dice(1).toSchema() } };
  const compiled: CompiledDecision[] = await client.compileSchema({
    type: "object",
    properties: request.questions,
  });
  const response: GenerateResponse = await client.generate(request, { timeoutMs: 30000 });
  console.log(compiled, (response.result.roll as ProbabilityAnswer).distribution);
} finally {
  await client.close();
}
```

The raw WASM module also exports model-free helpers:

```javascript
import init, {
  compile_schema_json,
  candidate_softmax_json,
} from "https://cdn.jsdelivr.net/npm/@turbojev/wasm@0.34.0/turbojev_wasm.js";
await init();
const cases = JSON.parse(casesJson);
console.log(
  JSON.parse(
    compile_schema_json(
      JSON.stringify({
        type: "object",
        properties: cases.choices.questions,
      }),
    ),
  ),
);
console.log(
  JSON.parse(
    candidate_softmax_json(
      JSON.stringify({
        logits: { shipping: -0.1, billing: -3.0 },
        temperature: 1,
      }),
    ),
  ),
);
```

### C# / .NET

Assign the catalogue to a raw string `casesJson`. Requires C# 11+ for raw strings;
The published 0.34.0 SDK targets .NET Standard 2.0, .NET Framework 4.8 and
.NET 8. Framework 4.8, .NET Core 3.1 and .NET 5–7 use the `netstandard2.0`
assembly. .NET 8 and newer select the `net8.0` assembly. Language syntax and runtime
targets are separate: older C# projects can call the same SDK without raw strings,
top-level statements or `await using`. Read the JSON elements rather
than assuming probability answers or nested objects are plain scalars:

```csharp
using System.Text.Json;
using TurboJev;

using var cases = JsonDocument.Parse(casesJson);
string Request(string name) => cases.RootElement.GetProperty(name).GetRawText();
var questions = cases.RootElement.GetProperty("choices").GetProperty("questions");
var schema = new { type = "object", properties = questions };
Console.WriteLine(TurboJevEngine.CompileSchemaJson(JsonSerializer.Serialize(schema)));
IReadOnlyList<CompiledDecision> compiled = TurboJevEngine.CompileSchema(schema);
var probabilities = TurboJevEngine.CandidateSoftmax(
    new Dictionary<string, double> { ["shipping"] = -0.1, ["billing"] = -3.0 });
Console.WriteLine(JsonSerializer.Serialize(probabilities));
Console.WriteLine(JsonSerializer.Serialize(new Dice(1).ToSchema()));

await using var client = await TurboJevEngine.LoadAsync(new {
    model = "model.gguf", device = "cpu", nCtx = 8192, nSeqMax = 1,
    execution = new { orderRobustness = "none" }
});
try {
    Console.WriteLine(await client.EvaluateJsonAsync(Request("evaluate")));
    Console.WriteLine(await client.EvaluateCompatibleJsonAsync(
        "typesafe-systemone-v1", Request("evaluate")));
    Console.WriteLine(await client.GenerateJsonAsync(Request("choices")));
    using var cancellation = new CancellationTokenSource();
    // A UI action may call cancellation.Cancel().
    Generation response = await client.GenerateAsync(
        cases.RootElement.GetProperty("choices"), TimeSpan.FromSeconds(30), cancellation.Token);
    Console.WriteLine(response.Result["team"].GetProperty("value").GetString());
    Console.WriteLine(response.Usage.Decisions);
    Console.WriteLine(JsonSerializer.Serialize(response.Skipped));
} catch (TurboJevException error) {
    Console.Error.WriteLine(error.Message);
    Console.Error.WriteLine(JsonSerializer.Serialize(error.Usage));
    throw;
}

await using var vision = await TurboJevEngine.LoadAsync(new {
    model = "vision.gguf", mmproj = "mmproj.gguf", device = "cpu", nCtx = 8192
});
var image = Convert.ToBase64String(await File.ReadAllBytesAsync("receipt.png"));
var imageRequest = new {
    context = "Read the receipt.", images = new[] { image },
    questions = cases.RootElement.GetProperty("image").GetProperty("questions")
};
Generation receipt = await vision.GenerateAsync(imageRequest);
Console.WriteLine(receipt.Result["total"].GetRawText());
```

`GenerateJsonAsync` has both a simple overload and a controlled overload with
`TimeSpan?` and `CancellationToken`. `await using` calls `DisposeAsync` and releases
the affinity worker. Do not invoke a nonexistent .NET `InfoAsync` method.

For a .NET Framework 4.8 application, install the published package with
`Install-Package TurboJev` in Visual Studio's Package Manager Console. Use `PackageReference`;
its package targets copy the bundled Windows x86/x64/ARM64 native directories
to the output, and the SDK loads the directory matching the process architecture.
Deploy those directories along with the managed assemblies and dependencies.
An x86 application loads x86 DLLs even on Windows x64. Framework ARM64 support
requires an ARM64-capable Framework runtime (4.8.1); DLL presence alone is not
an execution test. For .NET Core/.NET on Linux or macOS, select a supported
host RID; NuGet also includes Linux x86/x64/ARM64 and macOS x64/ARM64 libraries.
Modern .NET does not offer a Linux x86 host runtime, despite the C ABI asset.

The following method also works with traditional C# syntax. For a UI application,
await it from the event handler so inference does not block the UI:

```csharp
using System;
using System.Threading.Tasks;
using TurboJev;

public static class TicketDemo
{
    public static async Task RunAsync()
    {
        using (var client = await TurboJevEngine.LoadAsync(new
        {
            model = "model.gguf",
            device = "cpu",
            nThreads = 4
        }))
        {
            var response = await client.GenerateAsync(new
            {
                context = "Payments are failing after today's deployment.",
                questions = new
                {
                    team = new
                    {
                        type = "string",
                        instructions = "Which team should handle this?",
                        @enum = new[] { "outage", "billing", "feature" },
                        permutations = 1
                    }
                }
            });
            Console.WriteLine(response.Result["team"].GetString());
        }
    }
}
```

The next SDK implements both `IDisposable` and `IAsyncDisposable`; `using` and
`await using` release the same native affinity worker. For Framework applications,
NuGet supplies `System.Text.Json` and `Microsoft.Bcl.AsyncInterfaces` dependencies.
Enable automatic assembly binding redirects for executable Framework projects.
Older framework availability does not extend Microsoft's support lifecycle for
that runtime. Preserve any runtime errors instead of switching architectures.
Core 2.x and 3.0 are blocked by a transitive dependency's build target and are
not supported. Core 3.1 and .NET 5 are locally qualified legacy configurations;
System.Text.Json 8 emits a support warning for them. Do not suppress that warning
or present them as Microsoft-supported dependency configurations.

### Rust

Use the published core and runtime crates, matching shim and dependencies. Assign
the catalogue to a Rust raw string `cases_json`; no external request file is needed:

```rust
use serde_json::{json, Value};
use turbojev::{
    schema::{fields::{compile_json_schema, Dice, GenerateRequest}, EvaluationRequest},
    OrderRobustness, TurboJev,
};
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig}, reference_backend,
    ReferencePromptConfig, ReferencePromptRenderer,
};

fn demo(cases_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cases: Value = serde_json::from_str(cases_json)?;
    let fields = compile_json_schema(json!({
        "type": "object", "properties": cases["choices"]["questions"]
    }))?;
    println!("{:?}", fields);
    println!("{:?}", turbojev::calibration::candidate_softmax(&[-0.1, -3.0], 1.0)?);
    println!("{}", Dice::default().to_schema());
    let runtime = FfiLlamaCpp::open("model.gguf", NativeConfig {
        n_ctx: 8192, n_seq_max: 1, n_gpu_layers: 0,
        ..Default::default()
    })?;
    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "application-en-v1", "Choose one allowed answer code.", "Which answer fits?",
    ));
    let engine = TurboJev::new(reference_backend(renderer, runtime))
        .with_order_robustness(OrderRobustness::None);
    let evaluation: EvaluationRequest = serde_json::from_value(cases["evaluate"].clone())?;
    let answer = engine.evaluate(evaluation.clone())?;
    println!("{}", serde_json::to_string_pretty(&answer)?);
    println!("{}", turbojev::compat::encode_response(
        turbojev::compat::WireProfile::TypeSafeSystemOne, &evaluation, &answer,
    )?);
    let request: GenerateRequest = serde_json::from_value(cases["choices"].clone())?;
    let response = engine.generate(request.clone())?;
    println!("{:?} {:?} {:?}", response.result, response.skipped, response.usage);
    let started = std::time::Instant::now();
    let controlled = engine.generate_controlled(request, || {
        if started.elapsed() >= std::time::Duration::from_secs(30) {
            return Err(turbojev::backend::BackendError::Inference("deadline expired".into()));
        }
        Ok(())
    })?;
    println!("{:?}", controlled.usage);
    Ok(()) // Runtime is released by drop.
}
```

Rust compatibility is `decode_request` / `encode_request` / `encode_response`;
there is no facade `evaluateCompatible` method. For multimodal sessions use
`FfiLlamaCpp::open_multimodal("vision.gguf", "mmproj.gguf", config)` and send the
same catalogue image request with base64 `images` or a media envelope. For native
evaluate thinking, use `.with_thinking(true, 512)` on the runtime before building
the backend. For explicit GPU, select the required GPU layers/backend; `NativeConfig`
does not have the JSON host's `device` field. The core cannot supply fresh entropy:
set `GenerateRequest.seed` from the host when repeated sample calls need new draws.

Rust also exposes `with_planning_policy`, `with_execution_budget`,
`with_calibrator`, `backend`, `can_evaluate_fused` and `evaluate_fused`:

```rust
// Given a compatible backend and a validated vector of EvaluationRequest values:
let engine = TurboJev::new(backend)
    .with_planning_policy(turbojev::engine::PlanningPolicy::Reference)
    .with_execution_budget(turbojev::engine::ExecutionBudget::default());
if engine.can_evaluate_fused(&requests)? {
    let responses = engine.evaluate_fused(&requests)?;
    println!("{:?}", responses);
}
// A separately fitted calibrator can be installed at construction:
// let engine = turbojev::TurboJevEngine::with_calibrator(backend, calibrator);
```

Fusion depends on negotiated runtime support and measured numerical parity;
do not enable the unqualified llama.cpp multi-root optimization just for speed.
`TurboJevAsync` has asynchronous evaluate/generate methods for an actual
`AsyncScoringBackend`; it does not make a thread-affine native model movable.

### C: every stable ABI operation

C does not include a JSON parser. In this example, assign the exact selected
catalogue requests to `evaluation_json` and `generation_json`, and its question
map inside an object schema to `schema_json`. These are inline UTF-8 literals.
Use a JSON library or escaped string literals; never concatenate unescaped user input.

```c
#include <stdio.h>
#include "turbojev.h"

static int report(int32_t status, char *output) {
    if (status != TJ_OK) {
        fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message());
        tj_string_free(output);
        return 0;
    }
    if (output) puts(output);
    tj_string_free(output);
    return 1;
}

int demo(const char *evaluation_json, const char *generation_json, const char *schema_json) {
    tj_engine *engine = NULL;
    tj_generation_control *control = NULL;
    char *output = NULL;
    int32_t status;
    printf("C ABI: %u\n", tj_abi_version());
    status = tj_compile_schema_json(schema_json, &output);
    if (!report(status, output)) return 1;
    output = NULL;
    status = tj_candidate_softmax_json(
        "{\"logits\":{\"shipping\":-0.1,\"billing\":-3.0},\"temperature\":1}", &output);
    if (!report(status, output)) return 1;
    status = tj_engine_open_json(
        "{\"model\":\"model.gguf\",\"device\":\"cpu\",\"nCtx\":8192,\"nSeqMax\":1}", &engine);
    if (!report(status, NULL)) return 1;

    output = NULL;
    status = tj_engine_info_json(engine, &output);
    if (!report(status, output)) goto failed;
    output = NULL;
    status = tj_engine_evaluate_json(engine, evaluation_json, &output);
    if (!report(status, output)) goto failed;
    output = NULL;
    status = tj_engine_evaluate_compatible_json(
        engine, "typesafe-systemone-v1", evaluation_json, &output);
    if (!report(status, output)) goto failed;
    output = NULL;
    status = tj_engine_generate_json(engine, generation_json, &output);
    if (!report(status, output)) goto failed;
    status = tj_generation_control_open(30000, &control);
    if (!report(status, NULL)) goto failed;
    /* Another thread may call tj_generation_control_cancel(control). */
    output = NULL;
    status = tj_engine_generate_json_controlled(engine, generation_json, control, &output);
    tj_generation_control_free(control);
    control = NULL;
    if (!report(status, output)) goto failed;
    return report(tj_engine_close(engine), NULL) ? 0 : 1;

failed:
    tj_generation_control_free(control);
    report(tj_engine_close(engine), NULL);
    return 1;
}
```

Free each successful owned output exactly once. Error strings are borrowed and
thread-local; inspect them before another native call can replace them. Keep all
engine operations on its creator thread. Do not free a generation control while
another thread is still cancelling it. For vision, add `mmproj` to load JSON;
attach inline base64 `images` to the image generate request. Audio/video use the
same envelope below; no C-only `Dice` or `Choice` function exists.

### Android / Kotlin

Copy the catalogue into a Kotlin triple-quoted string `casesJson`:

```kotlin
import ai.turbojev.TurboJev
import org.json.JSONObject

val cases = JSONObject(casesJson)
val schema = JSONObject()
    .put("type", "object")
    .put("properties", cases.getJSONObject("choices").getJSONObject("questions"))
val compiled = TurboJev.compileSchema(schema)
println(compiled)
val loading = TurboJev.load(JSONObject()
    .put("model", modelPath)
    .put("device", "cpu")
    .put("nCtx", 8192)
    .put("nSeqMax", 1))

// Run blocking joins on a background executor, never Android's UI thread.
val engine = loading.join()
try {
    println(engine.evaluate(cases.getJSONObject("evaluate")).join())
    println(engine.evaluateCompatible(
        "typesafe-systemone-v1", cases.getJSONObject("evaluate")).join())
    val response = engine.generate(cases.getJSONObject("choices")).join()
    println(response.getJSONObject("result"))
    println(response.getJSONArray("skipped"))
    println(response.getJSONObject("usage"))
    // Any other supported catalogue case uses exactly engine.generate(case).
} finally {
    engine.close()
}

// In a separate session loaded with the matching vision model and mmproj:
val imageRequest = cases.getJSONObject("image")
val encoded = android.util.Base64.encodeToString(imageBytes, android.util.Base64.NO_WRAP)
imageRequest.put("images", org.json.JSONArray().put(encoded))
// println(visionEngine.generate(imageRequest).join())
```

`modelPath` and `imageBytes` are application inputs. Future completion failures
wrap `TurboJevException`; unwrap the cause when displaying native diagnostics.
The wrapper exports load, compileSchema, evaluate, generate, evaluateCompatible
and close. It currently does not export info, softmax or controlled-generation
convenience methods. Use the JSON dice catalogue entry rather than inventing a
Kotlin `Dice` class. Actual Android execution is separate from Windows evidence.

### Swift

Assign the catalogue to a Swift multiline string `casesJson`. Keep JSON values
typed through Foundation; the wrapper returns JSON strings:

```swift
import Foundation
import TurboJev

func demo(casesJson: String, modelPath: String) throws {
    let cases = try JSONSerialization.jsonObject(with: Data(casesJson.utf8)) as! [String: Any]
    func json(_ value: Any) throws -> String {
        String(data: try JSONSerialization.data(withJSONObject: value), encoding: .utf8)!
    }
    let choice = cases["choices"] as! [String: Any]
    let schema: [String: Any] = ["type": "object", "properties": choice["questions"]!]
    print(try TurboJevEngine.compileSchema(json(schema)))
    let engine = try TurboJevEngine(optionsJSON: json([
        "model": modelPath, "device": "cpu", "nCtx": 8192, "nSeqMax": 1
    ]))
    defer { try? engine.close() }
    print(try engine.info())
    print(try engine.evaluate(json(cases["evaluate"]!)))
    print(try engine.evaluateCompatible(
        profile: "typesafe-systemone-v1", requestJSON: json(cases["evaluate"]!)
    ))
    let responseJSON = try engine.generate(json(choice))
    let response = try JSONSerialization.jsonObject(with: Data(responseJSON.utf8)) as! [String: Any]
    print(response["result"]!, response["skipped"]!, response["usage"]!)
    // Scalars, enums, levels, arrays, workflows and Dice all use engine.generate.
}
```

Run synchronous work off the application's UI thread; the wrapper dispatches
native operations to its affinity thread. Catch `TurboJevError.native(code:message:)`
and `TurboJevError.closed`. It does not expose its native handle for arbitrary
external controlled-generation calls. A direct C ABI integration is possible,
but must own engine creation, affinity and closure independently.
For an image request, load the vision session with `mmproj` and set
`images: [try Data(contentsOf: imageURL).base64EncodedString()]` before generation.
The dice catalogue case is its JSON equivalent; no Swift `Dice` class is exported.

### Image, audio and video recipes for every binding

The same UTF-8 JSON envelope works in each native binding and browser runtime
that declares the modality. Substitute actual base64 bytes encoded in the host;
the Rust core never reads paths or downloads media. A selected text model cannot
process these inputs just because the schema is valid.

```json
{
  "contract": "turbojev.media.v1",
  "modality": "audio",
  "text": "Identify the command the speaker says.",
  "base64": "REPLACE_WITH_ACTUAL_SUPPORTED_AUDIO_BASE64"
}
```

Use this as `evaluate.state` with a `choice` question describing spoken commands,
or as `generate.context` with the desired bounded output fields. For a still
image use `modality: "image"` and actual encoded image bytes. For multiple images,
use `images: ["BASE64_IMAGE_1", "BASE64_IMAGE_2"]` instead of `base64`.

```json
{
  "contract": "turbojev.media.v1",
  "modality": "video",
  "text": "Does the object move from left to right?",
  "frames": [
    { "timestamp_ms": 0, "base64": "REPLACE_WITH_FIRST_IMAGE_BASE64" },
    { "timestamp_ms": 1000, "base64": "REPLACE_WITH_SECOND_IMAGE_BASE64" }
  ]
}
```

Replace both illustrative frame strings before calling inference. Use 1–32
frames with increasing nonnegative integer timestamps and at most 24 MiB of
combined decoded input. TurboJev receives decoded frames; it does not decode MP4,
invoke ffmpeg/ffprobe, or guarantee temporal reasoning. Browser helpers produce
these exact envelopes from Files/Blobs. Python uses `base64.b64encode(bytes)`,
Node `Buffer.toString("base64")`, C# `Convert.ToBase64String`, Rust a host base64
encoder, C its chosen encoder, Kotlin `Base64.NO_WRAP`, and Swift
`Data.base64EncodedString`. Encode real bytes, never a filesystem path string.

### Documentation verification boundary

The cookbook's method names and signatures were checked against 0.33.0 binding
source. Inline schema compilation checks establish request validity, not model
accuracy or successful inference on every host. Never describe an unexecuted
Android, Swift, x86 or ARM64 snippet as a passed runtime test. Open values,
thinking and modalities require their own actual compatible-model checks.

The 14 generate catalogue schemas compiled through published Python 0.33.0's
Rust validator. Python snippets passed syntax/export checks; Node and browser
TypeScript examples passed type checking. The C# example compiled against NuGet
0.33.0, the Rust example passed `cargo check` against registry 0.33.0, and the C
example passed strict C11 syntax checks. Rust final native linking and Android/
Swift execution were not checked in this documentation update.

Browser validation artifacts are independent of any landing page. From the SDK
repository root, run `scripts/prepare-wasm-tests.ps1` to build the test core under
ignored `.artifacts/wasm-tests/`, then
`python scripts/serve-browser-tests.py --port 4176` to serve the repository with
COOP/COEP headers. The audio page is
`/tests/e2e/browser-gguf-audio.html`; it still requires its real model/projector.
The Node audio harness accepts `--core PATH_TO_WASM` with matching generated
JavaScript alongside it. Preparing artifacts is not a passed model test.

### Complete configuration reference

Native Node/.NET and C/mobile load JSON use camelCase; Python `LoadOptions` uses
the mapped snake_case names. Browser loads use `WebLoadOptions` and place runtime
settings inside `runtimeOptions`, not arbitrary native JSON keys.

| Native JSON / Node / .NET | Python                        | Purpose                                                                    |
| ------------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| `model`                   | `model`                       | Local GGUF path; fixture URLs are only for binding tests                   |
| `mmproj`                  | `mmproj`                      | Matching multimodal projector path                                         |
| `device`                  | `device`                      | Explicit `cpu` or `gpu`; no silent device fallback                         |
| `nCtx`                    | `n_ctx`                       | Context size; allow room for prompts, media and generated values           |
| `nBatch`                  | `n_batch`                     | Model prefill batch size                                                   |
| `nSeqMax`                 | `n_seq_max`                   | Sequence capacity; 1 saves memory and repeats per-question prefill         |
| `nGpuLayers`              | `n_gpu_layers`                | GPU offload; do not combine nonzero layers with explicit CPU               |
| `nThreads`                | `n_threads`                   | Inference thread count                                                     |
| `nThreadsBatch`           | `n_threads_batch`             | Prefill thread count                                                       |
| `thinking`                | `thinking`                    | Load-time native evaluate reasoning, default false                         |
| `maxThinkingTokens`       | `max_thinking_tokens`         | Evaluate reasoning budget, default 512                                     |
| `execution`               | `execution`                   | Planner, memory/concurrency limits and order policy                        |
| `worker`                  | `worker`                      | Explicit custom process-runtime integration; not required for bundled GGUF |
| `prompt`                  | Not exported in `LoadOptions` | Profile, systemInstruction and bestMatchInstruction overrides              |
| `cacheDir`, `nativePath`  | Not exported in `LoadOptions` | Node model cache and explicit addon location                               |

`execution` supports `policy` (`auto`, `reference`, `deterministic-reuse`),
`orderRobustness` (`none`, `cyclic`, `balanced`), `maxParallelTasks`,
`memoryBudgetBytes` and opt-in `llamacppMultiRootFused`. Planner capability
declarations and numerical qualification still control eligibility. The normal
packages use their bundled native runtime; no external llama-server executable is
required. Custom process runtimes are an explicitly chosen integration.

Browser `WebLoadOptions` supports `model`, `runtimeModuleUrl`, `wasmModuleUrl`,
`workerUrl`, `runtimeOptions`, `orderRobustness`, legacy
`cyclicOrderRobustness` and `prompt`. GGUF runtime options include `modelUrl`,
`modelLabel`, `mmprojUrl`, `modality`, `device` (`wasm` or `webgpu`), `nThreads`,
`jinja`, `chatTemplate`, `warmup`, `debug`, `choiceScorerModuleUrl` and
`choiceScorerOptions`. A progress callback belongs in worker runtime code;
functions cannot be sent as structured-cloned `runtimeOptions` from the page.
ONNX options include `model`, `device`, `dtype` and a runtime-side progress callback.

For all bindings, `generate` supports exactly one of `questions` / `schema`,
`context` (alias `state`), `images`, `thinking`, `thinking_effort`,
`thinking_budget`, `text_max_tokens` (default 256), `numeric_max_tokens` (64),
`numeric_max_digits` (32), `print_final_prompt`, `mode`, `temperature` and `seed`.
Scalar fields support `type`, `instructions` / `description`, `enum` / `choices`,
`levels`, `depends_on`, `when`, `permutations`, `return_probabilities`, and their
own thinking overrides. Object fields use `properties` plus scoped dependencies
and conditions. Array fields use `items`, `minItems`, `maxItems`, `continue_from`,
dependencies and conditions; do not add unsupported per-item workflow features.

Native response data is authoritative: evaluate returns `answers`, `usage` and
`_turbojev` provenance; generate returns `result`, `skipped`, `thinking`,
`thinking_effort`, `usage` and optional `prompts`. A finite probability wrapper
retains its typed `value`, `confidence`, `probability`, `probabilities` and
`distribution`. Skipped fields are omitted, not assigned null. Select a suitable
model and inspect errors instead of fabricating the expected example outputs.
