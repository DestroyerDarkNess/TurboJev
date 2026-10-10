---
name: turbojev
description: Use when installing, integrating, designing decisions for, or troubleshooting TurboJev in Rust, Node.js/TypeScript, Python, C, .NET, browser/WASM, Android, or Swift applications.
---

# TurboJev

Help build applications with TurboJev's typed decision and structured workflow APIs. This guide describes the 0.33.0 published SDK: `evaluate`, `generate`, balanced choice ordering, field dependencies, conditional omission, probability wrappers and bounded reasoning. Use the repository's current source and docs as the authority: package versions, platform assets, and runtime capabilities change. Do not repeat old README version numbers or claim validation that has not been run.

This skill is self-contained: installation, contracts, options and limitations are included below. No external Markdown is required to follow it.

## First establish the integration

Identify the host language, target OS/architecture, model format, and intended inference runtime. TurboJev has a portable decision core and separate runtime adapters. The core does not load every model by itself. A checkpoint is usable only when the selected runtime supports its architecture, tokenizer, prompt template, and required modality.

## Install by language

Published native 0.33.0 packages cover Windows x86/x64/ARM64. Browser WASM is architecture independent; Rust source can target other platforms with a compatible toolchain. This is not evidence of native Linux/macOS packages or testing on every architecture. Check release assets before promising other availability. These packages expose TurboJev's typed-decision API; they do not all include the same model runtime.

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

Native and browser GGUF field generation accept the existing image/audio/video-frame envelope when the loaded model and matching projector support the modality. That is not a promise of receipt-reading accuracy. Native media reasoning is explicitly unsupported. Video inputs are decoded, timestamped image frames; TurboJev does not decode video containers or invoke external media executables. Current published native binaries cover Windows x86/x64/ARM64; portable source and WASM support do not imply Linux/macOS binary releases or execution on every architecture.

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

## Structured SDK extension (source 0.32.0)

Do not invoke new APIs from an older registry package. Build this source revision until its
release is verified. Standalone demos must keep using published packages, not local libraries.

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
