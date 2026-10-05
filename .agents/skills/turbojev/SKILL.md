---
name: turbojev
description: Use when installing, integrating, designing decisions for, or troubleshooting TurboJev in Rust, Node.js/TypeScript, Python, C, .NET, browser/WASM, Android, or Swift applications.
---

# TurboJev

Help build applications with TurboJev's typed decision API. Use the repository's current source and docs as the authority: package versions, platform assets, and runtime capabilities change. Do not repeat old README version numbers or claim validation that has not been run.

## First establish the integration

Identify the host language, target OS/architecture, model format, and intended inference runtime. TurboJev has a portable decision core and separate runtime adapters. The core does not load every model by itself. A checkpoint is usable only when the selected runtime supports its architecture, tokenizer, prompt template, and required modality.

## Install by language

Check the current package registry, release assets, target OS, and CPU architecture before promising availability. These packages expose TurboJev's typed-decision API; they do not all include the same model runtime.

| Host | Install | What it provides |
|---|---|---|
| Rust | `cargo add turbojev` | Portable typed-decision API. For GGUF, add `cargo add turbojev-runtime-llamacpp --features ffi`. |
| Node.js / TypeScript | `npm install @turbojev/node` | Worker-backed native binding. |
| Python | `python -m pip install turbojev` | Context-managed native binding. |
| C# / .NET | `dotnet add package TurboJev` | .NET binding over the stable C ABI. |
| C | Build native artifacts, include `bindings/c/include/turbojev.h`, and link the matching C ABI library. | Stable UTF-8 JSON ABI. |
| Browser JavaScript / TypeScript | `npm install @turbojev/web` | Browser host, worker, and WASM integration. |
| Android / Kotlin | Include the Android binding source/module and package matching native libraries. | JNI wrapper over the C ABI. |
| Swift / iOS / macOS | Add the local Swift package at `bindings/swift` and package the matching C ABI library. | Swift wrapper over the C ABI. |

For native samples, build artifacts for the same architecture as the host compiler/interpreter. The samples builder takes one architecture per run (for example, `--arch x64`); it does not mean “all architectures.” C consumers can start with `cargo build -p turbojev-capi --release`; build/link the llama.cpp shim only when GGUF inference is needed.

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
- Test for order sensitivity by permuting candidates. A model that repeatedly chooses the first candidate is order-biased, even when one fixed ordering appears accurate. `orderRobustness: "cyclic"` evaluates multiple rotations, costs additional model evaluations, and is not a guarantee of correctness. Keep single-pass mode by default unless reproducible tests justify the added latency.
- Validate with the exact model, quantization, runtime, and prompt path intended for deployment. Use reproducible cases and a simple reference policy where possible; report both decision quality and latency.

For model-controlled Snake, provide board facts (head, body, food, current direction, bounds, and coordinate convention) and ask for one actual direction: up, down, left, or right. Do not precompute food distances or closer/farther categories, choose a direction in the host, or skip evaluation. Apply the model choice unchanged; game physics reports collisions as failures. Log the complete request, response, selected action, applied action, and outcome. Earlier progress-category Snake results measured an assisted policy and do not establish model-controlled navigation. Test multiple starting states and candidate orders; a successful game is not evidence of general reliability.

For a deliberately simpler application, an explicit sensor representation may
derive relative food direction and immediate collision facts from game physics.
Disclose this as perception assistance and distinguish it from raw-coordinate
reasoning. Keep all four directions available and apply the returned direction
unchanged. Do not search for a path or map a model-selected progress category
to a host-selected move. With PyPI 0.29.2 and Qwen3-0.6B Q4_K_M, the sensor demo
completed three 40-turn games (seeds 0/1/2), eating 3/4/3 food with no collision
or replaced action. Six states under four rotations passed 18/24; the raw wall
state remained 0/4. These small results do not guarantee order independence,
optimal navigation or reliability on longer games.

A matched Windows x64 comparison using PyPI 0.29.2 and Qwen3-0.6B Q4_K_M
checked TurboJev against direct llama.cpp candidate logits and greedy JSON
generation both with and without the assistant prefill. All four paths selected
the same option in 36/36 cases, with exactly equal conditional probabilities
for direct logits and TurboJev (maximum absolute difference 0.0). Four raw wall
states under four rotations passed only 1/16; the corresponding sensor inputs
passed 13/16; a color control passed 4/4. All generations completed valid JSON
within 32 tokens, with thinking disabled. These results exclude an observed
scoring/mapping discrepancy in that sample; they do not prove the model cannot
solve the task with other prompts or inference modes, or qualify other runtimes.
Do not describe the remaining accuracy/order issue as solved by sensor games.

## Choose the right runtime

- **Native text inference:** use the llama.cpp adapter for GGUF models it can load. CPU is the normal default. GPU is opt-in; an unsupported GPU request should surface the runtime error, not silently change devices.
- **Node/TypeScript, Python, C, and .NET:** these bindings expose the shared native host. Their package or binary must match the target OS, CPU architecture, and language ABI. Check actual release assets before promising x86, x64, or ARM64 support.
- **Rust:** compose the core crate with the runtime adapter the application selects; adding the core crate alone does not provide GGUF inference.
- **Browser/WASM:** choose a supported browser runtime, such as Transformers.js with ONNX or the llama.cpp WASM path with compatible GGUF. Browser hardware, memory, model-format, tokenizer, and template constraints apply.
- **Android and Swift:** these bindings call native code through JNI/C ABI and need the corresponding host libraries packaged for the app. Do not present them as registry packages unless the current repository/release explicitly provides one.
- **Images, audio, and video:** the media envelope is a transport contract, not an inference model. Select a multimodal checkpoint and runtime that implement preprocessing and candidate scoring for that modality.

Browser runtime choices are separate integrations: Transformers.js for supported ONNX models, llama.cpp WebAssembly for GGUF models supported by that WASM build, or a custom implementation of the browser runtime interface. WebGPU must be requested explicitly; expose CPU/WASM where available and report unsupported-device errors honestly. Browser memory, hardware, model format, tokenizer, and template support can differ from the native host.

TurboJev's media envelope only standardizes how evidence enters `evaluate`; it does not include a vision/audio model, processor, or inference runtime. Verify that both the selected checkpoint and adapter support the modality and task. A GGUF filename alone does not guarantee compatibility.

TurboJev is useful for classification, routing, typed gates, and bounded rubric scoring when an application needs a structured choice from an open-weight model. It is not a general text generator, chat replacement, route planner, or promise that every model/runtime combination works.

## Troubleshooting

1. Reproduce with the same request, model file, quantization, runtime, and device.
2. Separate model-load errors from evaluation errors; preserve the complete native/runtime error and check model architecture, tokenizer/template compatibility, file path, and platform assets.
3. Inspect candidate descriptions, candidate order, and the returned distribution. A high score on one candidate is not a guarantee of a correct semantic answer.
4. Compare one-pass and cyclic order scoring only when order bias is suspected, and include the extra latency in the result.
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
and sequence capacity. These APIs are introduced in release 0.29.2;
do not assume the installed registry version already supports them.

The native adapter uses the actual model Jinja reasoning template and its
closing delimiters, then verifies candidate tokens after the literal JSON
answer prefill. It does not invent a thought or choose a direction in the host.
`usage.generated_tokens` counts actual model generation, summed across questions
and cyclic evaluations; direct scoring stays zero. Shared/parallel/fused scoring
is disabled for thinking, and the native host limits the context to at most four
sequences. Increase `nCtx` or reduce `nSeqMax` to make room for longer prompts and
thoughts. `THINKING_UNSUPPORTED`, `THINKING_CONTEXT_LIMIT` and
`THINKING_BUDGET_EXHAUSTED` are real failures; do not hide them or fabricate an
answer. Current media, worker and browser adapters reject this mode explicitly.
Native shim ABI 10 is required in 0.30.0; rebuild native dependents together.

Real Windows CPU results: Qwen3-0.6B Q4/Q8 passed 20/20 simpler rotated decisions
but failed 0/4 on the raw-coordinate Snake wall state. Qwen3.5-0.8B,
Qwen3.5-2B and LFM2-350M Q4_K_M each passed those 20 simpler decisions and only
1/4 wall rotations. A literal ASCII grid gave Qwen3.5-2B 2/4; its seed-0 game
still ate no food and hit a wall after six decisions. A relative-position and
neighbor-observation experiment also failed and was removed from the demo.
No representation above is a qualified Snake solution. In a real completed
thinking test, Qwen0.6B chose red after 123 generated tokens. Qwen3.5-2B did
not finish Snake reasoning within 512 tokens. Thinking has measurable cost and
is not an automatic correction for insufficient planning or instruction following.

Valen-Preview-0923 combines Qwen3.5-2B with a trained candidate decision head,
language-model LoRA and adapted visual components. After general training it
was trained on 30,000 Sokoban records for three epochs. Its data generator uses
optimal-action labels offline; its model evaluation receives pixels/rules and
executes the predicted action without a host solver or legal-action mask.
The model card reports 87.60% single-step Sokoban accuracy and 38/100 complete
games on an outcome-conditioned set, not universally correct navigation. Those
figures are not a controlled Snake comparison; we reviewed its code and card
but did not run its checkpoint locally. Copying a random, untrained scoring head
would not reproduce its learned policy. Its checkpoint is not a drop-in GGUF.


## JSON-typed fields in 0.30.0

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

The same JSON request is accepted by every binding. Supply exactly one of `questions` (a field map) or `schema` (an object with `properties`, optional `required`, and optional `additionalProperties: false`). All declared properties are requested; `required` validates property names, not a separate omission policy. Conditions control omissions. The schema subset supports scalar fields, not nested objects or arrays. `state` is an input alias for `context`.

| Field option | Behavior |
|---|---|
| `type` | `string`, `integer`, `number`, `boolean`, or `[type, "null"]`; retains JSON types |
| `enum` | 1–24 unique values of that type; a singleton is a constant with no model scoring |
| `instructions` / `description` | Question; description is the fallback |
| `depends_on` | Names of earlier results to include as plain values; forward references allowed |
| `when` | AND of tests on prior fields; adds implicit dependencies |
| `return_probabilities` | Boolean, finite fields only; returns a value/confidence/distribution wrapper |
| `permutations` | Explicit enums only: positive integer, `auto`, or `all` |
| `thinking` | Per-field override: false, true, or `auto` |
| `thinking_effort` | `none`, `low` (512 tokens), `medium` (2048), `high` (4096) |
| `thinking_budget` | Positive token budget, up to 32768; alternative to effort |

Token limits are request-level `text_max_tokens` (default 256) and `numeric_max_tokens` (default 64), each 1–32768. Booleans default to `[true, false]`, plus actual null for nullable booleans. Numeric equality distinguishes booleans from integers and does not round distinct 64-bit integer values. Unsupported keywords, including bounds, string length/format constraints and nested schemas, are rejected before inference rather than ignored. Invalid generated output and exhausted budgets are errors.

### Dependencies, conditions and null

Dependencies form a validated acyclic graph. A dependent prompt receives only its named previous answers, without probability wrappers. The media envelope and its bytes stay intact. Fields currently execute serially in dependency order; this API does not promise shared-prefix batching or parallel field generation. Response fields return in caller order.

```json
{
  "context": "The account has seven overdue invoices.",
  "questions": {
    "count": {"type": "integer", "instructions": "How many invoices?"},
    "escalate": {"type": "boolean", "when": {"count": {"gte": 5}}, "instructions": "Should support escalate this account?"},
    "reference": {"type": ["string", "null"], "instructions": "Return the payment reference, or null when it is absent."}
  }
}
```

Tests support equality, list membership, and objects with `in`, `not_in`, `ne`, `gt`, `gte`, `lt`, `lte`. Comparison operands are type-checked. Open string fields may be tested for null, but not arbitrary string equality. A false condition skips the field and all its dependents without inference, including automatic effort selection. Skipped fields are absent from `result` and named in `skipped`; they are not invented null answers. Actual null is present in `result`. The string `"null"` stays a string.

### Probabilities and permutation budgets

```json
{"type":"string","enum":["heads","tails"],"instructions":"Which side of a fair coin will appear?","permutations":"all","return_probabilities":true}
```

The result wrapper contains `value`, `confidence`, `probabilities` and `distribution`. `distribution` is an array of `{value, probability}` retaining actual JSON types. Map keys normally use string values or JSON spellings for other scalars. In nullable string enums, the literal string `"null"` uses the quoted map key `"\"null\""` to distinguish it from actual null's `"null"` key. Prefer the typed array if keys matter.

Each ordering receives a real model evaluation. Scores are normalized separately, remapped to original values, then **probabilities** are averaged. `auto` uses canonical ordering and a Williams design: N evaluations for an even N, 2N for an odd N greater than one. Integer budgets sample distinct orderings reproducibly using request `seed`; the identity is included. `all` visits every permutation. A hard cap of 720 prevents accidental factorial workloads. A six-sided die with `all` costs 720 evaluations; opt in deliberately. Averaging may reduce position bias but does not guarantee correct reasoning or calibrated confidence. This policy is separate from `evaluate`'s older opt-in cyclic logit averaging.

### Thinking and runtime capabilities

Default `generate` thinking is false, independently of the load-time policy used by `evaluate`. A field can override the request. Budget-only field overrides inherit the request's on/off setting. Effort selects the setting explicitly. Contradictory settings fail validation. `auto` asks the model to choose none/low/medium/high as an additional hidden finite decision, exposes that selection in `thinking_effort`, and then performs the actual field request. It cannot also specify a fixed effort or budget.

| Runtime | Finite fields | Text/numbers | Reasoning |
|---|---|---|---|
| Native llama.cpp/GGUF | Candidate logits | JSON grammar plus Rust type checks | Bounded text-model reasoning using the model's actual template |
| Browser llama.cpp/WASM | Candidate log probabilities | JSON Schema grammar plus Rust type checks | Explicit unsupported error |
| Transformers.js text | Token logits | Bounded greedy generation, validated afterward; no grammar mask | Explicit unsupported error |
| Existing custom scorer/fixture | Finite scoring | Requires implementing `infer_field` / `inferField` | Requires that hook |

Native and browser GGUF field generation accept the existing image/audio/video-frame envelope when the loaded model and matching projector support the modality. That is not a promise of receipt-reading accuracy. Native media reasoning is explicitly unsupported. Video inputs are decoded, timestamped image frames; TurboJev does not decode video containers or invoke external media executables. Current published native binaries cover Windows x86/x64/ARM64; portable source and WASM support do not imply Linux/macOS binary releases or execution on every architecture.

`usage.decisions` counts completed output fields, not hidden effort queries or permutation evaluations. `input_tokens` sums known actual model counts; `generated_tokens` counts actual generated reasoning/value tokens. `thinking` contains actual model reasoning where supported. Increase context and reduce `nSeqMax` for longer reasoning. Unsupported modes, context overflow and unfinished reasoning return errors; no fallback answer is fabricated.

### Binding calls

| Language | Method |
|---|---|
| Rust | `engine.generate(GenerateRequest)`; types in `turbojev::schema::fields` |
| Python | `client.generate(request)` or `client.generate(context=..., questions=...)` |
| Node/TypeScript | `await client.generate(request)`; exported `GenerateRequest` / `GenerateResponse` |
| Browser | `await client.generate(request)` or `await wasmEngine.generate_json(JSON.stringify(request))` |
| .NET | `await engine.GenerateJsonAsync(JsonSerializer.Serialize(request))` |
| C | `tj_engine_generate_json(handle, json, &output)` then `tj_string_free(output)` |
| Android | `engine.generate(JSONObject(request))` |
| Swift | `try engine.generate(json)` on its managed affinity thread |

The native ABI is shim version 10; rebuild native dependents together. Runtime options are scoped to each field and restored after errors. Existing `evaluate` requests and responses remain supported.



Wllama 3.8.1 exposes top-N vocabulary scores. The adapter first requests 128 scores; if a permitted label is missing, it queries the actual vocabulary size from the loaded model and requests all scores with the same prompt. This recovery adds one model call and counts its actual input tokens. No missing candidate receives an invented probability. The core projects 1–24 candidates and normalizes their scores; the browser adapter does not claim a selective output head.
