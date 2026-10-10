# Typed fields and workflows

The structured extensions on this page describe source version 0.32.0, awaiting registry publication. Existing published packages do not yet include these extensions.

`generate` adds JSON-typed fields to TurboJev's portable core. `evaluate` remains the direct `choice`, `noul` and ordered `score` API. Finite fields score model candidates; unrestricted strings and numbers generate a bounded value. Extraction therefore generates tokens and can take longer than a single candidate decision.

## Request and response

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

The same JSON request is accepted by every binding. Supply exactly one of `questions` (a field map) or `schema` (an object with `properties`, optional `required`, and optional `additionalProperties: false`). All declared properties are requested; `required` validates property names, not a separate omission policy. Conditions control omissions. The schema subset supports scalars, nested objects and top-level arrays. Arrays inside objects and nested arrays are rejected. `state` is an input alias for `context`.

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

Wllama 3.8.1 exposes top-N vocabulary scores and its Jinja autoparser does not apply custom GBNF on this path. The adapter resolves letter labels through the loaded model tokenizer using textual logit-bias keys, applies the same additive offset to every permitted label, and requests post-sampling probabilities with truncation disabled. A common offset preserves candidate log-odds and the conditional distribution; it does not favor the first option or select a winner. This uses one model call for 2–26 candidates. Missing or invalid scores are errors, never invented probabilities. The browser adapter does not claim a selective output head.

## Structured fields, descriptions and ordinal levels

Nested objects retain their shape in `result`. Scalar leaves are evaluated in dependency order;
local dependency names resolve within their containing object. An object-level condition applies
to its leaves. Object dependencies pass the available plain object and skip only when all its
leaves were skipped. Compiled metadata exposes complete paths such as `person.age`.

Top-level arrays accept scalar items or objects with scalar/nested-object leaves. Object items
are generated atomically under a JSON grammar. `minItems` defaults to 0; an explicit `maxItems`
is 1–50. Without a maximum, a call can append up to 50 new items to `continue_from`. Prefix items
are validated and retained. Non-nullable scalar arrays use null to stop once their minimum is
met. Originally nullable items retain actual null as data and use a separate append decision.
Duplicate deterministic items stop extraction; sampling permits two duplicate retries per array. A duplicate before
`minItems` returns an error. No host chooses or invents the items.

Use `choices: [{"value": "billing", "description": "Payments and refunds"}, ...]` instead of
`enum` for described options. A non-nullable `number` field can use 2–26 ordered
`levels: [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}]`; the result is
`sum(index * probability)`, not a rounded winner. Levels cannot be permuted or combined with enum.

Enums over 26 values require complete value-continuation scoring. Native GGUF and the text
ONNX adapter implement it; GGUF/WASM accepts a value-scorer plugin and otherwise errors.
The schema cap of 512 does not imply single-token latency or universal runtime support.

See the runnable JSON requests in `samples/typed-fields/structured.json`, `hierarchical.json`
and `discover-categories.json`, and the [cookbook](COOKBOOK.md).

## Dependencies, conditions and null

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

## Probabilities and permutation budgets

```json
{
  "type": "string",
  "enum": ["heads", "tails"],
  "instructions": "Which side of a fair coin will appear?",
  "permutations": "all",
  "return_probabilities": true
}
```

The result wrapper contains `value`, `confidence`, raw selected `probability`, `probabilities` and `distribution`. `distribution` is an array of `{value, probability}` retaining actual JSON types. Map keys normally use string values or JSON spellings for other scalars. In nullable string enums, the literal string `"null"` uses the quoted map key `"\"null\""` to distinguish it from actual null's `"null"` key. Prefer the typed array if keys matter.

Each ordering receives a real model evaluation. Scores are normalized separately, remapped to original values, then **probabilities** are averaged. `auto` uses canonical ordering and a Williams design: N evaluations for an even N, 2N for an odd N greater than one. Integer budgets sample distinct orderings reproducibly using request `seed`; the identity is included. `all` visits every permutation. A hard cap of 720 prevents accidental factorial workloads. A six-sided die with `all` costs 720 evaluations; opt in deliberately. Averaging may reduce position bias but does not guarantee correct reasoning or calibrated confidence. This policy is separate from `evaluate`'s load-time `none`, `cyclic` and `balanced` policies. Positive integer budgets include the caller listing and are not generally order invariant. Use `auto` for canonical balanced layouts. Numeric enums are unordered, unlike ordinal `score` or number fields with `levels`. Exact enum ties compare JSON value encodings. See [option ordering](ORDER_ROBUSTNESS.md).

Probability requests default to `permutations: "auto"`, unless explicitly set. Ordered levels
and large enums default to one pass. Set `permutations: 1` for a deliberate latency choice.
For an unordered finite field, confidence is `(p_max - 1/N) / (1 - 1/N)`. An ordinal field
uses modal-level distance relative to a uniform scale. Uniform uncertainty is zero. These
measures are not fitted calibration and do not prove correctness.

A gate such as `"when": {"team": {"confidence": {"gte": 0.8}}}` requires probability output
on `team`. Bounds must be numeric in 0–1; scalar value comparisons may be combined with them.

Request-level `temperature` defaults to 0. Positive temperature enables seeded finite sampling;
`seed` also controls layout selection. Confidence/distributions still describe the model scores.
Native GGUF samples open values; the ONNX adapter rejects seeded open-value sampling explicitly.

## Thinking and runtime capabilities

Default `generate` thinking is false, independently of the load-time policy used by `evaluate`. A field can override the request. Budget-only field overrides inherit the request's on/off setting. Effort selects the setting explicitly. Contradictory settings fail validation. `auto` asks the model to choose none/low/medium/high as an additional hidden finite decision, exposes that selection in `thinking_effort`, and then performs the actual field request. It cannot also specify a fixed effort or budget.

| Runtime                        | Finite fields               | Text/numbers                                                    | Reasoning                                                      |
| ------------------------------ | --------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| Native llama.cpp/GGUF          | Candidate logits            | JSON grammar plus Rust type checks                              | Bounded text-model reasoning using the model's actual template |
| Browser llama.cpp/WASM         | Candidate log probabilities | JSON Schema grammar plus Rust type checks                       | Explicit unsupported error                                     |
| Transformers.js text           | Token logits                | Bounded greedy generation, validated afterward; no grammar mask | Explicit unsupported error                                     |
| Existing custom scorer/fixture | Finite scoring              | Requires implementing `infer_field` / `inferField`              | Requires that hook                                             |

Native and browser GGUF field generation accept the existing image/audio/video-frame envelope when the loaded model and matching projector support the modality. That is not a promise of receipt-reading accuracy. Native media reasoning is explicitly unsupported. Video inputs are decoded, timestamped image frames; TurboJev does not decode video containers or invoke external media executables. Current published native binaries cover Windows x86/x64/ARM64; portable source and WASM support do not imply Linux/macOS binary releases or execution on every architecture.

`usage.decisions` counts completed output fields, not hidden effort queries or permutation evaluations. `input_tokens` sums known actual model counts; `generated_tokens` counts actual generated reasoning/value tokens. `thinking` contains actual model reasoning where supported. Increase context and reduce `nSeqMax` for longer reasoning. Unsupported modes, context overflow and unfinished reasoning return errors; no fallback answer is fabricated.

## Binding calls

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

Native `nSeqMax=1` works for scalar workflows and multi-question `evaluate`.
Multi-question evaluation uses independent reference scoring at that capacity.
Shared-prefix reuse requires at least two sequences (an immutable prefix and a
suffix); parallel suffix reuse requires at least three. The runtime advertises
only the optimizations its loaded context can perform. This changes scheduling,
not the selected CPU/GPU device or the meaning of the answers.

## Tests and evals adapted from TypeLLM

| Reference tests                                   | TurboJev coverage                                                                                                                                                             |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Schema, nullable, grammar fields, text            | Scalar and enum typing, duplicate/invalid values, null wire presence, bounded native grammar generation                                                                       |
| Dependencies, conditions                          | Forward references, cycles, typed operators, plain prior values, skipped descendants, skipped auto effort, media-envelope preservation                                        |
| Permutations, probabilities                       | Unique seeded samples, Williams positions/neighbours, normalization and probability averaging, typed null distribution                                                        |
| Thinking                                          | Validated flags, efforts and budgets, actual hidden effort calls, runtime errors rather than fabricated reasoning                                                             |
| Images                                            | Existing media contracts plus field-generation path; model accuracy is measured separately                                                                                    |
| Batching, concurrency, prefill, serving protocols | Existing TurboJev worker/affinity and runtime tests remain; TypeLLM's SGLang-specific batching, remote-server and logprob-transport tests are not claimed as implemented here |

Run `cargo test --workspace --locked` and the browser runtime tests. `evals/run.py` records real numeric and JevBench model quality, errors, provenance and latency. TypeLLM's numeric cases and the three example ports have attribution and licenses; its published model results are not reused as TurboJev evidence.

## Language examples and migration

The self-contained [TurboJev skill](../.agents/skills/turbojev/SKILL.md) includes installation, configuration, and `evaluate` / `generate` examples for Rust, Node/TypeScript, Python, C, .NET, browser/WASM, Android and Swift. It is maintained directly as Markdown and includes the complete request catalogue inline.

When migrating an application, keep the two response contracts distinct: `evaluate` returns `answers`; `generate` returns `result`, `skipped` and `usage`. A skipped conditional field is absent, while a nullable field can contain actual JSON null. Use a stable semantic key for each choice, and preserve the original criterion order for a score. The load-time `execution.orderRobustness` option changes `evaluate` choices only; generated enums use their own per-field `permutations`.

Native published packages currently include Windows x86/x64/ARM64 assets. Mobile bindings and other native operating systems require a compatible source build. Samples use checkout artifacts; the separate TurboJev.Demos repository uses public packages/CDNs and self-contained Python folders.

## SDK helpers and controls

Python exports `Choice`, `Decision`, `ArrayField`, `Generation`, `Usage`, `SchemaError`,

`compile_json_schema`, `candidate_softmax`, `run_schema`, `example_schema`, `encode_image`

and `benchmark_prefix_cache`. `generate` still returns a dictionary; `generate_typed` returns

attribute access. Schema compilation and candidate normalization need no loaded model.

Node exports model-free `compileJsonSchema` and `candidateSoftmax`; browser WASM exports

`compile_schema_json` and `candidate_softmax_json`. C exposes `tj_compile_schema_json` and

`tj_candidate_softmax_json`. C# adds `CompileSchema`, `CandidateSoftmax`, `GenerateAsync`

and typed `Generation`/usage metadata. All JSON methods continue to work.

Python controls use `client.generate(request, timeout=30)` or a `GenerationControl` that another

thread can cancel. Node/browser calls accept `{ timeoutMs: 30000, signal }`; C# accepts a

`TimeSpan` and `CancellationToken`. Native cancellation is checked before/after each complete

model operation. An in-flight decode must finish; this is not a hard interrupt. Browser operations

are serialized and cancellation messages bypass that queue. Engines remain usable afterward.

The optional `images` request list contains 1–32 encoded images in caller order. Python accepts

bytes, paths, data URIs or PIL-like objects and encodes them before Rust validation. URLs must be

fetched explicitly. Browser `imageEvidence(blobs, text)` encodes an image envelope. A compatible

vision model and matching projector remain required; no text-only fallback is silently used.

The [0.6.9 port matrix](TYPELLM_0_6_9_PORT.md) distinguishes features, runtime limits and

vendor-specific SGLang transports from the portable SDK.

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
