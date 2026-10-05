# Typed fields and workflows

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

Wllama 3.8.1 exposes top-N vocabulary scores and its Jinja autoparser does not apply custom GBNF on this path. The adapter resolves letter labels through the loaded model tokenizer using textual logit-bias keys, applies the same additive offset to every permitted label, and requests post-sampling probabilities with truncation disabled. A common offset preserves candidate log-odds and the conditional distribution; it does not favor the first option or select a winner. This uses one model call for 2–24 candidates. Missing or invalid scores are errors, never invented probabilities. The browser adapter does not claim a selective output head.

## Dependencies, conditions and null

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

## Probabilities and permutation budgets

```json
{"type":"string","enum":["heads","tails"],"instructions":"Which side of a fair coin will appear?","permutations":"all","return_probabilities":true}
```

The result wrapper contains `value`, `confidence`, `probabilities` and `distribution`. `distribution` is an array of `{value, probability}` retaining actual JSON types. Map keys normally use string values or JSON spellings for other scalars. In nullable string enums, the literal string `"null"` uses the quoted map key `"\"null\""` to distinguish it from actual null's `"null"` key. Prefer the typed array if keys matter.

Each ordering receives a real model evaluation. Scores are normalized separately, remapped to original values, then **probabilities** are averaged. `auto` uses canonical ordering and a Williams design: N evaluations for an even N, 2N for an odd N greater than one. Integer budgets sample distinct orderings reproducibly using request `seed`; the identity is included. `all` visits every permutation. A hard cap of 720 prevents accidental factorial workloads. A six-sided die with `all` costs 720 evaluations; opt in deliberately. Averaging may reduce position bias but does not guarantee correct reasoning or calibrated confidence. This policy is separate from `evaluate`'s older opt-in cyclic logit averaging.

## Thinking and runtime capabilities

Default `generate` thinking is false, independently of the load-time policy used by `evaluate`. A field can override the request. Budget-only field overrides inherit the request's on/off setting. Effort selects the setting explicitly. Contradictory settings fail validation. `auto` asks the model to choose none/low/medium/high as an additional hidden finite decision, exposes that selection in `thinking_effort`, and then performs the actual field request. It cannot also specify a fixed effort or budget.

| Runtime | Finite fields | Text/numbers | Reasoning |
|---|---|---|---|
| Native llama.cpp/GGUF | Candidate logits | JSON grammar plus Rust type checks | Bounded text-model reasoning using the model's actual template |
| Browser llama.cpp/WASM | Candidate log probabilities | JSON Schema grammar plus Rust type checks | Explicit unsupported error |
| Transformers.js text | Token logits | Bounded greedy generation, validated afterward; no grammar mask | Explicit unsupported error |
| Existing custom scorer/fixture | Finite scoring | Requires implementing `infer_field` / `inferField` | Requires that hook |

Native and browser GGUF field generation accept the existing image/audio/video-frame envelope when the loaded model and matching projector support the modality. That is not a promise of receipt-reading accuracy. Native media reasoning is explicitly unsupported. Video inputs are decoded, timestamped image frames; TurboJev does not decode video containers or invoke external media executables. Current published native binaries cover Windows x86/x64/ARM64; portable source and WASM support do not imply Linux/macOS binary releases or execution on every architecture.

`usage.decisions` counts completed output fields, not hidden effort queries or permutation evaluations. `input_tokens` sums known actual model counts; `generated_tokens` counts actual generated reasoning/value tokens. `thinking` contains actual model reasoning where supported. Increase context and reduce `nSeqMax` for longer reasoning. Unsupported modes, context overflow and unfinished reasoning return errors; no fallback answer is fabricated.

## Binding calls

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

Native `nSeqMax=1` works for scalar workflows and multi-question `evaluate`.
Multi-question evaluation uses independent reference scoring at that capacity.
Shared-prefix reuse requires at least two sequences (an immutable prefix and a
suffix); parallel suffix reuse requires at least three. The runtime advertises
only the optimizations its loaded context can perform. This changes scheduling,
not the selected CPU/GPU device or the meaning of the answers.

## Tests and evals adapted from TypeLLM

| Reference tests | TurboJev coverage |
|---|---|
| Schema, nullable, grammar fields, text | Scalar and enum typing, duplicate/invalid values, null wire presence, bounded native grammar generation |
| Dependencies, conditions | Forward references, cycles, typed operators, plain prior values, skipped descendants, skipped auto effort, media-envelope preservation |
| Permutations, probabilities | Unique seeded samples, Williams positions/neighbours, normalization and probability averaging, typed null distribution |
| Thinking | Validated flags, efforts and budgets, actual hidden effort calls, runtime errors rather than fabricated reasoning |
| Images | Existing media contracts plus field-generation path; model accuracy is measured separately |
| Batching, concurrency, prefill, serving protocols | Existing TurboJev worker/affinity and runtime tests remain; TypeLLM's SGLang-specific batching, remote-server and logprob-transport tests are not claimed as implemented here |

Run `cargo test --workspace --locked` and the browser runtime tests. `evals/run.py` records real numeric and JevBench model quality, errors, provenance and latency. TypeLLM's numeric cases and the three example ports have attribution and licenses; its published model results are not reused as TurboJev evidence.
