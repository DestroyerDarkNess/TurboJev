# TypeLLM 0.6.9 SDK port

This change adapts the public schema/workflow features of TypeLLM 0.6.9 to
TurboJev's Rust decision core. It does not bundle TypeLLM, call its service,
redistribute private weights, or reproduce its hosted benchmarks.

References reviewed: the local `TypeLLM-0.6.9` source, schema/runtime/image/
benchmark modules and their structured, score, condition, choice and value-scorer
tests; all four articles listed in the [cookbook](https://typellm.ai/cookbook)
when this port was prepared on 2026-10-10.

## Features and equivalents

| TypeLLM feature / public name                                | TurboJev equivalent                                                                                                                                              |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeLLMClient`                                              | `TurboJev` / `TurboJevEngine` / `TurboJevWeb`; existing local runtimes remain in use                                                                             |
| Nested object fields                                         | Rust flattens scalar decisions, resolves local dependency scopes and reconstructs nested results                                                                 |
| Object dependency and `when` inheritance                     | An object gates every leaf. A dependent object group is unavailable only if all its leaves were skipped                                                          |
| Described `choices`                                          | Typed `{value, description}` options; descriptions stay attached to values through permutations                                                                  |
| Ordinal `levels`                                             | Number field with 2–26 labels; the expected zero-based index preserves the original order                                                                        |
| Top-level scalar/object arrays                               | Rust manages append decisions, grammar-constrained objects, bounds, duplicate detection and continuation                                                         |
| `continue_from`                                              | Validated existing items are preserved; without `maxItems`, at most 50 new items are appended per call                                                           |
| Confidence `when`                                            | `gt`, `gte`, `lt`, `lte` on a parent's returned confidence; requires `return_probabilities` on that parent                                                       |
| New confidence measure                                       | Choice strength above uniform; ordinal dispersion around the modal level. Neither is fitted calibration                                                          |
| Probability default permutations                             | `return_probabilities: true` selects `auto` unless explicitly overridden; ordinal fields and large enums stay one-pass                                           |
| Boolean permutations                                         | Explicit permutation budgets apply to boolean decisions as well as enums                                                                                         |
| 26 control labels                                            | Small finite field codebooks allow 26 candidates; labels are verified with the actual tokenizer                                                                  |
| `MAX_ENUM_CHOICES` / value scorer                            | Schema limit 512; native GGUF and ONNX score complete value continuations for >26 values; GGUF/WASM accepts an explicit value-scorer plugin                      |
| `temperature` and `seed`                                     | Rust reports temperature-adjusted probabilities and performs seeded finite sampling; native GGUF supports seeded open-value sampling; greedy remains the default |
| Images list                                                  | Up to 32 images in caller order; native mtmd and GGUF/WASM retain every image in model input                                                                     |
| `compile_json_schema`, `Decision`, `ArrayField`              | Model-free Rust compiler; typed Python descriptions, TS metadata and C# compiled decisions                                                                       |
| `Choice`                                                     | Python described-choice builder; runtime control labels are deliberately not exposed as semantic answers                                                         |
| `Generation`, `Usage`                                        | Typed Python and C# result wrappers; TS response interfaces; existing JSON methods remain available                                                              |
| `candidate_softmax`                                          | One Rust double-precision normalizer, exposed through Python, Node, C/.NET and WASM                                                                              |
| `run_schema`, `example_schema`                               | Python one-call load/generate/close and a complete support-routing schema                                                                                        |
| `GenerationCancelled`, `GenerationTimeout`                   | Python exceptions, native host error codes, Node `AbortSignal`, C control handles and C# cancellation tokens                                                     |
| Browser cancellation/timeouts                                | Worker-owned controls check before/after actual runtime calls; overlapping operations on a loaded runtime are serialized                                         |
| `benchmark_prefix_cache`                                     | Python compares independent-reference and negotiated shared-prefix evaluation in separate loaded sessions, with warm-up and measured timings                     |
| SGLang HTTP clients, errors, cache flushing and server pools | Vendor-specific transport remains an optional external runtime integration. It is not renamed to pretend to be a bundled local backend                           |

The benchmark compares TurboJev execution plans, not SGLang's growing-prefix
radix cache. Missing cache-token counters remain `None`; a runtime that does
not actually select shared-prefix reuse fails the comparison explicitly.
It returns both answer payloads and a numerical parity check, not just a speed
ratio. A faster run whose probabilities exceed the reported tolerance is not
a qualified optimization.

## Runtime boundaries

| Operation                                   | Native GGUF                                          | Transformers.js / ONNX                   | llama.cpp / WASM                                      |
| ------------------------------------------- | ---------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| Finite fields up to 26 values               | Verified candidate codes                             | Verified candidate codes                 | Candidate log probabilities                           |
| Text enum 27–512                            | Full continuation sequence scoring                   | Full continuation sequence scoring       | Requires `useChoiceScorer` or `choiceScorerModuleUrl` |
| Open strings/numbers and object-array items | JSON grammar and Rust validation                     | Bounded greedy JSON, validated afterward | JSON Schema grammar and Rust validation               |
| Seeded open-value sampling                  | Supported                                            | Explicit unsupported error               | Uses Wllama completion temperature/seed               |
| Per-field thinking                          | Native text models                                   | Explicit unsupported error               | Explicit unsupported error                            |
| Image lists                                 | Compatible GGUF plus projector                       | Use a multimodal adapter                 | Compatible GGUF plus projector                        |
| Large multimodal enum                       | Explicit unsupported error on built-in sequence path | Explicit unsupported error               | Requires a media-aware value scorer                   |

The Rust schema cap does not promise that every runtime/model can perform every
operation. Exact sequence scoring uses the full set of options and the full
continuation, including its closing delimiter. It may require substantially
more work than a single-token decision. Unsupported paths fail; no fallback
probabilities or model answers are fabricated.

Arrays inside objects, nested arrays, ordinal/probability item fields and
per-property workflow options inside object-array items are unsupported.
Numeric enums remain unordered. TypeLLM rejects singleton enums; TurboJev
retains its existing constant-field behavior without inference.

## Observable changes

- `generate` confidence is now distinct from the selected option's raw
  `probability`. Existing `evaluate` calibration/response semantics are retained.
- Asking for probabilities may now trigger multiple actual model evaluations.
  Set `permutations: 1` explicitly when choosing latency over order averaging.
- Nested JSON properties preserve caller order. Execution still follows the
  dependency graph; output order does not expose that scheduling.
- Duplicate arrays stop deterministic extraction. Sampling permits two duplicate retries per array;
  a duplicate before `minItems` is an error rather than an invented item.
- Native timeout/cancellation is cooperative between complete model operations.
  It does not interrupt an in-flight prefill or decode. The engine remains
  thread-affine and can be reused after the aborted workflow returns.
- Python releases the GIL during controlled native generation through a private
  creator-thread guard. Loaded sessions cannot be shared across Python threads;
  cancellation controls can be cancelled from a different thread.

## Cookbook ports

- [Dense pages](https://typellm.ai/cookbook/dense-pages): image evidence, bounded
  object arrays and `continue_from`; see [TurboJev cookbook](COOKBOOK.md).
- [More than 255 classes](https://typellm.ai/cookbook/classification-over-255-classes):
  512-value schema cap and complete continuation scorer.
- [Hierarchical classification](https://typellm.ai/cookbook/hierarchical-classification):
  [hierarchical request](../samples/typed-fields/hierarchical.json).
- [Discover invoice categories](https://typellm.ai/cookbook/discover-invoice-categories):
  [category discovery request](../samples/typed-fields/discover-categories.json).

## Validation

Reproduce the new schema/core contract tests with:

```sh
cargo test -p turbojev-schema -p turbojev-core --test structured --locked
cargo test -p turbojev-host-native --test sdk --locked
```

Real native checks use the built C ABI and a real Qwen3-0.6B Q4_K_M GGUF:

```powershell
python samples/build.py native --arch x64 --download-model
python scripts/validate-sdk-evolution.py `
  --model C:/models/Qwen3-0.6B-Q4_K_M.gguf `
  --library samples/.artifacts/win-x64/turbojev_capi.dll `
  --library-dir samples/.artifacts/win-x64 `
  --output .artifacts/sdk-evolution/report.json
```

Use the model path printed by the builder, or an existing copy of the same
checkpoint. The builder recompiles the shim and packages its matching runtime
dependencies; do not point a freshly built C ABI at a stale installed shim.

The report separates contract failures from answer accuracy and records actual
latency. Compiler/fixture binding checks are in `validate-sdk-bindings.py`,
`validate-sdk-node.mjs` and `tests/e2e/dotnet-sdk`. They are not evidence of model
intelligence. Actual extraction and the 32-value shared-prefix category case
were additionally exercised with the real GGUF. Schema acceptance of 512 values
is not a 512-class accuracy benchmark.

A real CPU WASM extraction retained the nested name/age but returned an empty tag array for the two stated tags. This is recorded as an accuracy failure, not a passing extraction benchmark. Native extraction cases passed separately.

The CPU prefix-cache comparison selected the same routing category but changed
the boolean probability substantially. It failed numerical parity and does not
qualify shared-prefix execution for this build/model. Use independent reference
execution when exact probability behavior matters. The measured records,
including failed accuracy/parity checks, are retained in
[the SDK validation report](validation/sdk-evolution-0.32.0.json).

Windows x64 validation does not certify Windows x86/ARM64, other operating
systems, browser hardware or a model's vision accuracy. Registry demos keep
using published packages; new APIs must not be advertised as already published.
