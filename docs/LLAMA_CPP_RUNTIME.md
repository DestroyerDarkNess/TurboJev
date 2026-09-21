# llama.cpp runtime adapter (v0.25)

`turbojev-runtime-llamacpp` is TurboJev's first concrete local runtime adapter.
It is **not** a dependency of `turbojev-core`; applications opt into it explicitly.

## Supported execution paths

The adapter preserves three auditable scheduling plans over the same prompt/token contract:

1. **Reference** — independent full prompt per decision, serial execution,
   llama.cpp full-vocabulary logits, then gather candidate rows.
2. **Deterministic shared-prefix reuse** — prefill the exact common token prefix
   once, then score each suffix serially from an isolated branch.
3. **Bounded parallel suffixes** — prefill once, clone the immutable prefix into
   distinct branch sequence IDs, score a bounded group together, then discard
   every branch.

All three paths:

- use verified single-token candidate codes;
- emit zero autoregressive answer tokens;
- preserve the same TurboJev schema/calibration path;
- keep full-vocabulary projection.

v0.9 adds a candidate-only API contract and parity fixtures. The stock FFI implementation deliberately keeps `selective_output_head = false` until it can genuinely skip the full vocabulary projection; custom/future `LlamaCppApi` implementations may enable it only after conformance.
Multi-token sequence scoring remains a later adapter capability.

## Layering

```text
TurboJev semantic core
        |
RuntimeBackend
        |
LlamaCppCodec + LlamaCppExecutor
        |
LlamaCppApi
        |
optional FFI implementation
        |
turbojev_llama_shim ABI v5
        |
llama.cpp
        |
GGUF model
```

The Rust adapter depends on `LlamaCppApi`, not directly on llama.cpp declarations.
Upstream ABI changes are contained by the C++ shim.

## Shared-prefix lifecycle

TurboJev renders and tokenizes each complete task prompt first. The universal
runtime bridge computes the literal longest common **token** prefix; no
string-level cache assumption is allowed.

Sequence 0 is the immutable shared base. Serial mode copies it to sequence 1 for
each task. Parallel mode copies it to sequences `1..N` for one bounded group:

```text
sequence 0: common prefix (immutable)
      |
      +--> seq 1 + suffix A --\
      +--> seq 2 + suffix B ----> one llama_decode -> ordered score rows
      +--> seq 3 + suffix C --/
      |
      `--> remove seq 1..N; keep seq 0
```

If a suffix is empty, its candidate logits come from the retained final-prefix
logits and it does not need a synthetic decode token. Mixed empty and non-empty
branches preserve original request order.

If the exact token LCP is empty, correctness wins and the executor falls back to
independent scoring.

## Bounded scheduling

`LlamaCppExecutor` advertises parallel suffix support only when the loaded runtime
has reusable sequence memory and at least two branch sequence slots. Sequence 0
is excluded from the reported branch capacity.

Automatic planning bounds a parallel batch by:

- request task count;
- runtime `max_parallel_tasks` derived from `n_seq_max - 1`;
- caller `ExecutionBudget.max_parallel_tasks` when supplied;
- optional caller memory budget divided by an optional conservative per-branch
  estimate supplied through `FfiLlamaCpp::with_estimated_parallel_task_bytes`.

No CUDA/Metal/Vulkan/OS concept enters the portable planner.

The stock FFI capability set is:

```text
next_token_logits       true
sequence_scores         false
shared_prefix           runtime-dependent
parallel_suffixes       runtime-dependent
selective_output_head   false
local_inference         true
```


## Cross-request multi-root path (v0.25)

v0.25 adds a fourth, semantically distinct path for independent requests. TurboJev flattens eligible single-token decision tasks from several requests into independent sequence roots and submits those roots in one native decode operation:

```text
request A / task 0 -> seq 0 -> prompt A -> candidate logits A
request B / task 0 -> seq 1 -> prompt B -> candidate logits B
request B / task 1 -> seq 2 -> prompt C -> candidate logits C
                         \________ one llama_decode ________/
```

No prefix/KV is copied between roots. The shim clears sequence memory before and after the fused call. Output logits are retrieved using the last prompt token's batch index, matching current llama.cpp `llama_get_logits_ith` semantics.

The FFI path is compiled and native-smoke tested, but host capability activation is off by default. Set `execution.llamacppMultiRootFused=true` only after the concrete deployment passes TurboJev's independent-vs-fused conformance. This requirement is intentional because backend-specific batched-logit correctness cannot be inferred from API support alone.

Eligibility is checked per group. The shim queries the effective `llama_n_ctx`, `llama_n_ctx_seq`, `llama_n_batch`, and `llama_n_seq_max` after context creation rather than trusting requested values. Multi-token candidate codebooks, candidate-only projection, too many roots, an overlong individual root, or too many total prompt tokens automatically remain on serial cross-request execution.

## Prompt and tokenizer correctness

The native shim asks the loaded GGUF for its default llama.cpp chat template.
TurboJev validates every candidate at the exact assistant answer boundary:

```text
encode(chat_without_answer) + candidate_suffix
==
encode(chat_with_complete_candidate_answer)
```

Candidates must decode back to their exact opaque symbol. This adapter currently
accepts a codebook only when every candidate is exactly one token.

Human-language control text is adapter configuration, never portable-core policy.
Explicit `choice` / `noul` / `score` instructions from callers are preserved
verbatim.

## Native shim build

```sh
cmake -S crates/turbojev-runtime-llamacpp/native \
      -B build/turbojev-llama \
      -DTURBOJEV_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build/turbojev-llama --config Release
```

Point Cargo at the directory containing the resulting shared library and run one
of the adapter examples. `parallel_suffix_gguf` demonstrates `ExecutionBudget`
and Auto planning.

Packaging/link resolution belongs to the host/runtime package and never leaks
into portable core.

## Conformance gates

The v0.6 independent path and v0.7 cached-serial path remain correctness
baselines for parallel and selective execution. Tests cover:

- two distinct tokenizer/chat-template family fixtures;
- exact candidate answer-boundary validation;
- reference-vs-shared-vs-parallel candidate-logit parity;
- portable task/memory budget bounds;
- typed-answer parity at the TurboJev engine layer;
- native ABI compile/link/execute smoke testing;
- mixed empty/non-empty branch output ordering;
- 48 rounds of native cache-isolation stress over 2..7 parallel branches;
- a stateful native stub whose logits depend on emulated sequence contents.

The native smoke build uses `-Wall -Wextra -Wpedantic -Werror`.


## v0.9 selective-path boundary

`LlamaCppApi` now exposes selected variants for independent, shared-prefix, and
parallel suffix scoring. `LlamaCppExecutor` dispatches to those methods only
when `selective_output_head_supported()` is true. The optimized result reports
`candidate_rows_projected`; `verify_selective_projection` compares it to the
full-vocabulary path and rejects missing/inconsistent row accounting.

This is intentionally stricter than gathering A/B/C from ordinary llama.cpp
logits and calling that candidate-only. See `CANDIDATE_PROJECTION.md`.
