# Candidate-only output projection

TurboJev v0.9 adds a portable selective-output-head contract. The optimization
is intentionally defined independently of llama.cpp, GGUF, CUDA, Metal, WebGPU,
or any model family.

## Semantics

For a normal causal language model, the next-token vocabulary logits are:

```text
h = final post-normalization hidden state
W = output-head matrix [vocab, hidden]
b = optional output bias
s = optional model output scale

full_logits = s * (W @ h) + b
```

A TurboJev decision needs only a small candidate set, for example token IDs
`[A, B, C]`. A true selective implementation computes:

```text
candidate_rows = W[[A, B, C], :]
candidate_bias = b[[A, B, C]]
logits = s * (candidate_rows @ h) + candidate_bias
```

It must **not** compute `full_logits` and merely gather A/B/C afterwards. That
would produce the right values but would not be a candidate-only projection.

## Portable crates

`turbojev-projection` owns deterministic reference math:

- `DenseCandidateRows`
- `project_candidates_f32`
- `project_full_vocab_f32`
- `gather_candidate_rows_f32`
- `compare_logits`

`turbojev-runtime-projection` is a model/runtime-neutral executor. A concrete
runtime implements `DenseProjectionSource` and supplies:

1. the ordinary full-vocabulary/gather path used as a correctness baseline;
2. the final post-normalization hidden state;
3. only the requested output-head rows;
4. optional output scale/bias through the row/config objects.

The executor then performs the candidate matrix multiplication without knowing
the model format or operating system.

Device-native runtimes do not have to copy hidden states or weights to CPU.
They may implement an equivalent fused/device-side path directly. The portable
implementation exists as a reference contract and conformance oracle.

## Capability truthfulness

A runtime may set:

```text
selective_output_head = true
```

only if it has a real selective path. The optimized output must also report:

```text
candidate_rows_projected = sum(number of candidates in every task)
```

`turbojev-runtime-conformance::verify_selective_projection` verifies both:

- numerical parity against the full-vocabulary reference;
- exact candidate-row accounting.

This prevents an adapter from advertising full-vocabulary + gather as a
selective implementation.

## Numerical parity

Exact equality is appropriate for deterministic dense fixtures. Real runtimes
may use quantized weights, fused kernels, BF16/FP16 or different reduction
orders, so conformance accepts an explicit absolute tolerance. The tolerance is
part of the benchmark/conformance invocation; TurboJev does not silently relax
it.

The full-vocabulary path from v0.6-v0.8 remains the reference path.

## llama.cpp status

The stock llama.cpp adapter keeps `selective_output_head = false` unless its
low-level API implementation can genuinely bypass the ordinary vocabulary
head. The current public llama.cpp API exposes next-token logits and final
embeddings, but a generic stable public API for running the model while
selectively projecting arbitrary output rows is not assumed by TurboJev.

`LlamaCppApi` nevertheless contains explicit selected-path methods. This lets a
custom/future llama.cpp integration implement the optimization without changing
TurboJev core. The fixture implementation exercises candidate-only projection
combined with shared-prefix and bounded-parallel scheduling and must pass parity
before the capability is advertised.

## Multi-token candidates

Candidate-only output-head projection is a next-token optimization. A candidate
code that spans multiple tokens requires conditional sequence scoring. That is
a separate execution primitive and remains in the later multi-token milestone.
TurboJev never treats a multi-token sequence as one vocabulary row.
