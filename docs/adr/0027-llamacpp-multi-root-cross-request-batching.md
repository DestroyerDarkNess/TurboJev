# ADR 0027 — llama.cpp multi-root cross-request batching

Status: accepted for v0.25.

## Context

TurboJev already uses llama.cpp sequence IDs for two different intra-request optimizations: shared-prefix cached serial scoring and parallel suffix scoring. Those paths start from one semantic `EvaluationRequest` and therefore do not prove that arbitrary independent requests can be fused into one runtime call.

llama.cpp's public batch API allows each token to carry an independent sequence ID. Its official `parallel` example also submits work for multiple clients through one `llama_decode` call. That is the primitive TurboJev needs for a true cross-request path.

Correctness is backend-specific, however. A native API accepting multiple sequence IDs is not itself proof that candidate logits match independent execution on every accelerator/backend/build. TurboJev's reference-path rule therefore applies before capability activation.

## Decision

The TurboJev-owned llama.cpp shim ABI is bumped to v5 and gains an explicit multi-root operation:

```text
tj_llama_multi_root_supported
tj_llama_max_multi_root_tasks
tj_llama_max_multi_root_tokens
tj_llama_score_root_batch
```

One TurboJev decision task maps to one independent llama.cpp sequence root:

```text
root 0 -> seq_id 0 -> positions 0..N -> logits at final prompt token
root 1 -> seq_id 1 -> positions 0..M -> logits at final prompt token
...
```

All roots are submitted in one `llama_decode` call. No root copies KV state from another root. The shim clears sequence memory before and after the operation so fused calls cannot contaminate subsequent independent scoring.

The output row is requested using the final token's **batch token index**, matching current `llama_get_logits_ith` semantics. The native test stub mirrors this mapping so row-index mistakes cannot pass smoke tests.

The first v1 multi-root path supports only single-token candidate scoring. Exact multi-token candidates continue through TurboJev's sequence-scoring path and cause shape-aware fallback to serial cross-request execution.

## Capability activation

The native FFI implementation contains the multi-root operation, but host activation is conservative:

```text
execution.llamacppMultiRootFused = false   # default
```

A host may opt in only after the concrete model + llama.cpp revision + backend + hardware combination passes fused-vs-independent conformance. `FfiLlamaCpp` therefore exposes the capability only after explicit enablement.

The server also asks `NativeSession::can_evaluate_fused()` for each candidate group. Global capability does not override per-shape limits such as root count, total prompt tokens, projection mode, or multi-token candidates.

## Resource bounds

The shim reports hard structural limits from the loaded context:

- maximum roots from effective `llama_n_seq_max`;
- maximum total prompt tokens from effective `llama_n_batch`/`llama_n_ctx`;
- maximum tokens per individual root from effective `llama_n_ctx_seq`.

TurboJev does not fabricate a memory estimate from model family or accelerator. `estimatedFusedItemBytes` stays absent unless a host supplies a conservative measurement.

## Consequences

- llama.cpp has a real multi-root cross-request implementation instead of re-labelling parallel suffixes.
- Default deployments remain serial across requests until target-specific evidence is provided.
- The independent path remains the correctness oracle.
- Native smoke tests cover unequal root lengths, candidate sets, repeated fused calls, and post-fused cache isolation.
- Hard/cooperative cancellation remains disabled; llama.cpp abort behavior is not treated as universally safe across accelerator backends.
