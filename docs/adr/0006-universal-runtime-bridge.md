# ADR 0006: Universal runtime bridge

Status: accepted in v0.4.0.

## Decision

TurboJev separates semantic decisions from token-level model execution through `turbojev-runtime`. Runtime adapters do not implement Choice/Noul/Score, probability normalization, schema validation, candidate mapping, or execution planning. They provide only prompt-template/tokenizer encoding and candidate scoring.

The portable composition is:

```text
DecisionBatch
    -> semantic PromptDocument
    -> model PromptRenderer
    -> PromptCodec (chat template + tokenizer)
    -> verified candidate codebook at the exact answer boundary
    -> exact common-prefix factoring
    -> RuntimeExecutor
    -> raw candidate logits
    -> TurboJev core calibration and typed answers
```

## Why

This prevents TurboJev from becoming a GGUF, llama.cpp, WebGPU, CUDA, Metal, MLX, ONNX, Candle, or operating-system-specific library. A new runtime/model family can be added without changing the semantic core or public API.

## Candidate codes

Candidate codes are never hard-coded token IDs. `RuntimeBackend` searches deterministic opaque symbols and verifies each against the runtime's real chat-template answer boundary. Single-token codes are preferred. Multi-token codes are legal only when the executor declares exact sequence scoring.

## Shared prefixes

Prefix reuse is derived from the exact tokenized prompts, not from string assumptions. The bridge factors only the longest byte-independent token prefix that is literally common to all tasks. Reference mode can still disable reuse.

## Threading neutrality

Portable traits do not require `Send + Sync`. Some browser/WASM and accelerator host handles are intentionally thread-affine. Hosts that need thread-safe execution can require `ThreadSafeScoringBackend`; this is a host policy, not a TurboJev semantic requirement.
