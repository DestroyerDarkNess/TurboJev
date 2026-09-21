# ADR 0013 — Browser runtimes expose low-level codec/executor primitives

Status: accepted

## Context

A first browser prototype allowed JavaScript to implement the entire
`ScoringBackend`. That was convenient, but it moved candidate-code discovery and
prompt-boundary correctness outside `turbojev-runtime`. Different browser
plugins could therefore invent their own A/B/C token mappings or semantic
scoring behavior.

## Decision

`turbojev-wasm` composes `AsyncRuntimeBackend` in Rust. Browser plugins expose
only exact tokenizer/chat-template operations and already-tokenized runtime
scoring. Candidate code search, round-trip verification, shared-prefix factoring,
planning, calibration and typed answers stay in Rust.

The portable runtime layer therefore has both `PromptCodec` and
`AsyncPromptCodec`; the latter is framework-free and returns non-`Send` futures
suitable for Promise/worker-backed browser engines.

## Consequences

- Browser runtimes cannot silently hard-code candidate token IDs in TurboJev's
  generic path.
- WebGPU/JS handles may remain thread-affine.
- A third-party browser engine must expose enough low-level functionality to
  implement the codec/executor contract; a generation-only API is insufficient
  for the direct decision path.
- The same semantic/runtime conformance logic can be reused across native,
  browser and future mobile adapters.
