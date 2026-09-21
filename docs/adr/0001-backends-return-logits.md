# ADR 0001: Backends return logits, not public probabilities

Status: Accepted

## Context

Different inference runtimes expose different model internals. If each backend performs softmax, calibration and answer construction, public behavior can drift across llama.cpp, MLX, WebGPU and Rust-native implementations.

## Decision

`ScoringBackend::score` returns ordered raw candidate logits only. `turbojev-core` owns:

- finite-value validation
- probability normalization
- calibration
- deterministic winner selection
- `choice`, `noul`, and `score` construction

## Consequences

- one semantic implementation across every backend and binding
- calibration can evolve without touching GPU/runtime code
- backend parity tests compare a smaller, clearer contract
- hosted compatibility adapters remain outside model execution
