# ADR 0008 — Runtime adapters require conformance evidence

Status: accepted in v0.5.0.

Concrete runtime adapters are accepted only after testing exact answer-boundary
tokenization, capability consistency, score shape/finite values, and numerical
parity against a reference execution path. Runtime-specific tests live in the
adapter crate; invariant logic lives in `turbojev-runtime-conformance`.
