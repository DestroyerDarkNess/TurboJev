# ADR 0009: Establish a concrete reference runtime before optimization

## Status

Accepted in v0.6.

## Context

Shared-prefix reuse, batched suffixes, quantized execution and selective output
projection can change floating-point results or tokenizer/cache behavior. A fast
implementation without a concrete local reference makes regressions hard to
attribute.

## Decision

The first concrete runtime adapter is `turbojev-runtime-llamacpp`, implemented
as an optional outer crate. Its v0.6 advertised capabilities are intentionally
limited to independent, serial, full-vocabulary, single-token candidate
scoring. It exposes a stable TurboJev-owned C shim around llama.cpp.

No optimized capability is advertised until its output is compared against the
reference path using the runtime/canonical conformance tooling.

## Consequences

- the portable core still has no llama.cpp dependency;
- v0.6 prioritizes correctness over throughput;
- later optimizations have a measurable baseline;
- llama.cpp API churn is isolated in one native shim instead of leaking through
  Rust/public language bindings.
