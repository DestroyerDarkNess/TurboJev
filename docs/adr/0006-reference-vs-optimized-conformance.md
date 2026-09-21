# ADR 0006 — Every optimized path must be comparable to a reference path

## Status
Accepted in v0.3.

## Decision
`turbojev-conformance` compares complete typed responses using explicit numerical tolerances. Runtime adapters must be able to benchmark optimized execution against an independent/reference execution path when the runtime supports both.

## Why
Shared-prefix caches, batched suffixes, lower precision, quantization and candidate-only output projection can introduce numerical drift. Performance improvements are accepted only when their semantic and numerical effects are measurable.

## Consequences
TurboJev reports decision agreement separately from probability/score deltas instead of treating bitwise equality as the only correctness criterion.
