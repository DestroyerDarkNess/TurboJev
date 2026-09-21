# ADR 0026 — Fused runtime batching requires an explicit backend/runtime SPI and parity evidence

Status: Accepted for TurboJev v0.24.

## Context

Phase 11 needs cross-request batching without confusing it with the already-existing parallel suffix path inside a single `EvaluationRequest`. Those are different operations: independent requests have independent prompt roots, candidate codebooks, execution plans and typed responses.

## Decision

1. `turbojev-backend` defines `FusedScoringBackend` and `FusedBackendCapabilities` separately from ordinary `ScoringBackend` capabilities.
2. `TurboJevEngine::evaluate_fused` is available only for backends implementing that explicit SPI. It validates every request, preserves per-request planning/calibration and requires one backend output per request in input order.
3. `turbojev-runtime` defines `FusedRuntimeExecutor`. Implementations must submit independent prepared runtime batches through one underlying runtime operation; looping over `RuntimeExecutor::score` is not a conforming production implementation.
4. `turbojev-runtime-conformance::verify_fused_runtime_executor` compares the fused operation against independent reference scoring and records maximum candidate-logit error.
5. The deterministic fixture backend implements the fused SPI so server -> host -> core -> backend can be tested end-to-end. The fixture is conformance infrastructure, not evidence that a production model runtime supports fused batching.
6. Stock llama.cpp remains `fusedBatch=false`. Its shared-prefix/parallel-suffix ABI requires one common prompt root and must not be relabeled as independent-request batching.
7. Fused planning is bounded by fairness, host item caps, backend item caps and—when both sides provide values—a host memory budget divided by the backend's conservative `estimatedFusedItemBytes`.

## Consequences

A production adapter cannot activate `fusedBatch=true` merely because its host API accepts arrays. It must implement the explicit fused runtime/backend path and pass reference parity. The HTTP scheduler may exercise true fused execution through conformance fixtures today while remaining serial for runtimes that have not proved the capability.
