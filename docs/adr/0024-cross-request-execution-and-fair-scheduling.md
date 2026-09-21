# ADR 0024 — Cross-request execution capability and fair scheduling

Status: Accepted in v0.22.

## Context

v0.21 introduced bounded HTTP admission but submitted a complete batch as one worker command. That preserved correctness, yet a large batch could monopolize the single thread-affine `NativeSession`. It also left no portable contract by which a future backend could truthfully advertise fused execution across independent server requests.

## Decision

Introduce portable `turbojev-server-execution` and keep scheduling mechanics in host-only `turbojev-server`.

The portable crate owns:

- `CrossRequestCapabilities`;
- `CrossRequestMode`;
- `BatchPlanningPolicy`;
- bounded per-turn planning;
- `CrossRequestExecutor` SPI;
- cancellation/deadline observation interface;
- exact serial-vs-fused response parity conformance helper.

The host crate owns:

- monotonic clocks and request deadlines;
- atomics used by soft cancellation;
- the bounded MPSC admission queue;
- round-robin job scheduling;
- operational counters;
- HTTP timeout header parsing.

The stock `NativeSession` executor advertises `fusedBatch=false`, `cooperativeCancellation=false`, and `hardDeadlines=false`. Therefore v0.22 remains serial across independent requests while improving fairness. An adapter may advertise fused batching only after exact parity against independent reference execution.

## Consequences

Large client batches no longer monopolize the session worker. Timed-out jobs stop consuming work for items that have not started. Already-running native inference is not falsely described as cancellable. Server-response metadata can distinguish serial round-robin execution from a future fused batch path.
