# ADR 0023 — Local server protocol boundary

## Status
Accepted in v0.21.

## Decision
Split Phase 11 server work into `turbojev-server-protocol` and `turbojev-server`.

`turbojev-server-protocol` is portable and owns only route/profile/batch/error semantics. `turbojev-server` owns sockets, file-backed configuration, HTTP parsing, host threads, bounded admission and the native session worker.

A single dedicated worker owns `NativeSession`. Concurrent clients do not directly share or clone the model session. Batch responses preserve input order and v0.21 labels cross-request execution as session-thread serial rather than claiming fused runtime batching.

The initial server binds loopback only. Gateway aliases do not imply unsupported third-party wire compatibility.

## Consequences

- portable crates remain free of `std::net`, filesystem and environment APIs;
- thread-affine runtimes remain valid server backends;
- backpressure is explicit and bounded;
- future provider adapters can evolve without contaminating core inference semantics.
