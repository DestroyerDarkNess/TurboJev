# ADR 0007 — Executor-neutral async

Status: accepted in v0.5.0.

TurboJev exposes separate synchronous and asynchronous backend/runtime traits.
Portable crates do not depend on Tokio or any other scheduler. Async futures do
not require `Send`, because browser and mobile host handles can be thread-affine.
Outer server adapters may impose stronger threading requirements.
