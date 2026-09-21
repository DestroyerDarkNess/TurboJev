# ADR 0028 — Evidence-bound runtime qualification

## Decision

Production runtime capabilities are authorized only by a versioned qualification manifest bound to an exact runtime/model/backend/hardware fingerprint. A boolean opt-in alone is insufficient for llama.cpp multi-root execution.

## Consequences

- v0.25's multi-root implementation remains available, but v0.26 requires conformance evidence before host activation.
- Evidence is invalidated by any fingerprint change.
- The portable qualification crate performs no filesystem, hardware, process or OS probing.
- CPU-only upstream abort callback behavior is not generalized into accelerator-independent cancellation support.
