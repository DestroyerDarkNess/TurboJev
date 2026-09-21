# ADR 0025 — Server deployment lifecycle and native runtime batch bridge

## Status

Accepted for TurboJev v0.23.

## Context

v0.22 introduced a portable cross-request execution SPI and fair server scheduler. The server adapter still hardcoded the stock `NativeSession` capability set, there was no readiness/drain distinction, and remote deployment guidance stopped at "loopback only". A production-facing local service needs a truthful runtime capability source, deterministic shutdown semantics, request correlation, and an explicit reverse-proxy boundary without contaminating portable inference crates.

## Decision

1. Cross-request capability truth belongs to `turbojev-host-native`, alongside the loaded concrete runtime session. `NativeSession::execution_capabilities()` is mapped into the portable server execution contract.
2. `NativeSession::evaluate_fused()` is the future fused entry point. A runtime that reports `fusedBatch=false` must reject this call. Serial looping is not accepted as evidence of fused batching.
3. Stock fixture and llama.cpp sessions remain serial across independent requests in v0.23. llama.cpp parallel suffix execution inside one semantic request is not reclassified as cross-request fusion.
4. Server lifecycle is `ready -> draining -> stopped`. `/healthz` is liveness; `/readyz` is admission readiness.
5. `ShutdownToken` stops new accepts, flips readiness before drain, preserves already-admitted work for `drainTimeoutMs`, and returns a timeout error if the grace period expires.
6. Every HTTP response receives a bounded/sanitized request ID header. Request IDs never enter prompt/model semantics.
7. Public deployment remains outside the built-in HTTP host. Reverse-proxy mode still binds loopback, may require `X-Forwarded-Proto: https`, and delegates TLS/auth/rate limiting to a colocated proxy.
8. Load testing remains an external host tool with versioned JSON input/output contracts and no dependency from inference crates.

## Consequences

- Future native runtimes can add fused execution without modifying HTTP protocol semantics.
- The server cannot accidentally advertise batching merely because a runtime performs an unrelated internal batch optimization.
- Orchestrators can distinguish live from ready state and drain safely through the embedding API.
- The default server remains safe for local use; remote exposure requires an explicit proxy architecture.
- No OS signal API, TLS library, authentication framework, async runtime, or benchmark dependency enters the portable core.
