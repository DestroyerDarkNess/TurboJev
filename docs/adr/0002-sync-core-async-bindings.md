# ADR 0002: Synchronous scoring core, asynchronous platform bindings

Status: Accepted for v0.1

## Context

Rust async traits complicate object safety, FFI and WASM because native futures are commonly `Send` while browser futures often are not. Inference after model loading is fundamentally a bounded operation that bindings can schedule on workers/threads.

## Decision

The v0.1 backend SPI is synchronous. Model acquisition and platform lifecycle are separate concerns. Node/browser/mobile bindings may expose Promise/async APIs and execute the synchronous scoring operation in the appropriate worker/runtime.

## Consequences

- simple FFI contract
- no executor dependency in core
- deterministic unit testing
- future genuinely asynchronous backends can be introduced through an execution adapter without changing schema types
