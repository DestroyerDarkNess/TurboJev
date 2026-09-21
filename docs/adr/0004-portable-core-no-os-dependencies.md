# ADR 0004 — Portable core has no OS dependencies

## Status

Accepted in v0.2.0.

## Context

TurboJev targets desktop, server, mobile, embedded-capable hosts, browser/WASM,
and multiple host languages. Coupling the semantic engine to filesystem,
network, GPU, browser, or OS APIs would make portability depend on conditional
compilation and duplicate behavior across targets.

## Decision

All portable crates are forbidden from directly depending on an operating
system, browser API, hardware runtime, model loader, or network stack.

Concrete model execution is provided by optional runtime adapters implementing
the same `ScoringBackend` contract. Model locations are opaque `ModelSource`
values resolved outside the portable core.

## Consequences

- The core can be reused on every target supported by Rust and the chosen host adapter.
- New model families/formats do not require modifications to decision semantics.
- Platform-specific optimization remains possible, but only behind adapter boundaries.
- Loading/downloading models is deliberately not the responsibility of `turbojev-core`.
