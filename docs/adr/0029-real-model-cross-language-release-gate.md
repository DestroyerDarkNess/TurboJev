# ADR 0029 — Real-model cross-language release gate

## Decision

The final pre-1.0 validation uses one small open-weight GGUF artifact across Node/TypeScript, Python, C and .NET/C#. Every host executes the same canonical request through the same native Rust session/runtime stack.

## Consequences

- Fixture-only wrapper tests remain useful but cannot close the release gate.
- The final report records exact model and request SHA-256 values.
- Language wrappers may differ in ergonomics but not semantic decision logic.
- Swift remains build-gated unless the release environment provides an Apple runtime host.
