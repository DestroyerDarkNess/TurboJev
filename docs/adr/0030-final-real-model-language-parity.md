# ADR 0030 — Final real-model language parity is release evidence

## Status

Accepted for v0.28.

## Context

TurboJev already has portable/core conformance, native shim smoke tests and thin host bindings. Those tests cannot prove that one real checkpoint traverses every native language surface with identical semantics. They also cannot prove that llama.cpp multi-root fusion is numerically safe on a concrete model/runtime/backend/hardware target.

## Decision

The pre-1.0 release evidence is one versioned final-validation report produced from a pinned small open-weight GGUF. Node/TypeScript, Python, C and .NET/C# are mandatory rows. Every runner emits the complete canonical response; a common harness validates typed answers and compares complete responses with explicit numerical tolerance. Real fused qualification is a separate required evidence stream and is bound to an exact runtime fingerprint before it can authorize fused execution.

The release workflow pins the required toolchains and runtime/model artifacts. Offline fixtures, a llama.cpp stub, structural checks or package builds can validate implementation, but cannot be substituted for the final real-model report.

## Consequences

TurboJev can close implementation independently from release evidence while remaining honest about what has actually been exercised. Binding drift becomes observable as semantic-response differences rather than merely process failures. A model/runtime/backend/hardware change invalidates fused qualification evidence and requires re-qualification.
