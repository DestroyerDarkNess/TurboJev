# ADR 0011: Candidate-only projection must be truthful and parity-gated

Status: accepted in v0.9.

## Context

TurboJev can obtain candidate logits by computing a normal full-vocabulary LM
head and gathering a few token positions. Nimble demonstrates that runtimes can
save work by projecting only candidate output rows. The two methods are
numerically intended to represent the same logits but are not computationally
equivalent.

Calling full-vocabulary + gather "candidate-only" would make capability
negotiation, benchmarks and performance metadata misleading.

## Decision

`selective_output_head = true` means the runtime has an execution path that does
not compute the complete vocabulary projection for that decision.

Every first-party selective implementation must be checked against the frozen
full-vocabulary reference path. Selective outputs also report the number of
candidate rows actually projected. Conformance rejects missing or inconsistent
row accounting.

The portable `turbojev-projection` crate defines reference dense math, while
`turbojev-runtime-projection` demonstrates a complete runtime-neutral selective
executor. Concrete runtimes may use more efficient device-native kernels while
preserving the same observable contract.

## Consequences

- Performance claims remain meaningful.
- Stock llama.cpp support stays conservative until a genuine stable selective
  path exists.
- The planner can safely choose `CandidatesOnly` from capabilities alone.
- Full-vocabulary execution remains available as a correctness oracle.
- Multi-token sequence scoring is not conflated with output-row projection.
