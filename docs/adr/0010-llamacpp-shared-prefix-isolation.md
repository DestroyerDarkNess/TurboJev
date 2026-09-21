# ADR 0010 — llama.cpp shared-prefix branches are isolated

## Status
Accepted in v0.7.

## Context
TurboJev must reuse expensive shared state without allowing one semantic decision
to influence another. A mutable KV/recurrent cache reused in-place across tasks
would make task order observable and break the independent-decision contract.

## Decision
The llama.cpp adapter reserves sequence 0 as the immutable shared prefix. For
each serial task it copies sequence 0 into ephemeral sequence 1, evaluates only
that task's suffix, gathers candidate logits, and removes sequence 1 before the
next task. Final-prefix logits are retained separately for the valid empty-suffix
case.

## Consequences
- Task order cannot mutate the base prefix cache.
- Shared-prefix execution remains serial in v0.7.
- Memory use is bounded to the base plus one active branch.
- Parallel branches require a later milestone with explicit memory budgeting and
  conformance tests.
- The independent v0.6 path remains available as the correctness reference.
