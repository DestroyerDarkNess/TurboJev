# Bounded parallel suffix execution

TurboJev v0.8 closes the shared-state execution phase with a third execution
mode: bounded parallel suffix branches. This document describes the portable
contract; concrete runtimes may use any mechanism that preserves it.

## Planner inputs

The planner receives only model/runtime-neutral data:

- workload task count;
- semantic backend capabilities;
- optional backend `max_parallel_tasks`;
- optional backend `estimated_parallel_task_bytes`;
- optional caller `ExecutionBudget.max_parallel_tasks`;
- optional caller `ExecutionBudget.memory_budget_bytes`.

The selected limit is the minimum safe positive bound. A limit below two turns
`Auto` into serial execution while preserving shared-prefix reuse when available.

## llama.cpp implementation

The v0.8 adapter uses llama.cpp sequence memory:

```text
sequence 0: immutable shared prefix
sequence 1: ephemeral task branch
sequence 2: ephemeral task branch
...
```

For each bounded group TurboJev copies sequence 0 into the required branch IDs,
submits all non-empty suffix tokens in one decode batch, gathers one output row
per non-empty suffix, serves empty suffixes from retained prefix logits, and then
deletes every ephemeral sequence.

The adapter preserves request order even though native output rows exist only for
suffixes that actually emitted a scored position.

## Correctness gate

Three paths must remain comparable:

- independent reference;
- shared-prefix cached serial;
- shared-prefix bounded parallel.

The runtime conformance layer compares raw candidate logits before TurboJev
softmax/calibration. Candidate-only projection is not part of v0.8 and cannot be
advertised by the llama.cpp adapter yet.


## v0.9 projection interaction

Scheduling and projection are orthogonal dimensions of `ExecutionPlan`. A
parity-verified runtime may combine `Shared + Parallel + CandidatesOnly`. The
llama.cpp fixture suite covers this combination, while the stock FFI path keeps
full-vocabulary projection until a genuine selective native path exists.
