# ADR 0009 — Parallel suffixes are bounded, isolated branches

Status: accepted in v0.8.

## Context

A shared semantic state can be prefetched once and reused by many independent
questions. Executing all suffixes serially leaves GPU/accelerator parallelism
unused; executing an unbounded number can exceed sequence or memory capacity.
TurboJev must stay portable and therefore cannot query CUDA, Metal, WebGPU or an
operating-system memory API from its core.

## Decision

Parallel scheduling is expressed only through portable data:

- `BackendCapabilities::parallel_suffixes` states semantic support;
- `BackendResourceHints::max_parallel_tasks` states a runtime hard bound;
- `BackendResourceHints::estimated_parallel_task_bytes` optionally supplies a conservative branch estimate;
- `ExecutionBudget` supplies caller caps for task count and/or branch memory;
- `ExecutionPlan::parallel_task_limit` is the resulting bound.

The llama.cpp adapter keeps sequence 0 immutable. Each parallel task gets a
separate ephemeral sequence copied from sequence 0. All branch suffix tokens in
one bounded group are submitted together; branches are removed afterwards.
Empty suffixes read retained prefix logits and do not require synthetic tokens.

Reference and cached-serial execution remain correctness baselines. An adapter
may advertise parallel suffixes only while its optimized output remains within
the explicit conformance tolerance.

## Consequences

TurboJev can exploit parallelism on desktop, mobile and web runtimes without
putting a device/OS concept in portable core code. Hosts with little memory can
lower the budget without changing semantic APIs. Candidate-only projection is a
separate optimization and is not implied by parallel scheduling.
