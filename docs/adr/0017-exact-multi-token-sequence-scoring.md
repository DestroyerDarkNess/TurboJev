# ADR 0017 — Exact multi-token candidate sequence scoring

- Status: Accepted
- Version: v0.15

## Context

Single-token opaque candidate codes are ideal, but a universal runtime cannot
assume every tokenizer exposes enough stable one-token symbols at the exact
assistant answer boundary. Treating a multi-token code as one candidate logit,
summing raw token logits, or normalizing only over candidate tokens would change
the probability model.

## Decision

TurboJev supports multi-token candidate codes only through exact conditional
sequence log-likelihood. Candidate token sequences must be non-empty, unique,
and prefix-free. `turbojev-sequence` provides the portable trie/reference math.
Each edge uses the full-vocabulary log normalizer and each unique candidate
prefix is evaluated once.

When a multi-token codebook is selected, v0.15 downgrades unsafe task-level
parallel and candidate-only plans to serial/full-vocabulary execution while
preserving valid shared prompt-prefix reuse. The runtime reports this downgrade
through `BackendMetrics.effective_plan`, which the core uses for public execution
metadata.

Candidate-only multi-token execution may be re-enabled in a future runtime only
if that runtime can provide the exact full-vocabulary normalizer (or an
equivalent exact `logZ`) at each continuation state.

## Consequences

- Single-token behavior and speed remain unchanged.
- Multi-token probabilities are mathematically meaningful rather than heuristic.
- Candidate-prefix work can be reused without model-family assumptions.
- Some optimizations are conservatively disabled in the fallback path until a
  stronger runtime primitive exists.
- Runtime conformance can compare exact sequence scores against an independent
  reference implementation.
