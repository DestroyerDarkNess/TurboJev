# ADR 0005 — Wire compatibility is an adapter

## Status
Accepted in v0.3.

## Decision
External Jev/TypeSafe/AI SDK request and response shapes are translated by `turbojev-compat`. The inference core does not contain provider names or transport rules.

## Why
The TypeSafe native API calls the boolean primitive `noul`, while AI SDK exposes the same semantic primitive as `boolean` and returns `probability`. Those are wire differences, not model differences. Keeping them outside the core preserves model/runtime/platform independence.

## Consequences
- adding a future OpenRouter Decisions profile cannot modify scoring;
- bindings can select a profile without duplicating schema logic;
- compatibility can evolve independently from the canonical TurboJev contract.
