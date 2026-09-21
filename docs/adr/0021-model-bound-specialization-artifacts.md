# ADR 0021 — Bind specialized training artifacts to the real checkpoint boundary

## Status

Accepted in v0.19.

## Context

TurboJev v0.18 can materialize deterministic semantic candidate-logit records without hard-coding token ids. Specialized training still needs the exact prompt/chat template and tokenizer behavior of the selected open-weight checkpoint. Pulling a trainer framework into the runtime would violate TurboJev's portability and dependency direction.

## Decision

Add portable `turbojev-specialized-artifact` between semantic materialization and external training.

The crate reuses `PromptRenderer`, `PromptCodec`, candidate symbol discovery, and exact answer-boundary conformance checks. It exports a versioned model-bound dataset containing rendered messages and exact token ids. Candidate-logit CE v1 requires every candidate code to be one token. Multi-token inference remains separate and unchanged.

TurboJev may emit/validate specialized manifests and calibration provenance, but optimization/backpropagation remains outside the project core and no training framework becomes a dependency.

## Consequences

- Generic inference remains unchanged.
- Specialized data cannot silently drift from the runtime renderer/tokenizer contract.
- External trainers receive deterministic, framework-neutral records.
- A checkpoint with a mismatched tokenizer/prompt/renderer/calibration profile fails before inference.
- Supporting a future multi-token training objective requires a new explicit objective/contract rather than reinterpreting candidate-logit CE.
