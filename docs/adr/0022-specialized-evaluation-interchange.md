# ADR 0022 — Framework-neutral specialization evaluation interchange

Status: Accepted in v0.20.

## Context

Phase 10 already defines recipes, semantic/materialized datasets, exact model-bound candidate records, checkpoint manifests and calibration handoff. The remaining evaluation problem is different: generic and specialized checkpoints may be executed by different host programs, but TurboJev still needs one auditable semantic comparison format.

Putting model loading or framework-specific evaluation code into the portable core would violate the runtime/model/OS-agnostic architecture and would couple TurboJev to external training stacks.

## Decision

Introduce portable `turbojev-specialized-eval` and versioned external-run/report/bundle contracts.

External harnesses emit raw candidate logits plus stable semantic candidate keys and measurement provenance. TurboJev validates same-benchmark/same-environment comparability and computes baseline quality, controlled robustness, latency/throughput/memory summaries and specialized-minus-generic deltas.

Headline accuracy/NLL/Brier/ECE use baseline cases only. Candidate-order and paraphrase observations are measured separately as semantic invariance and accuracy.

A separate host CLI, `turbojev-specialized-report`, owns filesystem/argument parsing. It is not a portable dependency.

## Consequences

- No evaluation framework becomes a TurboJev dependency.
- Candidate permutations remain semantic because keys travel with logits.
- Performance comparisons fail fast when backend/host/quantization/execution path differ.
- Calibration math is reused from the existing calibration crate instead of reimplemented.
- Reports encode measurements and deltas, not a synthetic overall score.
- Experiment bundles can pin all evidence by SHA-256 without adding network/storage behavior to portable crates.
