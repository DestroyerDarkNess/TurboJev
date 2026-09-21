# ADR 0019 — Specialized training is a portable contract, not a core framework dependency

## Status

Accepted for v0.17.

## Context

TurboJev needs decision-specialized checkpoints while remaining universal across model families, operating systems, runtimes and host languages. Pulling a particular training framework into the semantic core would violate that boundary and would make checkpoint semantics depend on incidental tooling.

## Decision

1. Add portable `turbojev-specialized` contracts for LoRA/QLoRA recipes, candidate-logit cross-entropy, data/debiasing policy, explicit split identities and generic-vs-specialized evaluation plans.
2. Keep all trainer execution outside portable crates.
3. Require train, validation, calibration and benchmark split ids to be pairwise distinct.
4. Attach an optional `DecisionModelContract` to `ModelSpec`; generic models omit it.
5. A specialized checkpoint manifest must pin base-model revision, tokenizer revision, prompt IR, renderer profile, candidate encoding, context limit and calibration identity.
6. Candidate/order permutation is described as deterministic policy with a stable seed; host trainers materialize it without changing semantic labels.
7. Compare generic and specialized checkpoints on the same benchmark suite. Training success is not inferred from training loss alone.

## Consequences

The same specialized-model artifact can be trained with different host tooling while preserving a common runtime contract. TurboJev can reject manifest drift and obvious split leakage before inference. Generic mode remains unchanged and no Python/CUDA/MLX dependency enters the portable core.
