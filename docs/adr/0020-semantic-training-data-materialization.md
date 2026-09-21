# ADR 0020 — Semantic training data and deterministic materialization

## Status
Accepted in v0.18.0.

## Context
Specialized checkpoints need order/code debiasing without teaching a second request schema or hard-coding
a tokenizer/model family. Training and inference must agree exactly on choice, noul and score candidates.

## Decision
1. Source datasets use `EvaluationRequest` plus typed ground-truth targets.
2. `turbojev-core::compile_decision_batch` is public and is the single request-to-candidate normalization path for both inference and training.
3. `turbojev-training-data` is portable and deterministic; it owns validation, data-policy gates, permutations, provenance and weight conservation.
4. Candidate codes enter through `CandidateCodeProvider`; model/tokenizer boundary verification remains a host/runtime responsibility.
5. Materialization emits semantic `PromptDocument` records rather than rendered chat text or token ids.
6. Requested permutation counts are capped at the number of unique permutations instead of emitting duplicate augmented rows.
7. Every source decision conserves its source sample weight across all augmented records.

## Consequences
- Generic inference behavior remains unchanged.
- Training cannot silently redefine boolean or score candidate semantics.
- Dataset augmentation is reproducible from recipe seed + semantic ids.
- Model-specific trainer integrations can vary without changing the dataset contract.
- Exact tokenizer safety remains explicit and cannot be falsely inferred from portable text symbols.
