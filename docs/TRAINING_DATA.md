# TurboJev specialized training data

TurboJev v0.18 separates **semantic source data** from **deterministically materialized semantic training records**. v0.19 adds the separate model-bound/tokenized layer.
The source corpus contains typed requests plus ground-truth answers; it does not contain token ids,
chat templates, generated JSON, or assumptions about Qwen/Llama/Gemma.

## Contracts

- `turbojev.semantic-training-dataset.v1`: source examples, languages, typed targets and data-quality provenance.
- `turbojev.materialized-training-dataset.v1`: deterministic candidate-logit records after candidate-order and candidate-code assignment debiasing.

A semantic example can contain several questions over one shared state. TurboJev reuses the same
`EvaluationRequest -> DecisionBatch` compiler used by inference, then materializes one training decision
per question. This prevents training-only candidate semantics from drifting away from runtime semantics.

## Target representation

Targets remain typed until materialization:

```text
choice -> stable criterion key
noul   -> false / true
score  -> zero-based level index
```

The materialized record stores the semantic `PromptDocument`, `target_key`, `target_index`, and
`target_code`. A host trainer computes candidate logits and applies cross entropy directly to
`target_index`; it never trains the model to generate a JSON response.

## Deterministic debiasing

For every source decision the materializer independently expands:

```text
semantic candidate order
        x
candidate-code assignment
```

Both use the recipe seed and stable scope identifiers. The identity permutation is always retained.
Requested permutation counts are capped by the actual permutation space (`n!`), so two-candidate
boolean decisions produce exactly two distinct order variants rather than duplicate rows.

Augmentation does not silently multiply sample importance. If a decision with source weight `w`
produces `k` records, each receives `w/k`; the sum remains `w`.

## Candidate-code safety boundary

`turbojev-training-data` does **not** invent token ids. A `CandidateCodeProvider` must supply non-empty,
unique code strings that the host trainer has verified against the selected checkpoint's renderer and
tokenizer at the exact answer boundary. The portable materializer only permutes those verified codes.

This preserves the existing TurboJev invariant:

```text
no hard-coded A/B/C token ids by model family
```

and keeps tokenizer/model runtime dependencies outside the portable crate.

## Recipe policy checks

Before materialization the dataset must satisfy the recipe:

- dataset id equals the recipe training split;
- declared language set is covered exactly;
- typed targets match every question;
- required contrastive pairs exist;
- required negative/no-match, adversarial-criteria and long-shared-state examples exist;
- train/validation/calibration/benchmark remain separate through the v0.17 recipe contract.

## Next boundary

`turbojev-specialized-artifact` (v0.19) now consumes `CandidateLogitTrainingRecord`, renders it with the pinned renderer profile, validates/encodes codes with the real tokenizer/chat-template boundary, and emits a framework-neutral `ModelBoundTrainingDataset`. Actual LoRA/QLoRA optimization remains external; filesystem, model-download, CUDA/Metal and trainer-framework concerns do not enter this crate or TurboJev core.
