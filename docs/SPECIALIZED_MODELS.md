# TurboJev specialized models

TurboJev Phase 10 keeps the public decision API unchanged while allowing a checkpoint to be specialized for direct candidate-logit decisions.

## Separation of concerns

The portable crate `turbojev-specialized` contains only versioned contracts and validation rules. It does not train a model, download weights, open files, select CUDA/Metal/WebGPU, or choose a training framework. A host-side trainer may use PEFT/Transformers, MLX, a Rust-native trainer, or another stack as long as it materializes the same recipe and output manifest.

```text
semantic training examples
        │
        ├── candidate-order permutations
        ├── candidate-code permutations
        ├── multilingual variants
        ├── contrastive pairs
        ├── negative/no-match cases
        └── adversarial criteria
        │
        ▼
Prompt IR + renderer profile + tokenizer revision
        │
        ▼
base model + LoRA/QLoRA
        │
        ▼
raw candidate logits
        │
        ▼
CE(target candidate)
```

The objective is `candidate-logit-cross-entropy`. Specialized training does not teach the model to emit JSON or prose answers.

## Independent splits

Every recipe names four pairwise-distinct split identities:

- `train`
- `validation`
- `calibration`
- `benchmark`

Recipe validation rejects repeated split ids. Calibration remains a post-training operation on its own holdout and the generic-vs-specialized benchmark remains independent from both training and calibration.

## Runtime contract

A successful training run publishes a `turbojev.specialized-model.v1` manifest. Its embedded `DecisionModelContract` pins:

- `turbojev.prompt-ir.v2` or the exact future prompt IR used for training;
- renderer profile;
- candidate encoding policy;
- tokenizer revision;
- maximum context;
- optional calibration profile.

`ModelSpec` can now carry this decision contract directly. Generic models leave `decision_contract` unset, so the generic API and runtime path remain unchanged.

## Phase 10 experiment matrix

v0.17 includes framework-neutral recipe fixtures for four experiment classes:

| class | method | purpose |
| --- | --- | --- |
| 0.6B | LoRA | smallest dedicated decision model baseline |
| 0.8B | LoRA | small-model quality/latency comparison |
| 2B | QLoRA | mid-size specialization baseline |
| 4B | QLoRA | upper Phase-10 quality baseline |

The fixture model ids are intentionally placeholders rather than an allowlist. TurboJev remains model-family agnostic.

## Same-benchmark evaluation

`turbojev.specialization-evaluation.v1` requires generic and specialized models to be compared on one named benchmark suite with an explicit metric set:

- accuracy
- NLL
- multiclass Brier
- ECE
- latency
- peak memory
- decisions/sec
- candidate-order robustness
- paraphrase robustness

No quality claim is implied by the contract. Actual checkpoints must be trained and measured before a model comparison can be reported.

## Training framework boundary (v0.19)

TurboJev defines recipes, datasets, model-bound exports, manifests, calibration handoff and evaluation contracts; it does **not** embed a LoRA/QLoRA trainer. `turbojev-specialized-artifact` binds records to the selected open-weight checkpoint's real renderer and tokenizer so an external trainer can optimize them without duplicating TurboJev semantics.
