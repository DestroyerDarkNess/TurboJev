# TurboJev model-bound specialized artifacts

TurboJev v0.19 adds the boundary between portable specialization data and the exact renderer/tokenizer of an open-weight checkpoint. It does **not** add a trainer.

## Why this layer exists

v0.18 intentionally stopped before token ids. A semantic/materialized record can say that the allowed answer codes are `A`, `B`, `C`, and `D`, but those strings are not universally one token and are not universally stable at a model's real assistant boundary.

v0.19 binds records through the same model-facing abstractions already used by inference:

```text
MaterializedTrainingDataset
        │
        ▼
PromptRenderer (checkpoint profile)
        │
        ▼
PromptCodec (real chat template + tokenizer)
        │
        ├── encode_prompt
        ├── encode_prompt_with_answer
        └── decode round-trip
        │
        ▼
ModelBoundTrainingDataset
```

The exported record contains exact rendered messages, prompt token ids, candidate token ids, target index and target token id. It is framework-neutral and can be consumed by an external LoRA/QLoRA trainer, a custom Rust trainer, MLX, Python tooling, or another implementation.

## Candidate-logit CE remains exact

The v1 specialized objective is:

```text
candidate_logits = logits[candidate_token_ids]
loss = cross_entropy(candidate_logits, target_index)
```

That objective requires one token per candidate at the answer boundary. `BoundaryVerifiedCodeProvider` searches deterministic candidate symbol sets and accepts a set only when every code:

1. extends the exact encoded prompt boundary;
2. round-trips through the real tokenizer;
3. maps to a unique token sequence;
4. is prefix-safe;
5. is exactly one token for candidate-logit CE v1.

TurboJev's multi-token **inference** fallback remains valid and unchanged. v0.19 deliberately does not pretend multi-token sequence training is equivalent to one-step candidate-logit CE.

## Portable export contract

`turbojev.model-bound-training-dataset.v1` records:

- exact base model + revision + tokenizer revision;
- prompt IR and renderer profile;
- tokenizer runtime profile;
- candidate-code profile;
- exact rendered messages;
- exact prompt token ids;
- semantic candidate key + rendered code + token id;
- target key/index/code/token id;
- augmentation provenance and conserved sample weight;
- maximum prompt/context token usage.

The crate has no filesystem, network, GPU, optimizer, model download, Python, CUDA, Metal, browser or OS dependency.

## External training boundary

TurboJev produces the semantic and model-bound artifact. Optimization stays outside TurboJev:

```text
TurboJev recipe + dataset
          │
          ▼
model-bound export
          │
          ▼
external trainer chosen by the user
          │
          ├── LoRA
          └── QLoRA
          │
          ▼
open-weight adapter / merged checkpoint
          │
          ▼
TurboJev specialized manifest + compatibility gate
```

No external trainer is a TurboJev runtime dependency.

## Calibration handoff

A fitted calibration profile can be attached only when its provenance matches the specialized checkpoint:

- holdout id equals the recipe calibration split;
- model id equals the produced specialized model id;
- tokenizer revision equals the recipe/base tokenizer revision;
- prompt profile equals the pinned prompt IR;
- renderer profile equals the recipe renderer profile.

Missing provenance is rejected rather than inferred.

## Runtime compatibility

`validate_runtime_compatibility` checks the loaded specialized runtime before inference begins. Prompt IR, renderer, tokenizer revision, context capacity, calibration profile, and pinned candidate-code profile must match the manifest. Generic models remain unaffected because they do not require a `DecisionModelContract`.
