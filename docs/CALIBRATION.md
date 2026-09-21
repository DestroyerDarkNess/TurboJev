# Calibration tooling

TurboJev keeps **decision scoring** and **probability calibration** separate. Raw candidate logits (or exact sequence log-likelihoods) determine ordering. Calibration maps those scores to a probability distribution whose confidence should be evaluated on data that was not used to train the model or tune the decision prompt.

## v0.16 workflow

1. Collect a held-out dataset using `turbojev.calibration-dataset.v1`. Each row contains raw candidate logits in schema order plus the reference candidate index. Optional positive weights are supported.
2. Run `turbojev-calibrate fit holdout.json profile.json --id my-profile`.
3. The CLI minimizes weighted multiclass negative log likelihood with a scalar temperature `T > 0` using deterministic golden-section search in `log(T)` space.
4. The produced `turbojev.calibration-profile.v1` pins the temperature, holdout identity, model/tokenizer/prompt/renderer provenance when provided, and before/after NLL, multiclass Brier score, top-label ECE and accuracy.
5. Load the manifest and construct `TemperatureCalibrator::from_profile`. A profile is marked fitted; plain softmax remains explicitly unfitted.

The fitting set must remain independent of training data and benchmark/test sets. A lower ECE by itself is not proof of a better model: report NLL, Brier, ECE, accuracy, sample count, bin count and provenance together.

## Dataset

```json
{
  "contract": "turbojev.calibration-dataset.v1",
  "id": "support-routing-holdout-v3",
  "model_id": "qwen3.5-4b-q4",
  "tokenizer_revision": "...",
  "prompt_profile": "turbojev-reference-es-v1",
  "renderer_profile": "qwen-chat-v1",
  "examples": [
    {"logits": [1.2, 3.8, -0.4], "target": 1},
    {"logits": [2.1, 0.2, -1.7], "target": 0, "weight": 2.0}
  ]
}
```

## CLI

```text
turbojev-calibrate report holdout.json --temperature 1.0 --ece-bins 15
turbojev-calibrate fit holdout.json calibrated.json --id qwen-routing-v3
turbojev-calibrate report holdout.json --temperature 1.73
```

The optimizer is intentionally dependency-light and deterministic. It fits only one scalar temperature; it does not alter argmax decisions because dividing all candidate scores by the same positive scalar preserves ordering.

## Metrics

- **NLL**: weighted mean `-log(p_target)`. This is the fitting objective.
- **Brier**: weighted mean multiclass squared error `sum_k (p_k - y_k)^2`.
- **ECE**: weighted top-label expected calibration error using equal-width confidence bins.
- **Accuracy**: weighted top-1 accuracy. Temperature scaling should leave this unchanged except for exact numeric ties.

Calibration profiles are versioned artifacts. A production profile should be rejected when its model/tokenizer/prompt/renderer provenance no longer matches the runtime contract that produced the logits.
