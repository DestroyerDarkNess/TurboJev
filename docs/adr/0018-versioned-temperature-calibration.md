# ADR 0018 — Versioned temperature calibration profiles

- Status: Accepted
- Version: v0.16

## Context

Candidate softmax probabilities are conditional distributions, not automatically calibrated confidence. Mixing calibration into model execution would make runtime comparisons ambiguous and would encourage reusing a temperature after changing the model, tokenizer or prompt contract.

## Decision

TurboJev fits one positive scalar temperature on an independent held-out set by minimizing weighted multiclass NLL. The calibration dataset and resulting profile have explicit versioned contracts. Profiles record provenance and before/after NLL, Brier, ECE and accuracy. Portable fitting/math lives in `turbojev-calibration`; filesystem/CLI behavior lives in the non-portable `turbojev-calibrate` tool.

Temperature scaling never changes candidate ordering: all logits are divided by the same positive scalar. Plain `SoftmaxCalibrator` therefore remains marked `fitted = false`, while a profile-loaded `TemperatureCalibrator` is marked fitted.

## Consequences

- Calibration can be audited and reproduced independently of inference.
- Runtime adapters remain unaware of fitting datasets or files.
- A profile can be pinned by `DecisionModelContract.calibration_profile` without adding OS dependencies to the core.
- Calibration datasets must not be silently reused after model/tokenizer/prompt/renderer changes.
