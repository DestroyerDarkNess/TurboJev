# Option order: choices, scores and alternative Jev implementations

Source audit on 2026-10-07. This compares implementation paths, not measured model
accuracy or latency. Remote sources are pinned below because the local copies in
`other/` can be older. No alternative's benchmark was reproduced during this audit.

## The distinction to preserve

For `choice`, option identities and descriptions matter; listing order is a
nuisance variable. A distribution should be **equivariant**: when options move,
their probabilities move with them. The selected semantic answer should stay the
same, except where an explicit tie policy says otherwise.

For `score`, the caller defines an ordinal scale. Reversing `calm, annoyed,
furious` changes which meaning has index 0. Keep the complete scale and its order
in the model input, and compute the expected level from its distribution.

It does not follow that one shared model or backbone is impossible. A
task-conditioned readout can see the whole option set and retain ordinal features
for score tasks. The conflict arises when the same operation removes ordering
information that a score task needs. These projects mostly address it with
different prompt and aggregation policies, rather than a universal newly trained
classification head.

## What the source actually implements

| Project / pinned source | Choice path | Score path | Qualification |
| --- | --- | --- | --- |
| [TypeLLM](https://github.com/TypeLLM/TypeLLM/blob/c76206c753c6558b7e08696098e428cec9569ab4/typellm/runtime.py#L930) | Whole-set label logits; canonical balanced, sampled or exhaustive layouts; align by identity and average probabilities. | [Ordered `levels`](https://github.com/TypeLLM/TypeLLM/blob/c76206c753c6558b7e08696098e428cec9569ab4/typellm/schema.py#L665); rejects `permutations`, computes expected index. | No trained replacement head required. Balanced layouts address positions and immediate neighbours, not every possible interaction. |
| [pijev](https://github.com/TypeLLM/pijev/blob/c85474a500c8d52404e344ae052731b93e506f56/pijev/__init__.py#L42) | Canonical layouts, label-aligned probability mean. | Score and Noul pass through unchanged. | Wraps the official hosted Jev SDK; it is an algorithm reference, not a local GGUF runtime. |
| [AnyJev, current upstream](https://github.com/nokia-applied-research/AnyJev/blob/d458d13968eec0e2cfa8c81b64629b96cf5cb4a9/anyjev/decider.py#L436) | L0 cyclic log-space aggregation; optional canonical order and label-prior correction. | `ordered=True` bypasses permutations. | The [level contract](https://github.com/nokia-applied-research/AnyJev/blob/d458d13968eec0e2cfa8c81b64629b96cf5cb4a9/docs/levels.md) says 0.3.0 removed L1/L2. Tacit uses distilled checkpoints and optional bounded reasoning. Neither guarantees correctness. |
| [AnyJev branch retaining L1/L2](https://github.com/achristofaro/anyjev/blob/3cd8c6fcd9e90fc04214575ade6779da1e3f3704/docs/method_v3.md#L62) | Per-question hidden-state heads, fitted on labels; shuffled listings for some small option sets. | A separate question/layout-bound artifact, not a universal ordinal head. | Approximate learned robustness; needs hidden-state access, model-specific data and held-out calibration. Do not confuse output remapping with exact invariance. |
| [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev/blob/a4aafa85e1aba95329db36e085373d1572ab6917/src/llm2jev/inference/prompt.py#L46) | Yes/no judgment per candidate **with all candidates visible**. | Each judgment sees the full scale explicitly labelled lower-to-higher. | Fixes omission of the competing candidates. The ordered prompt can still influence the model; independent binary scores are not proof of invariance. |
| [tic-top/llm2jev](https://github.com/tic-top/llm2jev/blob/2b252d504972764211ef172c1155ac0fedc9c3de/llm2jev/prompt.py#L86) | Whole-set label-token scoring with a checked answer boundary. | Keeps numbered level meanings and order. | Separate repository from Yinsongxu's project. Reading one token does not itself remove option bias. |

## TurboJev 0.30.1 source findings

- Built-in renderers already expose the whole candidate set. TurboJev is not
  limited to scoring each option without seeing its competitors.
- `evaluate` keeps score levels in caller order, assigns original numeric keys,
  and returns `sum(index * probability)`. Cyclic correction applies only to
  unordered `choice` questions.
- `evaluate` cyclic mode averages **raw logits**, then calibrates. With the
  ordinary softmax calibrator this corresponds to a normalized geometric mean
  of aligned probabilities. It is a different policy from TypeLLM's arithmetic
  probability mean, not automatically an implementation error.
- Cycles of the caller's list do not enumerate arbitrary permutations. They
  retain cyclic neighbour relationships. Cancellation is exact for a purely
  additive position bias; it is not a general guarantee for set interactions.
- `generate` already has opt-in `permutations: "auto"`, `"all"`, or an integer,
  and averages probabilities. Auto builds canonical balanced layouts. Single
  pass remains the default. A numeric enum is still an enum, **not** an implicit
  ordered rubric; use `evaluate`'s `score` for ordinal ratings.
- Generation's integer sampling includes the caller's original layout. A fixed
  seed therefore does not promise the same sampled semantic layouts after an
  arbitrary input reorder. Do not advertise every permutation budget as invariant.
- The training-data materializer previously shuffled score-level listings like
  choices. It retained numeric keys and aligned targets, but this disagreed with
  inference's ordered-scale policy. The patch now keeps one score listing while
  independently permuting output codes. Choice and boolean augmentation are
  unchanged, and every source decision retains its total training weight.

The training patch cannot change existing published binaries or model weights.
Regenerate affected score training datasets before further fine-tuning. It is
not evidence that a previously observed model-accuracy issue is now solved.

## Implemented in 0.31.0

`evaluate` now accepts opt-in `balanced` in native and browser hosts. It reuses
the canonical Williams design, calibrates each layout once, averages keyed
probabilities and leaves scores/nouls unchanged. Exact choice ties use semantic
keys; finite generated enum ties use JSON value encodings. Higher probabilities
always win, without an epsilon tie window. Tests cover all 24 input orders of a
four-way set-interaction fixture, sync/async parity and non-unit calibration.
This is structural/numerical evidence, not a real-model accuracy guarantee.

## Further implementation work

1. Keep task semantics explicit through training, prompt rendering and aggregation:
   unordered choice, boolean, ordered score. Expose an ordered-scale indicator to
   renderers rather than inferring the type from candidate keys. Keep all
   competing options, or all rubric levels, visible.
2. Measure the new opt-in balanced policy with real models and production-like
   option sets. Preserve explicit mode selection and account for every prefill;
   batching one request is not equivalent to performing one model evaluation.
4. Consider per-model/per-question label-prior correction only after evaluation.
   Do not make batch-prior subtraction unconditional: skewed real traffic can
   make that correction harmful. Temperature calibration cannot change argmax.
5. If adding learned heads, use task-conditioned features or separate choice
   and ordinal readouts. Include all options in the context, retain ordinal
   indices for scores, and bind artifacts to the model, rubric and calibration.
   Hidden-state exposure and training would be additional work, not a portable
   guarantee supplied by today's GGUF/WASM adapter.

## Validation and next benchmark

Two focused regressions accompany the audit: auto enum layouts must be identical
by semantic value after reversed and rotated input listings; score training must
retain the scale, align its target through code augmentation and conserve weight.
The score test failed before the patch (16 records instead of 4).

After the patch, 29 tests passed across `turbojev-core`,
`turbojev-training-data` and `turbojev-specialized-artifact`, using the locked
workspace dependencies on Windows x64. No real-model run or cross-platform
runtime qualification was performed for this training-only correction.

For a model benchmark, keep the GGUF hash, quantization, device, template and
tokenizer fixed. Compare single pass, cyclic log mean, canonical balanced mean
and exhaustive mean where affordable. Include tasks whose answer depends on the
option set, arbitrary reorderings beyond rotations, ties, and ordered rubrics.
Measure keyed probability drift, answer flips, accuracy, Brier/NLL, ordinal
MAE and latency. A reversed score scale is a changed rubric, not a choice-order
invariance test. Head training and adaptive reasoning need separate experiments.

Full deterministic permutation averaging has a convex-loss guarantee relative
to the mean loss of the included predictions, not to the best listing or to a
ground-truth accuracy guarantee. The [pijev explanation](https://github.com/TypeLLM/pijev/blob/c85474a500c8d52404e344ae052731b93e506f56/README.md#guaranteed-better-calibration)
states this distinction. Do not extrapolate another project's benchmark to
TurboJev or treat a synthetic regression as a real-model accuracy measurement.
