# Designing decisions for open-weight models

TurboJev scores application-supplied candidates. Quality depends on the selected
model, runtime, template, evidence, and question. A typed response does not prove
that a model reasoned correctly or can plan a route.

## Describe the actual decision

Provide relevant facts, define coordinate conventions or domain terms, and ask
for a bounded action. Keep prompts concise without removing necessary evidence.
Meaningful candidate keys describe the action the application will execute.

For a model-controlled Snake demo, ask for a direction rather than an intermediate
progress category. State board facts without computing candidate distances or
a recommended action:

```json
{
  "state": "Board: 12 columns and 10 rows. Coordinates are (x, y); x increases right and y increases down. Head: (6, 5). Body: [(5, 5)]. Food: (4, 1). Current direction: right.",
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose the next direction to reach the food. Avoid walls and the body; do not reverse directly. Decide this move only.",
      "criteria": {
        "up": "Move one cell up (y - 1).",
        "down": "Move one cell down (y + 1).",
        "left": "Move one cell left (x - 1).",
        "right": "Move one cell right (x + 1)."
      }
    }
  }
}
```

## Disclose host assistance

Applications may enforce legality, precompute features, or use deterministic
policies. Those are application decisions and must be disclosed in evaluations.
Providing closer/farther categories and resolving the actual direction in host
code measures an assisted policy; it does not demonstrate that the model chose
the direction. Earlier Snake results used that approach and are not comparable
with the current direct-direction demo.

The [Python Snake demo](https://github.com/DestroyerDarkNess/TurboJev.Demos/tree/main/snake-python)
now offers all four directions on every turn and applies the returned choice
unchanged. The host checks collisions after the choice, which ends the game on
failure. There are no host-selected moves, rescues, probability fallbacks, or
skipped forced-move evaluations. Each turn logs its full request and response,
model choice, applied choice, and resulting position for audit.

## Test order, quality, and latency

The built-in adapters score candidates after an explicit answer prefill rather
than at the beginning of an unconstrained assistant message. This removes a
demonstrated first-option bias in the audited case without enabling cyclic
scoring. It does not guarantee correct reasoning: the raw-coordinate Snake
wall case still fails with Qwen3-0.6B Q4_K_M. See the complete
[candidate-order audit](validation/DECISION_ORDER.md), including failures.

Permute candidates to detect positional bias. Optional
`orderRobustness: "cyclic"` adds model evaluations and can reduce order bias,
but does not guarantee correct reasoning. Record its cost and configuration.

Use the exact model, quantization, runtime, and prompt used in deployment.
Report failures as well as successes, and separate model decisions from any
host intervention. Keep load time separate from inference time. A small model
may fail to navigate from coordinates; measure that failure rather than adding
a planner and reporting the resulting score as model-only performance.

Candidate probabilities are conditional scores, not calibrated confidence
unless an appropriate calibration profile has been fitted and applied.


## Reasoning and specialized models

Optional native text `thinking` is off by default. It generates bounded model
reasoning before the same typed candidate decision and reports its real token
usage. It is not a guarantee of correct planning. The audited coordinate-based
Snake states remain unqualified with the tested ordinary GGUFs. Valen's Sokoban
checkpoint has additional trained decision weights and is not unchanged Qwen
behind a different API. See [the detailed model and reasoning audit](MODEL_REASONING.md)
for the exact measurements, configuration examples and remaining limitations.
