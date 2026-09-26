# Designing decisions for open-weight models

TurboJev scores the candidates supplied by an application. It does not turn a
small model into a general planner. Decision quality depends on whether the
selected model, its prompt/template, and the candidate descriptions can express
the distinction the application needs.

## Make the decision explicit

Small models can struggle when they must infer relationships from terse
structured values, do arithmetic, or plan several actions ahead. JSON and
coordinates are valid state, but they may hide the important relationship from
a small checkpoint. State the relevant facts plainly, including their meaning
for the decision.

For example, a Snake game can provide `head=(6,5)` and `food=(4,1)`, but should
also say that the food is two squares west and four squares north. Then ask for
one decision: which legal move category reduces the distance to the food?
Avoid asking a small model to plan a complete route when the host can make one
safe, local decision at a time.

Keep the request as short as the decision allows. Do not repeat the same move
list in the state, instructions, and candidate descriptions; extra prompt text
adds model prefill work and can increase latency. Preserve the facts that affect
the choice, then measure both quality and per-decision time after editing.

```json
{
  "state": "Snake head at (6, 5); food at (4, 1). The food is 2 squares west and 4 squares north. Current distance is 6. Safe moves: up reduces distance to 5; down increases it to 7; left reduces it to 5.",
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose the progress category that leaves the head at the smallest Manhattan distance from the food. This is one step, not a full route plan.",
      "criteria": {
        "closer": "A safe legal move that reduces the distance to the food.",
        "farther": "A safe legal move that increases the distance to the food."
      }
    }
  }
}
```

The example uses semantic candidate names and descriptions rather than opaque
labels such as `A`, `B`, or `C`. Include only candidates that are possible in
the current state. If the game rules leave just one possible decision class,
the host can apply it without spending a model evaluation.

## Keep constraints in the host

Use TurboJev for the semantic choice. Keep hard rules in application code:
remove illegal or unsafe candidates before evaluation, map the selected
candidate to an action, and resolve equivalent actions with a deterministic
tie-break. This lets the model make the intended decision without asking it to
enforce invariants or choose arbitrarily between equally good outcomes.

## Treat option order as a test variable

Some models favor candidates based on their position. Test the same request
with candidate orders permuted. If order changes the answer, try the optional
`orderRobustness: "cyclic"` policy and compare its quality and latency on the
target model. Cyclic scoring adds model evaluations; it can reduce positional
bias but does not fix weak reasoning or guarantee the right answer. See
[candidate-order robustness](ORDER_ROBUSTNESS.md) for its cost and behavior.

## Validate the whole task

Use the exact model, quantization, runtime, and prompt path intended for
deployment. Run reproducible cases where a simple reference policy can verify
that the task is reachable, then compare TurboJev's decisions and latency over
multiple inputs or seeds. A strong result on one example does not establish
general reliability, and candidate probabilities are not calibrated
confidence unless a matching calibration profile was fitted.

The [Python Snake demo](https://github.com/DestroyerDarkNess/TurboJev.Demos/tree/main/snake-python)
applies this pattern and reports its score beside a simple reference policy.
