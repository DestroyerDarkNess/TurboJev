# Candidate-order robustness

`choice` candidates are often conceptually unordered, while many language
models show a positional preference. TurboJev can remove a repeatable cyclic
position effect at inference time without relying on a model-specific head.

Enable the policy when the order of a `choice` question is not meaningful:

```json
{
  "model": "./model.gguf",
  "execution": { "orderRobustness": "cyclic" }
}
```

This is a load-time native-host option, shared by Rust, Node/TypeScript,
Python, C, and .NET bindings. `none` is the default.

For a choice with `K` candidates, TurboJev evaluates all `K` cyclic rotations.
Each returned logit row is mapped back to the caller's original candidate keys;
the raw logits are averaged, then the configured calibrator computes the final
probabilities. The response records:

```json
"_turbojev": {
  "orderRobustness": "cyclic",
  "orderEvaluations": 3
}
```

`orderEvaluations` is the actual number of backend calls used for correction;
other question types in the same request add their normal call. `noul` and
`score` are not rotated because their candidate order carries meaning.

The policy is bounded to 32 candidates per `choice`, is also respected by the
native `classify` helper, and deliberately disables cross-request fused
execution for that session. It is an explicit robustness-cost tradeoff, not a
claim of calibrated confidence. Fit or select a calibration profile for the
same model, prompt, and order policy used in production.

## Binding examples

Node/TypeScript:

```ts
const jev = await TurboJev.load({
  model: "./model.gguf",
  execution: { orderRobustness: "cyclic" },
});
```

Python:

```python
with TurboJev.load(
    "./model.gguf", execution={"orderRobustness": "cyclic"}
) as jev:
    result = jev.classify("My package never arrived", ["billing", "shipping"])
```

C and .NET pass the same load-options JSON object to `tj_engine_open_json` or
`TurboJevEngine.LoadAsync`, respectively.
