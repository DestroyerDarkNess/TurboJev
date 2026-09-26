# Candidate-order robustness

`choice` candidates are often conceptually unordered, while many language
models show a positional preference. TurboJev can remove a repeatable cyclic
position effect at inference time without relying on a model-specific head.

Rust core defaults to `none`; native host bindings default to `cyclic`. Both
corrected modes perform one candidate-level inference per choice option.
`cyclic` averages rotated candidate lists, while `binary` asks an independent yes/no match
question for each candidate and combines those scores. Choose `binary` when
comparing every option in one list produces unstable results:

```json
{
  "model": "./model.gguf",
  "execution": { "orderRobustness": "binary" }
}
```

This is a load-time native-host option, shared by Rust, Node/TypeScript,
Python, C, and .NET bindings. The browser/WASM engine takes an optional
`cyclicOrderRobustness` boolean when loading. Custom decision-scoring runtimes
keep their own scoring behavior.

For a choice with `K` candidates, `cyclic` evaluates all `K` rotations and maps
each score row back to the original keys before averaging logits. `binary`
evaluates each candidate as its own yes/no task and uses the yes-minus-no score
as that candidate's evidence. The configured calibrator then computes the
final probabilities. The response records:

```json
"_turbojev": {
  "orderRobustness": "binary",
  "orderEvaluations": 3
}
```

`orderEvaluations` is the number of candidate-level evaluations used for
correction. `noul` and `score` remain single-pass because their candidate order
carries meaning. A four-option choice uses four candidate evaluations; `none`
uses one list evaluation.

The policy is bounded to 32 candidates per `choice`, is also respected by the
native evaluation path, and deliberately disables cross-request fused
execution for that session. It is an explicit robustness-cost tradeoff, not a
claim of calibrated confidence. Fit or select a calibration profile for the
same model, prompt, and order policy used in production.

## Binding examples

Node/TypeScript:

```ts
const jev = await TurboJev.load({
  model: "./model.gguf",
  execution: { orderRobustness: "binary" },
});
```

Browser/WASM:

```js
// Pass true as the optional fourth WasmEngine constructor argument to enable
// cyclic order correction. Omit it or pass false for single-pass inference.
const engine = new wasm.WasmEngine(runtime, descriptorJson, undefined, true);
```

Python:

```python
with TurboJev.load(
    "./model.gguf", execution={"orderRobustness": "binary"}
) as jev:
    result = jev.evaluate({"state": "My package never arrived", "questions": {"route": {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": "Billing", "shipping": "Shipping"}}}})
```

C and .NET pass the same load-options JSON object to `tj_engine_open_json` or
`TurboJevEngine.LoadAsync`, respectively.
