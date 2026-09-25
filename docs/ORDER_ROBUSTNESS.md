# Candidate-order robustness

`choice` candidates are often conceptually unordered, while many language
models show a positional preference. TurboJev can remove a repeatable cyclic
position effect at inference time without relying on a model-specific head.

The default policy is `none` (single-pass) for Rust, Node/TypeScript, Python,
C, C#/.NET, and browser token-scoring engines. Enable `cyclic` when resistance
to positional bias is more important than latency:

```json
{
  "model": "./model.gguf",
  "execution": { "orderRobustness": "cyclic" }
}
```

This is a load-time native-host option, shared by Rust, Node/TypeScript,
Python, C, and .NET bindings. The browser/WASM engine takes an optional
`cyclicOrderRobustness` boolean when loading. Custom decision-scoring runtimes
keep their own scoring behavior.

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
`score` are not rotated because their candidate order carries meaning. This
uses one backend evaluation per candidate in each choice, so a four-option
choice costs four evaluations. The default `none` policy scores a four-option
choice in one backend evaluation.

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

Browser/WASM:

```js
// Pass true as the optional fourth WasmEngine constructor argument to enable
// cyclic order correction. Omit it or pass false for single-pass inference.
const engine = new wasm.WasmEngine(runtime, descriptorJson, undefined, true);
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
