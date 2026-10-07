# Candidate-order robustness

`choice` candidates are often conceptually unordered, while many language
models show a positional preference. TurboJev can mitigate it with additional
model evaluations, without relying on a model-specific head.

The default policy is `none` (single-pass) for Rust, Node/TypeScript, Python,
C, C#/.NET, and browser engines. Opt into `balanced` when resistance to listing
order is more important than single-pass latency:

```json
{
  "model": "./model.gguf",
  "execution": { "orderRobustness": "balanced" }
}
```

This is a load-time native-host option, shared by Rust, Node/TypeScript,
Python, C, and .NET bindings. The browser/WASM engine takes an optional
`orderRobustness` string when loading. The legacy `cyclicOrderRobustness` boolean
remains supported; the explicit string takes precedence. The same Rust
aggregation serves token and decision runtimes.

In `cyclic` mode, a choice with `K` candidates evaluates all `K` cyclic rotations.
Each returned logit row is mapped back to the caller's original candidate keys;
the raw logits are averaged, then the configured calibrator computes the final
probabilities. The response records:

In `balanced` mode, canonical Williams layouts balance positions and immediate
neighbours: K evaluations for even K, 2K for odd K. Each layout is calibrated
once, then probabilities are averaged by original semantic key. The same set
produces the same layouts after arbitrary reorderings under deterministic
scoring. Cyclic rotations retain neighbour relationships from the caller's
listing and do not provide this general property. Neither mode guarantees
correct reasoning or calibrated confidence.

```json
"_turbojev": {
  "order_robustness": "balanced",
  "order_evaluations": 6
}
```

`order_evaluations` is the actual number of backend calls used for correction;
other question types in the same request add their normal call. `noul` and
`score` are not rotated because their candidate order carries meaning. This
uses one backend evaluation per candidate in each choice, so a four-option
choice costs four evaluations in either mode. Balanced with three choices costs
six evaluations. The default `none` policy scores a choice in one backend call;
its zero `order_evaluations` means no correction passes, not zero inference.

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
  execution: { orderRobustness: "balanced" },
});
```

Browser/WASM:

```js
// High-level browser load: TurboJevWeb.load({ model, runtimeModuleUrl,
// wasmModuleUrl, orderRobustness: "balanced" }).
// Direct WASM keeps the legacy fourth boolean; the fifth string takes precedence.
const engine = new wasm.WasmEngine(runtime, descriptorJson, undefined, false, "balanced");
```

Python:

```python
with TurboJev.load(
    "./model.gguf", execution={"orderRobustness": "balanced"}
) as jev:
    result = jev.evaluate({"state": "My package never arrived", "questions": {"route": {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": "Billing", "shipping": "Shipping"}}}})
```

C and .NET pass the same load-options JSON object to `tj_engine_open_json` or
`TurboJevEngine.LoadAsync`, respectively.

Rust uses `.with_order_robustness(OrderRobustness::Balanced)` on its engine.
Exact choice ties select the lexicographically smallest semantic key in every
mode. Unequal probabilities always select the larger value; there is no epsilon
tie window. Generated enum ties use JSON value encodings.

`generate` has a separate per-field `permutations` option (`1`, `"auto"`,
`"all"` or an integer budget). Auto uses the same canonical balanced design.
Integer sampling includes the original listing and does not promise invariance
after arbitrary shuffles. Use `evaluate`'s `score` for ordered rubric scoring;
a numeric enum is an unordered finite choice, not an implicit ordinal scale.
