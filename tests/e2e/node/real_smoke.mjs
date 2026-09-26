import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TurboJev } from "../../../bindings/node/dist/index.js";

const [modelPath, addonPath] = process.argv.slice(2);
const model = resolve(modelPath), nativePath = resolve(addonPath);
await assert.rejects(TurboJev.load({model: model + ".missing", nativePath}));
const engine = await TurboJev.load({model: pathToFileURL(model).href, nativePath, nGpuLayers: 0});
try {
  const request = message => ({ state: { message }, questions: { route: { type: "choice", instructions: "Which team should handle this?", criteria: { billing: "Payments", shipping: "Delivery", other: "Other" } } } });
  const results = await Promise.all(Array.from({length: 3}, () =>
    engine.evaluate(request("Mi paquete llegó roto 📦"))));
  for (const result of results) {
    assert.ok(["billing", "shipping", "other"].includes(result.answers.route.choice));
    assert.deepEqual(result.answers, results[0].answers);
  }
  await assert.rejects(engine.evaluate({ state: "hello", questions: { route: { type: "choice", instructions: "Choose", criteria: { only: "Only" } } } }));
  assert.ok(["billing", "shipping", "other"].includes((await engine.evaluate(request("Refund please"))).answers.route.choice));
} finally {
  await engine.close();
}
await engine.close();
await assert.rejects(engine.info());
console.log(JSON.stringify({passed: true, checks: ["failed_load_cleanup", "file_url_model",
  "unicode", "three_concurrent_calls", "invalid_options", "recovery", "close_twice", "use_after_close"]}));
