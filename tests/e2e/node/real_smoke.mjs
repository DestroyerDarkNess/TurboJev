import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TurboJev } from "../../../bindings/node/dist/index.js";

const [modelPath, addonPath] = process.argv.slice(2);
const model = resolve(modelPath), nativePath = resolve(addonPath);
await assert.rejects(TurboJev.load({model: model + ".missing", nativePath}));
const engine = await TurboJev.load({model: pathToFileURL(model).href, nativePath, device:"cpu", nGpuLayers: 0,nCtx:8192,nSeqMax:1,nThreads:4});
try {
  const request = message => ({ state: { message }, questions: { route: { type: "choice", instructions: "Which team should handle this?", criteria: { billing: "Payments", shipping: "Delivery", other: "Other" } } } });
  const results = await Promise.all(Array.from({length: 3}, () =>
    engine.evaluate(request("Mi paquete llegó roto 📦"))));
  for (const result of results) {
    assert.ok(["billing", "shipping", "other"].includes(result.answers.route.choice));
    assert.deepEqual(result.answers, results[0].answers);
  }
  const multi = await engine.evaluate({state:"The service is down.", questions:{
    team:{type:"choice",instructions:"Which team handles this?",criteria:{outage:"Service down",billing:"Payments"}},
    urgent:{type:"noul",instructions:"Is the service unavailable?"}
  }});
  assert.deepEqual(Object.keys(multi.answers),["team","urgent"]);
  assert.equal(multi._turbojev.execution.prefix_reuse,"none");
  await assert.rejects(engine.evaluate({ state: "hello", questions: { route: { type: "choice", instructions: "Choose", criteria: { only: "Only" } } } }));
  assert.ok(["billing", "shipping", "other"].includes((await engine.evaluate(request("Refund please"))).answers.route.choice));
  const generated=await engine.generate({context:"There are exactly seven items.",questions:{count:{type:"integer",instructions:"How many items are stated?"},seen:{type:"boolean",depends_on:["count"],instructions:"Was an item count provided?"}}});
  assert.ok(Number.isInteger(generated.result.count));
  assert.equal(typeof generated.result.seen,"boolean");
  assert.ok(generated.usage.generated_tokens>0);
  await assert.rejects(engine.generate({context:"x",questions:{bad:{type:"number",minimum:0}}}));
} finally {
  await engine.close();
}
await engine.close();
await assert.rejects(engine.info());
console.log(JSON.stringify({passed: true, checks: ["failed_load_cleanup", "file_url_model",
  "unicode", "three_concurrent_calls", "single_sequence_multiple_questions", "invalid_options", "recovery", "generate_integer_dependency", "reject_unsupported_schema", "close_twice", "use_after_close"]}));
