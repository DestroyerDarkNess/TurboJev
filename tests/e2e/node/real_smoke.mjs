import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { TurboJev } from "../../../bindings/node/dist/index.js";

const [modelPath, addonPath] = process.argv.slice(2);
const model = resolve(modelPath), nativePath = resolve(addonPath);
await assert.rejects(TurboJev.load({model: model + ".missing", nativePath}));
const engine = await TurboJev.load({model: pathToFileURL(model).href, nativePath, nGpuLayers: 0});
try {
  const options = ["billing", "shipping", "other"];
  const results = await Promise.all(Array.from({length: 3}, () =>
    engine.classify("Mi paquete llegó roto 📦", options)));
  for (const result of results) {
    assert.ok(options.includes(result.choice));
    assert.deepEqual(result, results[0]);
  }
  await assert.rejects(engine.classify("hello", ["only"]));
  assert.ok(options.includes((await engine.classify("Refund please", options)).choice));
} finally {
  await engine.close();
}
await engine.close();
await assert.rejects(engine.info());
console.log(JSON.stringify({passed: true, checks: ["failed_load_cleanup", "file_url_model",
  "unicode", "three_concurrent_calls", "invalid_options", "recovery", "close_twice", "use_after_close"]}));
