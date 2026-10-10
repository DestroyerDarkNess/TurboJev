import assert from "node:assert/strict";
import path from "node:path";
import { TurboJev, compileJsonSchema, candidateSoftmax } from "../bindings/node/dist/index.js";

const nativePath = path.resolve(process.argv[2]);
const schema = {
  type: "object",
  properties: { person: { type: "object", properties: { age: { type: "integer", enum: [37] } } } },
};
assert.deepEqual(compileJsonSchema(schema, { nativePath })[0].path, ["person", "age"]);
assert.deepEqual(candidateSoftmax({ a: 10000, b: 10000 }, 0.0001, { nativePath }), {
  a: 0.5,
  b: 0.5,
});
const client = await TurboJev.load({ model: "fixture://sdk-contract", nativePath });
try {
  const request = { context: "fixture contract", schema };
  assert.deepEqual((await client.generate(request)).result, { person: { age: 37 } });
  await assert.rejects(
    client.generate(request, { timeoutMs: 0 }),
    (error) => error.code === "GENERATION_TIMEOUT",
  );
  const control = new AbortController();
  control.abort();
  await assert.rejects(
    client.generate(request, { signal: control.signal }),
    (error) => error.code === "GENERATION_CANCELLED",
  );
  assert.equal((await client.generate(request)).usage.decisions, 1);
} finally {
  await client.close();
}
console.log("Node model-free compiler, normalization, nested results and worker controls passed.");
