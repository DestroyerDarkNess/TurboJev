import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const { TurboJev, Dice, compileJsonSchema, candidateSoftmax } = await import(
  pathToFileURL(resolve(process.argv[2])),
);
const nativePath = resolve(process.argv[3]);
const schema = {
  type: "object",
  properties: {
    label: { type: "string", enum: ["niño 🐈 中文"], permutations: 1 },
    count: { type: "integer", enum: [7] },
  },
};
assert.equal(compileJsonSchema(schema, { nativePath }).length, 2);
assert.deepEqual(candidateSoftmax({ café: 10000, 茶: 10000 }, 1, { nativePath }), {
  café: 0.5, 茶: 0.5,
});
assert.ok(JSON.stringify(new Dice(1).toSchema()).includes("six"));
const engine = await TurboJev.load({ model: "fixture://version-compatibility", nativePath });
try {
  const response = await engine.generate({ context: "niño 🐈 中文", schema });
  assert.deepEqual(response.result, { label: "niño 🐈 中文", count: 7 });
  assert.equal(response.usage.decisions, 2);
} finally {
  await engine.close();
}
console.log(JSON.stringify({ language: "node", version: process.version, arch: process.arch, passed: true }));
