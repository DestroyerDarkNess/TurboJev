import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
test("worker delegates semantic decisions to Rust/WASM", async () => {
  const t = await readFile(new URL("../worker.js", import.meta.url), "utf8");
  assert.match(t, /WasmEngine/);
  assert.doesNotMatch(t, /function softmax|Math\.exp/);
  assert.doesNotMatch(t, /labelBase/);
});

test("Dice produces the same portable schema without deciding an answer", async () => {
  const { Dice } = await import("../dist/dice.js");
  const schema = new Dice(8).toSchema();
  assert.deepEqual(schema.enum, ["one", "two", "three", "four", "five", "six"]);
  assert.equal(schema.permutations, 8);
  assert.equal(schema.return_probabilities, true);
});
