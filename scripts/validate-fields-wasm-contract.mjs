import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Contract gate for the exact WASM package about to be published.
// The scorer is deterministic test data, not evidence of model accuracy.
const directory = path.resolve(process.argv[2]);
const core = await import(pathToFileURL(path.join(directory, "turbojev_wasm.js")));
await core.default({
  module_or_path: fs.readFileSync(path.join(directory, "turbojev_wasm_bg.wasm")),
});
let calls = 0;
const runtime = {
  async inferField(json) {
    calls++;
    const task = JSON.parse(json);
    return JSON.stringify({
      value: task.name === "count" ? 7 : null,
      logits: [],
      thinking: "",
      input_tokens: 5,
      generated_tokens: 3,
    });
  },
  async scoreDecisionBatch(json) {
    calls++;
    const batch = JSON.parse(json);
    return JSON.stringify({
      scores: batch.tasks.map((task) => ({
        id: task.id,
        logits: task.candidates.map((c) => (c.key === "true" ? 5 : -5)),
      })),
      metrics: { input_tokens: 8 },
    });
  },
};
const descriptor = {
  name: "contract-test",
  model: "fixture",
  codecProfile: "contract-fields-v1",
  scoringMode: "decisions",
  modalities: ["text"],
  capabilities: {
    nextTokenLogits: false,
    sequenceScores: false,
    sharedPrefix: false,
    parallelSuffixes: false,
    selectiveOutputHead: false,
    localInference: true,
  },
};
const engine = new core.WasmEngine(runtime, JSON.stringify(descriptor), undefined, false);
const result = JSON.parse(
  await engine.generate_json(
    JSON.stringify({
      context: "Contract input",
      questions: {
        count: { type: "integer" },
        reference: { type: ["string", "null"] },
        escalate: {
          type: "boolean",
          when: { count: { gte: 5 } },
          return_probabilities: true,
          permutations: 1,
        },
        skipped: { type: "string", when: { count: { lt: 5 } } },
        constant: { type: "integer", enum: [9] },
      },
    }),
  ),
);
assert.equal(result.result.count, 7);
assert.equal(result.result.reference, null);
assert.equal(result.result.escalate.value, true);
assert.equal(result.result.constant, 9);
assert.deepEqual(result.skipped, ["skipped"]);
assert.equal(calls, 3);
const before = calls;
await assert.rejects(
  engine.generate_json(
    JSON.stringify({ context: "", questions: { bad: { type: "integer", minimum: 0 } } }),
  ),
);
assert.equal(calls, before);
const compiled = JSON.parse(
  core.compile_schema_json(
    JSON.stringify({
      type: "object",
      properties: {
        invoice: { type: "object", properties: { total: { type: "number" } } },
        rows: { type: "array", items: { type: "object", properties: { sku: { type: "string" } } } },
        category: { type: "string", enum: Array.from({ length: 512 }, (_, i) => `category_${i}`) },
      },
    }),
  ),
);
assert.deepEqual(
  compiled.map((field) => field.name),
  ["invoice.total", "rows", "category"],
);
assert.equal(compiled[2].choices.length, 512);
const probabilities = JSON.parse(
  core.candidate_softmax_json(
    JSON.stringify({ logits: { a: 1e300, b: 1e300 }, temperature: 1e-100 }),
  ),
);
assert.deepEqual(probabilities, { a: 0.5, b: 0.5 });
engine.free();
const balanced = new core.WasmEngine(
  runtime,
  JSON.stringify(descriptor),
  undefined,
  false,
  "balanced",
);
const evaluation = JSON.parse(
  await balanced.evaluate_json(
    JSON.stringify({
      state: "facts",
      questions: {
        route: {
          type: "choice",
          instructions: "Which option?",
          criteria: { z: "Z", a: "A", b: "B" },
        },
        severity: { type: "score", instructions: "Severity?", criteria: ["Low", "Medium", "High"] },
      },
    }),
  ),
);
assert.equal(evaluation.answers.route.choice, "a");
assert.equal(evaluation.answers.severity.score, 1);
assert.equal(evaluation._turbojev.order_robustness, "balanced");
assert.equal(evaluation._turbojev.order_evaluations, 7);
balanced.free();
assert.throws(
  () => new core.WasmEngine(runtime, JSON.stringify(descriptor), undefined, false, "invalid"),
);
console.log(
  "Published WASM contract: typed values, null, dependencies, skipped fields, probabilities and pre-inference rejection passed.",
);

const structuredRuntime = {
  async inferField(json) {
    const task = JSON.parse(json);
    if (task.values)
      return JSON.stringify({
        thinking: "",
        logits: task.values.map((value) =>
          value === "furious" || value === "option_31" ? 20 : -20,
        ),
        input_tokens: 4,
        generated_tokens: 0,
      });
    return JSON.stringify({
      value: task.name === "customer.age" ? 37 : null,
      logits: [],
      thinking: "",
      input_tokens: 4,
      generated_tokens: 1,
    });
  },
  async scoreDecisionBatch(json) {
    const batch = JSON.parse(json);
    return JSON.stringify({
      scores: batch.tasks.map((task) => ({
        id: task.id,
        logits: task.candidates.map((candidate) => (candidate.key === '\"furious\"' ? 20 : -20)),
      })),
      metrics: { input_tokens: 4 },
    });
  },
};
const structuredEngine = new core.WasmEngine(
  structuredRuntime,
  JSON.stringify(descriptor),
  undefined,
  false,
);
const structured = JSON.parse(
  await structuredEngine.generate_json(
    JSON.stringify({
      context: "Contract fixture",
      questions: {
        customer: { type: "object", properties: { age: { type: "integer" } } },
        tags: {
          type: "array",
          items: { type: "string" },
          continue_from: ["existing"],
          maxItems: 1,
        },
        tone: {
          type: "number",
          levels: ["calm", "annoyed", "furious"].map((label) => ({ label })),
          return_probabilities: true,
        },
        label: { type: "string", enum: Array.from({ length: 32 }, (_, i) => `option_${i}`) },
      },
    }),
  ),
);
assert.deepEqual(structured.result.customer, { age: 37 });
assert.deepEqual(structured.result.tags, ["existing"]);
assert.ok(structured.result.tone.value > 1.99);
assert.equal(structured.result.label, "option_31");
structuredEngine.free();
console.log(
  "Structured WASM contract: model-free compiler/normalizer, nested values, array prefix, ordered levels and full-value scorer dispatch passed.",
);
