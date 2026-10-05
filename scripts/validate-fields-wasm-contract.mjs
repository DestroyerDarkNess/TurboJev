import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Contract gate for the exact WASM package about to be published.
// The scorer is deterministic test data, not evidence of model accuracy.
const directory = path.resolve(process.argv[2]);
const core = await import(pathToFileURL(path.join(directory, 'turbojev_wasm.js')));
await core.default({ module_or_path: fs.readFileSync(path.join(directory, 'turbojev_wasm_bg.wasm')) });
let calls = 0;
const runtime = {
  async inferField(json) {
    calls++;
    const task = JSON.parse(json);
    return JSON.stringify({ value: task.name === 'count' ? 7 : null, logits: [], thinking: '', input_tokens: 5, generated_tokens: 3 });
  },
  async scoreDecisionBatch(json) {
    calls++;
    const batch = JSON.parse(json);
    return JSON.stringify({ scores: batch.tasks.map(task => ({ id: task.id, logits: task.candidates.map(c => c.key === 'true' ? 5 : -5) })), metrics: { input_tokens: 8 } });
  },
};
const descriptor = { name: 'contract-test', model: 'fixture', codecProfile: 'contract-fields-v1', scoringMode: 'decisions',
  modalities: ['text'], capabilities: { nextTokenLogits: false, sequenceScores: false, sharedPrefix: false,
    parallelSuffixes: false, selectiveOutputHead: false, localInference: true } };
const engine = new core.WasmEngine(runtime, JSON.stringify(descriptor), undefined, false);
const result = JSON.parse(await engine.generate_json(JSON.stringify({ context: 'Contract input', questions: {
  count: { type: 'integer' },
  reference: { type: ['string', 'null'] },
  escalate: { type: 'boolean', when: { count: { gte: 5 } }, return_probabilities: true },
  skipped: { type: 'string', when: { count: { lt: 5 } } },
  constant: { type: 'integer', enum: [9] },
} })));
assert.equal(result.result.count, 7);
assert.equal(result.result.reference, null);
assert.equal(result.result.escalate.value, true);
assert.equal(result.result.constant, 9);
assert.deepEqual(result.skipped, ['skipped']);
assert.equal(calls, 3);
const before = calls;
await assert.rejects(engine.generate_json(JSON.stringify({ context: '', questions: { bad: { type: 'integer', minimum: 0 } } })));
assert.equal(calls, before);
engine.free();
console.log('Published WASM contract: typed values, null, dependencies, skipped fields, probabilities and pre-inference rejection passed.');
