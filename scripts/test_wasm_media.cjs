// Executes compiled WASM with model-independent protocol fixtures, not real inference.
const assert = require('node:assert/strict');
const path = require('node:path');
const { WasmEngine } = require(path.resolve(process.argv[2]));

const descriptor = {
  name: 'test-runtime', model: 'protocol-fixture', scoringMode: 'decisions',
  modalities: ['text', 'image', 'audio', 'video'],
  capabilities: {
    nextTokenLogits: false, sequenceScores: false, sharedPrefix: false,
    parallelSuffixes: false, selectiveOutputHead: false, localInference: true,
  },
};
const request = state => ({ state, questions: {
  scene: { type: 'choice', instructions: 'Choose', criteria: { a: 'First', b: 'Second' } },
  flag: { type: 'noul', instructions: 'True?' },
  level: { type: 'score', instructions: 'Rate', criteria: ['low', 'high'] },
} });

async function main() {
  let seen, calls = 0, invalidScores = false;
  const runtime = {
    async scoreDecisionBatch(batchJson) {
      calls++;
      const batch = JSON.parse(batchJson);
      seen = batch.state;
      return JSON.stringify({ scores: batch.tasks.map(t => ({ id: t.id, logits: invalidScores ? [1] : [0, 2] })) });
    },
  };
  const engine = new WasmEngine(runtime, JSON.stringify(descriptor));
  try {
    const baseline = JSON.parse(await engine.evaluate_json(JSON.stringify(request('text'))));
    assert.equal(baseline.answers.scene.choice, 'b');
    assert.ok(Math.abs(baseline.answers.scene.confidence - 1 / (1 + Math.exp(-2))) < 1e-6);
    for (const modality of ['image', 'audio', 'video']) {
      const state = { contract: 'turbojev.media.v1', text: 'context', modality, base64: 'AP8=' };
      const result = JSON.parse(await engine.evaluate_json(JSON.stringify(request(state))));
      assert.deepEqual(seen, state);
      assert.deepEqual(result, baseline);
    }
    invalidScores = true;
    await assert.rejects(engine.evaluate_json(JSON.stringify(request('text'))));
    invalidScores = false;
  } finally { engine.free(); }

  const textOnly = new WasmEngine(runtime, JSON.stringify({ ...descriptor, modalities: ['text'] }));
  const before = calls;
  try {
    await assert.rejects(textOnly.evaluate_json(JSON.stringify(request({
      contract: 'turbojev.media.v1', modality: 'image', base64: 'AA=='
    }))), /UNSUPPORTED_MODALITY/);
    assert.equal(calls, before);
  } finally { textOnly.free(); }
  assert.throws(() => new WasmEngine({}, JSON.stringify(descriptor)), /scoreDecisionBatch/);
  assert.throws(() => new WasmEngine(runtime, JSON.stringify({ ...descriptor, modalities: ['pdf'] })), /modalities/);
  assert.throws(() => new WasmEngine(runtime, JSON.stringify({ ...descriptor, scoringMode: 'tokens' })), /scoringMode/);
  console.log('WASM media protocol: 4 modalities, typed parity, invalid scores and capability rejection passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
