import assert from "node:assert/strict";
import test from "node:test";
import { createRuntime } from "../dist/index.js";

test("shared first tokens do not collapse distinct complete continuations", async () => {
  const prefixes = [];
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
    dispose() {}
  }
  const runtime = await createRuntime({
    loadTransformers: async () => ({
      Tensor,
      AutoTokenizer: { from_pretrained: async () => ({}) },
      AutoModelForCausalLM: {
        from_pretrained: async () => ({
          async forward({ input_ids }) {
            const ids = Array.from(input_ids.data, Number);
            prefixes.push(ids);
            // Candidates share token 1. Their second-token probabilities are different.
            return {
              logits: new Tensor(
                "float32",
                new Float32Array(ids.length === 1 ? [0, 3, 0, 0] : [0, 0, 1, 5]),
                [1, 1, 4],
              ),
            };
          },
        }),
      },
    }),
  });
  const descriptor = await runtime.initialize({ model: "fixture" });
  assert.equal(descriptor.capabilities.sequenceScores, true);
  const response = JSON.parse(
    await runtime.scoreRuntimeBatch(
      JSON.stringify({
        tasks: [
          {
            id: "route",
            promptTokens: [0],
            candidates: [{ tokenIds: [1, 2] }, { tokenIds: [1, 3] }],
          },
        ],
      }),
      "{}",
    ),
  );
  assert.ok(Math.abs(response.scores[0].logits[1] - response.scores[0].logits[0] - 4) < 1e-12);
  assert.deepEqual(prefixes, [[0], [0, 1], [0, 1]]);
  assert.equal(response.metrics.input_tokens, 5);
  await assert.rejects(
    runtime.inferField(JSON.stringify({ temperature: 0.5 })),
    /SAMPLING_UNSUPPORTED/,
  );
});
