import test from "node:test";
import assert from "node:assert/strict";
import { createRuntime } from "../dist/index.js";

test("adapter advertises implemented capabilities and defaults to CPU", async () => {
  let options;
  const runtime = await createRuntime({
    loadTransformers: async () => ({
      Tensor: class {},
      AutoTokenizer: { from_pretrained: async () => ({}) },
      AutoModelForCausalLM: { from_pretrained: async (_, input) => { options = input; return {}; } },
    }),
  });
  const descriptor = await runtime.initialize({ model: "fixture" });
  assert.equal(options.device, "wasm");
  assert.equal(descriptor.capabilities.sequenceScores, true);
  assert.equal(descriptor.capabilities.sharedPrefix, false);
  assert.equal(descriptor.capabilities.selectiveOutputHead, false);
  assert.equal(descriptor.capabilities.parallelSuffixes, false);
  await runtime.close();
});
