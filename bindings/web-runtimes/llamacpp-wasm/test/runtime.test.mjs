import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { createRuntime } from '../src/index.js';
import { reserveMediaWorkers } from '../src/media-worker-pool.js';

const workerSource = 'var Module=typeof Module!="undefined"?Module:{};';

function makeWllama({ image = false, audio = false, decisionError } = {}) {
  const calls = [];
  let modelSource;
  let exited = false;
  class Wllama {
    constructor(paths, options) {
      calls.push({ kind: 'construct', paths, options });
    }
    async loadModelFromUrl(source, options) {
      modelSource = source;
      calls.push({ kind: 'load', options, resources: this.getWorkerResources() });
    }
    getWorkerResources() { return { wasmPath: 'test.wasm', compat: false }; }
    supportInputModality(modality) {
      return modality === 'image' ? image : modality === 'audio' ? audio : false;
    }
    async createChatCompletion(options) {
      calls.push({ kind: 'completion', options });
      if (options.messages[0]?.content === 'Reply with one word: ready.') {
        return { choices: [{ message: { content: 'ready' } }] };
      }
      if (decisionError) throw decisionError;
      return {
        usage: { prompt_tokens: 12 },
        choices: [{ logprobs: { content: [{ top_logprobs: [
          { token: 'A', bytes: [65], logprob: -0.2 },
          { token: 'B', bytes: [66], logprob: -1.2 },
        ] }] } }],
      };
    }
    async exit() { exited = true; }
  }
  return {
    loadWllama: async () => ({ Wllama, LoggerWithoutDebug: {} }),
    loadWorkerCode: async () => workerSource,
    calls: () => calls,
    modelSource: () => modelSource,
    exited: () => exited,
  };
}

const options = extra => ({
  modelUrl: 'https://models.example/model.gguf',
  modelLabel: 'test model',
  modality: 'text',
  ...extra,
});

const task = {
  id: 'question', intent: 'criterion', instruction: 'Select the matching answer.',
  candidates: [{ key: 'first', description: 'First answer' }, { key: 'second', description: 'Second answer' }],
};

test('text uses the GGUF Jinja template and scores after the assistant prefill', async () => {
  const mock = makeWllama();
  const runtime = await createRuntime(mock);
  await runtime.initialize(options());
  await runtime.scoreDecisionBatch(JSON.stringify({ state: 'Evidence', tasks: [task] }));
  const load = mock.calls().find(call => call.kind === 'load').options;
  const completion = mock.calls().filter(call => call.kind === 'completion').at(-1).options;
  assert.equal(load.jinja, true);
  assert.equal(load.prefill_assistant, true);
  assert.equal(completion.messages.at(-1).content, '{"answer": "');
  assert.equal(completion.max_tokens, 1);
  assert.equal(completion.chat_template_kwargs.enable_thinking, false);
  await runtime.close();
});

async function evaluateMedia(modality, media, capabilities) {
  const mock = makeWllama(capabilities);
  const runtime = await createRuntime(mock);
  const descriptor = await runtime.initialize(options({ modality, mmprojUrl: 'https://models.example/mmproj.gguf' }));
  const output = JSON.parse(await runtime.scoreDecisionBatch(JSON.stringify({ state: media, tasks: [task] })));
  const completion = mock.calls().filter(call => call.kind === 'completion').at(-1).options;
  return { descriptor, output, completion, runtime, mock, modality };
}

test('loads a model and matching projector and advertises the detected modalities', async () => {
  const mock = makeWllama({ image: true, audio: true });
  const runtime = await createRuntime(mock);
  const descriptor = await runtime.initialize(options({ mmprojUrl: 'https://models.example/mmproj.gguf' }));
  assert.deepEqual(mock.modelSource(), {
    url: 'https://models.example/model.gguf',
    mmprojUrl: 'https://models.example/mmproj.gguf',
  });
  assert.equal(mock.calls().find(call => call.kind === 'load').options.jinja, false);
  assert.deepEqual(descriptor.modalities, ['text', 'image', 'video', 'audio']);
  await runtime.close();
  assert.equal(mock.exited(), true);
});

test('passes image bytes to llama.cpp WASM while preserving candidate score order', async () => {
  const result = await evaluateMedia('image', {
    contract: 'turbojev.media.v1', text: 'Inspect this picture.', modality: 'image', base64: 'AQID',
  }, { image: true });
  const content = result.completion.messages[1].content;
  assert.equal(result.completion.messages.length, 3);
  assert.deepEqual(result.completion.messages.at(-1), { role: 'assistant', content: '{"answer": "' });
  assert.equal(result.mock.calls().find(call => call.kind === 'load').options.prefill_assistant, true);
  assert.equal(content[1].type, 'image');
  assert.deepEqual([...new Uint8Array(content[1].data)], [1, 2, 3]);
  assert.match(content.at(-1).text, /Select the matching answer/);
  assert.deepEqual(result.output.scores[0].logits, [-0.2, -1.2]);
  await result.runtime.close();
});

test('loads an audio model without running a text-only warmup', async () => {
  const mock = makeWllama({ audio: true });
  const runtime = await createRuntime(mock);
  const descriptor = await runtime.initialize(options({ modality: 'audio', mmprojUrl: 'https://models.example/audio-mmproj.gguf' }));
  const load = mock.calls().find(call => call.kind === 'load').options;
  assert.equal(load.warmup, false);
  assert.deepEqual(descriptor.modalities, ['text', 'audio']);
  assert.equal(mock.calls().filter(call => call.kind === 'completion').length, 0);
  await runtime.close();
});

test('passes explicit Jinja, chat-template, and thread settings to Wllama', async () => {
  const mock = makeWllama({ image: true });
  const runtime = await createRuntime(mock);
  await runtime.initialize(options({ modality: 'image', jinja: false, chatTemplate: '{{ messages }}', nThreads: 2 }));
  const load = mock.calls().find(call => call.kind === 'load').options;
  assert.equal(load.jinja, false);
  assert.equal(load.chat_template, '{{ messages }}');
  assert.equal(load.n_threads, 2);
  await runtime.close();
});

test('passes real audio bytes to llama.cpp WASM and preserves decision scores', async () => {
  const result = await evaluateMedia('audio', {
    contract: 'turbojev.media.v1', text: 'Listen for the spoken command.', modality: 'audio', base64: 'AQID',
  }, { audio: true });
  const content = result.completion.messages[1].content;
  assert.equal(content[0].type, 'audio');
  assert.deepEqual([...new Uint8Array(content[0].data)], [1, 2, 3]);
  assert.equal(content[1].type, 'text');
  assert.match(content[1].text, /Listen for the spoken command/);
  assert.match(content.at(-1).text, /Select the matching answer/);
  assert.deepEqual(result.output.scores[0].logits, [-0.2, -1.2]);
  await result.runtime.close();
});

test('rejects audio input when the loaded GGUF has no audio encoder', async () => {
  const mock = makeWllama();
  const runtime = await createRuntime(mock);
  await runtime.initialize(options());
  await assert.rejects(runtime.scoreDecisionBatch(JSON.stringify({
    state: { contract: 'turbojev.media.v1', modality: 'audio', base64: 'AQ==' }, tasks: [task],
  })), /UNSUPPORTED_MODALITY.*does not support audio input/);
  assert.equal(mock.calls().filter(call => call.kind === 'completion').length, 1);
  await runtime.close();
});

test('passes video frames in timestamp order as image content blocks', async () => {
  const result = await evaluateMedia('video', {
    contract: 'turbojev.media.v1', text: 'What changes?', modality: 'video',
    frames: [
      { timestamp_ms: 100, base64: 'AQ==' },
      { timestamp_ms: 900, base64: 'Ag==' },
    ],
  }, { image: true });
  const content = result.completion.messages[1].content;
  assert.deepEqual(content.filter(part => part.type === 'image').map(part => [...new Uint8Array(part.data)]), [[1], [2]]);
  assert.match(content[1].text, /100 ms/);
  assert.match(content[3].text, /900 ms/);
  await result.runtime.close();
});

test('rejects a requested modality the loaded model does not expose', async () => {
  const mock = makeWllama();
  const runtime = await createRuntime(mock);
  await assert.rejects(
    runtime.initialize(options({ modality: 'image', mmprojUrl: 'https://models.example/mmproj.gguf' })),
    /UNSUPPORTED_MODALITY.*image/,
  );
  assert.equal(mock.exited(), true);
});

test('rejects malformed base64 rather than sending corrupt media to WASM', async () => {
  const mock = makeWllama({ image: true });
  const runtime = await createRuntime(mock);
  await runtime.initialize(options({ mmprojUrl: 'https://models.example/mmproj.gguf' }));
  await assert.rejects(runtime.scoreDecisionBatch(JSON.stringify({
    state: { contract: 'turbojev.media.v1', modality: 'image', base64: '%%%bad%%%' }, tasks: [task],
  })), /INVALID_MEDIA/);
  await runtime.close();
});

test('gives an actionable error when Wllama cannot format a model media prompt', async () => {
  const mock = makeWllama({ image: true, decisionError: new Error('Failed to format input: Failed to tokenize prompt') });
  const runtime = await createRuntime(mock);
  await runtime.initialize(options({ modality: 'image' }));
  await assert.rejects(runtime.scoreDecisionBatch(JSON.stringify({
    state: { contract: 'turbojev.media.v1', modality: 'image', base64: 'AQ==' }, tasks: [task],
  })), /MODEL_PROMPT_FORMAT_UNSUPPORTED.*projector verified with the selected Wllama version/);
  await runtime.close();
});

test('prestarts media workers while retaining the single-thread inference setting', async () => {
  const mock = makeWllama({ audio: true });
  const runtime = await createRuntime(mock);
  await runtime.initialize(options({ modality: 'audio', mmprojUrl: 'https://models.example/projector.gguf', nThreads: 1 }));
  const load = mock.calls().find(call => call.kind === 'load');
  assert.equal(load.options.n_threads, 1);
  const context = { Module: { pthreadPoolSize: 0, mainScriptUrlOrBlob: 'throw new Error("Multithreading is not enabled")' }, Blob };
  vm.runInNewContext(load.resources.jsPath.code, context);
  assert.equal(context.Module.pthreadPoolSize, 6);
  assert.ok(context.Module.mainScriptUrlOrBlob instanceof Blob);
  assert.equal(await context.Module.mainScriptUrlOrBlob.text(), workerSource);
  await runtime.close();
});

test('worker reservation works when Wllama embeds the glue and preserves a larger pool', () => {
  const patched = reserveMediaWorkers(workerSource, 4).replace('var Module', 'var ___Module');
  const context = { Module: { pthreadPoolSize: 16 } };
  vm.runInNewContext(patched, context);
  assert.equal(context.Module.pthreadPoolSize, 16);
  const defaultContext = { Module: {} };
  vm.runInNewContext(patched, defaultContext);
  assert.equal(defaultContext.Module.pthreadPoolSize, 12);
  assert.throws(() => reserveMediaWorkers('unknown glue', 4), /Unsupported Wllama worker/);
});

test('rejects audio before loading when SharedArrayBuffer isolation is missing', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
  Object.defineProperty(globalThis, 'crossOriginIsolated', { configurable: true, value: false });
  const mock = makeWllama({ audio: true });
  const runtime = await createRuntime(mock);
  try {
    await assert.rejects(runtime.initialize(options({ modality: 'audio' })), /cross-origin isolation.*COOP.*COEP/);
    assert.equal(mock.calls().length, 0);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'crossOriginIsolated', previous);
    else delete globalThis.crossOriginIsolated;
    await runtime.close();
  }
});



test('thinking requests fail explicitly rather than silently using the direct scorer', async () => {
  const deps = makeWllama();
  const runtime = await createRuntime(deps);
  await assert.rejects(runtime.initialize(options({ thinking: true })), /THINKING_UNSUPPORTED/);
  assert.equal(deps.calls().length, 0);
});
