import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRuntime } from '../bindings/web-runtimes/llamacpp-wasm/src/index.js';
import { reserveMediaWorkers } from '../bindings/web-runtimes/llamacpp-wasm/src/media-worker-pool.js';

// This runs the real llama.cpp/mtmd WASM binary, adapter and Rust core in Node.
// It validates CPU inference; it does not emulate or qualify browser APIs.
// A disk-backed virtual FS avoids retaining a second full copy of large GGUFs.
// Prerequisites: Node 24+, @wllama/wllama 3.8.1 extracted locally, built core.
const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.chdir(rootDir);
if (typeof WebAssembly.Suspending !== 'function') throw new Error('Use Node 24+ with --experimental-wasm-jspi');
const args = process.argv.slice(2);
const textAudit = args.includes('--text');
const option = (name, fallback) => {
  const position = args.indexOf(name);
  if (position < 0) return fallback;
  if (!args[position + 1] || args[position + 1].startsWith('--')) throw new Error(`Missing value for ${name}`);
  return args[position + 1];
};
const base = path.resolve(option('--wllama', '.cache/wllama-3.8.1/package'));
if (JSON.parse(fs.readFileSync(`${base}/package.json`)).version !== '3.8.1') throw new Error('This harness requires Wllama 3.8.1');
const modelPath = path.resolve(option('--model', '.cache/real-validation/multimodal/lfm2-audio/LFM2-Audio-1.5B-Q8_0.gguf'));
const projectorPath = textAudit ? null : path.resolve(option('--mmproj', '.cache/real-validation/multimodal/lfm2-audio/mmproj-LFM2-Audio-1.5B-Q8_0.gguf'));
const corePath = path.resolve(option('--core', '.artifacts/wasm-tests/turbojev_wasm_bg.wasm'));
const coreModulePath = path.join(path.dirname(corePath), 'turbojev_wasm.js');
if (!fs.existsSync(coreModulePath) || !fs.existsSync(corePath)) {
  throw new Error('Prepare the WASM core with scripts/prepare-wasm-tests.ps1, or pass --core PATH_TO_WASM.');
}
const core = await import(pathToFileURL(coreModulePath));
const outputPath = path.resolve(option('--output', '.cache/multimodal-validation/audio-wasm-node.json'));
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
const scratch = path.resolve('.cache/wasm-audio-node');
const glueDir = `${scratch}/glue`;
process.on('uncaughtException', error => { console.error(error.message); process.exit(1); });
fs.mkdirSync(glueDir, { recursive: true });
for (const name of ['glue', 'messages']) {
  const text = fs.readFileSync(`${base}/src/glue/${name}.ts`, 'utf8').replaceAll("from './messages'", "from './messages.ts'").replaceAll("from './glue'", "from './glue.ts'");
  fs.writeFileSync(`${glueDir}/${name}.ts`, text);
}
const { glueSerialize, glueDeserialize } = await import(pathToFileURL(`${glueDir}/glue.ts`));
const started = performance.now();
const log = text => console.log(`${((performance.now() - started) / 1000).toFixed(1)}s ${text}`);
let resolveReady;
const ready = new Promise(resolve => resolveReady = resolve);
const filename = `${scratch}/runner.cjs`;
const wasmSource = fs.readFileSync(`${base}/src/wasm/wllama.js`, 'utf8');
fs.writeFileSync(filename, wasmSource);
const sandbox = {
  Module: {
    wasmBinary: fs.readFileSync(`${base}/src/wasm/wllama.wasm`),
    noInitialRun: true,
    onRuntimeInitialized: () => resolveReady(),
    print: text => console.log(text), printErr: text => console.error(text),
  },
  require: createRequire(filename), __filename: filename, __dirname: path.dirname(filename),
  process, Buffer, console, WebAssembly, TextDecoder, TextEncoder, performance, crypto: globalThis.crypto,
  setTimeout, clearTimeout, setInterval, clearInterval,
  navigator: { gpu: { requestAdapter: async () => null } },
};
sandbox.module = { exports: {} };
sandbox.global = sandbox;
vm.runInNewContext(reserveMediaWorkers(fs.readFileSync(filename, 'utf8'), 4), sandbox, { filename });
await ready;
const M = sandbox.Module;
log('WASM initialized');
M.FS.mkdir('/models');
for (const [virtualName, realPath] of [['model.gguf', modelPath], ...(projectorPath ? [['mmproj.gguf', projectorPath]] : [])]) {
  const fd = fs.openSync(realPath, 'r');
  M.FS.createDataFile('/models', virtualName, new Uint8Array(), true, false);
  const node = M.FS.lookupPath(`/models/${virtualName}`).node;
  node.usedBytes = fs.fstatSync(fd).size;
  node.stream_ops = { ...node.stream_ops, read(stream, buffer, offset, length, position) {
    return fs.readSync(fd, Buffer.from(buffer.buffer, buffer.byteOffset + offset, length), 0, length, position);
  } };
}
const malloc = M.cwrap('wllama_malloc', 'number', ['number', 'number']);
const start = M.cwrap('wllama_start', 'string', [], { async: true });
const call = M.cwrap('wllama_action', 'number', ['string', 'number'], { async: true });
async function action(name, body) {
  const bytes = glueSerialize(body);
  const ptr = Number(malloc(BigInt(bytes.length), 0));
  M.HEAPU8.set(bytes, ptr);
  const out = Number(await call(name, BigInt(ptr)));
  if (!out) throw new Error(`${name}: null response`);
  const length = new DataView(M.wasmMemory.buffer).getUint32(ptr, true);
  return glueDeserialize(new Uint8Array(M.wasmMemory.buffer, out, length).slice());
}
log(await start());
class Wllama {
  getWorkerResources() { return { wasmPath: 'local', compat: false }; }
  async loadModelFromUrl(source, options) {
    this.info = await action('load', {
      _name: 'load_req', model_paths: ['/models/model.gguf'], mmproj_path: projectorPath ? '/models/mmproj.gguf' : '',
      n_ctx_auto: false, use_mmap: false, use_mlock: false, n_parallel: 1,
      log_level: 2, n_ubatch: 512, ...options,
    });
    log(`Real model loaded: audio=${this.info.has_audio_input}`);
  }
  supportInputModality(modality) { return modality === 'audio' ? this.info.has_audio_input : this.info.has_image_input; }
  async createChatCompletion(options) {
    const files = [];
    const messages = options.messages.map(m => ({ ...m, content: Array.isArray(m.content) ? m.content.map(c => {
      if (c.type === 'text') return c;
      files.push(new Uint8Array(c.data));
      return { type: 'text', text: this.info.media_marker };
    }) : m.content }));
    const completion = await action('completion', {
      _name: 'cmpl_req', is_chat: true, data_json: JSON.stringify({ ...options, messages }), files,
    });
    let response;
    for (let i = 0; i < 2000; i++) {
      const result = await action('get_result', { _name: 'gres_req', req_id: completion.req_id });
      if (result._name === 'erro_evt' || result.is_error) throw new Error(result.message || result.data_json);
      if (result.data_json && result.data_json !== 'null') response = JSON.parse(result.data_json);
      if (!result.has_more) {
        if (!response) throw new Error('WASM returned no model scores');
        return response;
      }
    }
    throw new Error('WASM exceeded polling limit');
  }
  async exit() {}
}
const runtime = await createRuntime({
  loadWllama: async () => ({ Wllama, LoggerWithoutDebug: console }),
  loadWorkerCode: async () => wasmSource,
});
const descriptor = await runtime.initialize({
  modelUrl: 'local:model', ...(projectorPath ? { mmprojUrl: 'local:projector' } : {}), modelLabel: path.basename(modelPath),
  modality: textAudit ? 'text' : 'audio', device: 'wasm', nThreads: 4, jinja: true, warmup: false,
});
await core.default({ module_or_path: fs.readFileSync(corePath) });
const engine = new core.WasmEngine(runtime, JSON.stringify(descriptor), undefined, !textAudit);
if (textAudit) {
  const entries = [['dog', 'An animal'], ['billing', 'Payments and invoices'], ['shipping', 'Delivery, damage and replacement'], ['technical', 'Software bugs']];
  const results = [];
  for (let offset = 0; offset < entries.length; offset++) {
    const order = [...entries.slice(offset), ...entries.slice(0, offset)];
    const request = { state: 'My package arrived broken. Please send a replacement.', questions: {
      route: { type: 'choice', instructions: 'Which team should handle this request?', criteria: Object.fromEntries(order) },
    } };
    const begin = performance.now();
    const response = JSON.parse(await engine.evaluate_json(JSON.stringify(request)));
    results.push({ order: order.map(entry => entry[0]), request, response, seconds: (performance.now() - begin) / 1000 });
    fs.writeFileSync(outputPath, JSON.stringify({ host: 'Node CPU WASM, not browser', wllama: '3.8.1', model: modelPath, results }, null, 2));
    log(`TEXT first=${order[0][0]} selected=${response.answers.route.choice}`);
    if (response.answers.route.choice !== 'shipping') throw new Error(`Expected shipping, got ${response.answers.route.choice}`);
    if (response.usage.generated_tokens !== 0 || response._turbojev.order_evaluations !== 0) throw new Error('Unexpected additional inference');
  }
  engine.free();
  log('PASS: four real WASM text candidate permutations');
  process.exit(0);
}
const reports = [];
for (const clip of ['on', 'off']) {
  for (const reverse of [false, true]) {
    const criteria = Object.fromEntries(reverse ? [['off', 'Turn off the light'], ['on', 'Turn on the light']] : [['on', 'Turn on the light'], ['off', 'Turn off the light']]);
    const request = {
      state: { contract: 'turbojev.media.v1', modality: 'audio', text: '', base64: fs.readFileSync(`tests/fixtures/multimodal-audio/${clip}.wav`).toString('base64') },
      questions: { decision: { type: 'choice', instructions: 'Which command does the speaker say?', criteria } },
    };
    log(`Evaluating ${clip} ${reverse ? 'reversed' : 'normal'}`);
    const begin = performance.now();
    const response = JSON.parse(await engine.evaluate_json(JSON.stringify(request)));
    const answer = response.answers.decision;
    const elapsed = (performance.now() - begin) / 1000;
    reports.push({ clip, reverse, elapsed, response });
    fs.writeFileSync(outputPath, JSON.stringify({ host: 'node-wasm', runtime: 'wllama-3.8.1', device: 'cpu', nThreads: 4, node: process.version, platform: process.platform, arch: process.arch, model: modelPath, projector: projectorPath, results: reports }, null, 2));
    log(`DECISION ${clip} ${reverse ? 'reversed' : 'normal'} ${JSON.stringify(answer)} ${elapsed.toFixed(2)}s`);
    if (answer.type !== 'choice' || answer.choice !== clip) throw new Error(`Expected ${clip}, got ${answer.choice}`);
    if (response.usage.generated_tokens !== 0 || response._turbojev.order_evaluations !== 2) throw new Error('Invalid typed evaluation metadata');
  }
}
for (let i = 0; i < reports.length; i += 2) {
  for (const key of ['on', 'off']) {
    if (Math.abs(reports[i].response.answers.decision.probabilities[key] - reports[i + 1].response.answers.decision.probabilities[key]) > 1e-6) throw new Error('Candidate order changed probabilities');
  }
}
log('PASS: 4 real TurboJev WASM audio decisions');
engine.free();
process.exit(0);
