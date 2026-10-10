import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createRuntime } from "../bindings/web-runtimes/llamacpp-wasm/src/index.js";
import { reserveMediaWorkers } from "../bindings/web-runtimes/llamacpp-wasm/src/media-worker-pool.js";

// This runs the real llama.cpp/mtmd WASM binary, adapter and Rust core in Node.
// It validates CPU inference; it does not emulate or qualify browser APIs.
// A disk-backed virtual FS avoids retaining a second full copy of large GGUFs.
// Prerequisites: Node 24+, @wllama/wllama 3.8.1 extracted locally, built core.
const rootDir = fileURLToPath(new URL("../", import.meta.url));
process.chdir(rootDir);
if (typeof WebAssembly.Suspending !== "function")
  throw new Error("Use Node 24+ with --experimental-wasm-jspi");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const position = args.indexOf(name);
  if (position < 0) return fallback;
  if (!args[position + 1] || args[position + 1].startsWith("--"))
    throw new Error(`Missing value for ${name}`);
  return args[position + 1];
};
const base = path.resolve(option("--wllama", ".cache/wllama-3.8.1/package"));
if (JSON.parse(fs.readFileSync(`${base}/package.json`)).version !== "3.8.1")
  throw new Error("This harness requires Wllama 3.8.1");
const modelPath = path.resolve(
  option(
    "--model",
    `${process.env.USERPROFILE || process.env.HOME}/.cache/turbojev/Qwen3-0.6B-Q4_K_M.gguf`,
  ),
);
const projectorPath = path.resolve(
  option(
    "--mmproj",
    ".cache/real-validation/multimodal/lfm2-audio/mmproj-LFM2-Audio-1.5B-Q8_0.gguf",
  ),
);
const corePath = path.resolve(option("--core", ".cache/fields-wasm-web/turbojev_wasm_bg.wasm"));
const outputPath = path.resolve(option("--output", ".cache/fields-wasm-validation.json"));
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
const scratch = path.resolve(".cache/fields-wasm-node");
const glueDir = `${scratch}/glue`;
process.on("uncaughtException", (error) => {
  console.error(error.message);
  process.exit(1);
});
fs.mkdirSync(glueDir, { recursive: true });
for (const name of ["glue", "messages"]) {
  const text = fs
    .readFileSync(`${base}/src/glue/${name}.ts`, "utf8")
    .replaceAll("from './messages'", "from './messages.ts'")
    .replaceAll("from './glue'", "from './glue.ts'");
  fs.writeFileSync(`${glueDir}/${name}.ts`, text);
}
const { glueSerialize, glueDeserialize } = await import(pathToFileURL(`${glueDir}/glue.ts`));
const started = performance.now();
const log = (text) => console.log(`${((performance.now() - started) / 1000).toFixed(1)}s ${text}`);
let resolveReady;
const ready = new Promise((resolve) => (resolveReady = resolve));
const filename = `${scratch}/runner.cjs`;
const wasmSource = fs.readFileSync(`${base}/src/wasm/wllama.js`, "utf8");
fs.writeFileSync(filename, wasmSource);
const sandbox = {
  Module: {
    wasmBinary: fs.readFileSync(`${base}/src/wasm/wllama.wasm`),
    noInitialRun: true,
    onRuntimeInitialized: () => resolveReady(),
    print: (text) => console.log(text),
    printErr: (text) => console.error(text),
  },
  require: createRequire(filename),
  __filename: filename,
  __dirname: path.dirname(filename),
  process,
  Buffer,
  console,
  WebAssembly,
  TextDecoder,
  TextEncoder,
  performance,
  crypto: globalThis.crypto,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  navigator: { gpu: { requestAdapter: async () => null } },
};
sandbox.module = { exports: {} };
sandbox.global = sandbox;
vm.runInNewContext(reserveMediaWorkers(fs.readFileSync(filename, "utf8"), 4), sandbox, {
  filename,
});
await ready;
const M = sandbox.Module;
log("WASM initialized");
M.FS.mkdir("/models");
for (const [virtualName, realPath] of [["model.gguf", modelPath]]) {
  const fd = fs.openSync(realPath, "r");
  M.FS.createDataFile("/models", virtualName, new Uint8Array(), true, false);
  const node = M.FS.lookupPath(`/models/${virtualName}`).node;
  node.usedBytes = fs.fstatSync(fd).size;
  node.stream_ops = {
    ...node.stream_ops,
    read(stream, buffer, offset, length, position) {
      return fs.readSync(
        fd,
        Buffer.from(buffer.buffer, buffer.byteOffset + offset, length),
        0,
        length,
        position,
      );
    },
  };
}
const malloc = M.cwrap("wllama_malloc", "number", ["number", "number"]);
const start = M.cwrap("wllama_start", "string", [], { async: true });
const call = M.cwrap("wllama_action", "number", ["string", "number"], { async: true });
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
  getLoadedContextInfo() {
    return this.info;
  }
  getWorkerResources() {
    return { wasmPath: "local", compat: false };
  }
  async loadModelFromUrl(source, options) {
    this.info = await action("load", {
      _name: "load_req",
      model_paths: ["/models/model.gguf"],
      mmproj_path: "",
      n_ctx_auto: false,
      use_mmap: false,
      use_mlock: false,
      n_parallel: 1,
      log_level: 2,
      n_ubatch: 512,
      ...options,
    });
    log(`Real model loaded: audio=${this.info.has_audio_input}`);
  }
  supportInputModality(modality) {
    return modality === "audio" ? this.info.has_audio_input : this.info.has_image_input;
  }
  async createChatCompletion(options) {
    const files = [];
    const messages = options.messages.map((m) => ({
      ...m,
      content: Array.isArray(m.content)
        ? m.content.map((c) => {
            if (c.type === "text") return c;
            files.push(new Uint8Array(c.data));
            return { type: "text", text: this.info.media_marker };
          })
        : m.content,
    }));
    const completion = await action("completion", {
      _name: "cmpl_req",
      is_chat: true,
      data_json: JSON.stringify({ ...options, messages }),
      files,
    });
    let response;
    for (let i = 0; i < 2000; i++) {
      const result = await action("get_result", { _name: "gres_req", req_id: completion.req_id });
      if (result._name === "erro_evt" || result.is_error)
        throw new Error(result.message || result.data_json);
      if (result.data_json && result.data_json !== "null") response = JSON.parse(result.data_json);
      if (!result.has_more) {
        if (!response) throw new Error("WASM returned no model scores");
        if (args.includes("--debug-scores") && options.logprobs)
          log(
            JSON.stringify({
              requested: options.top_logprobs,
              returned: response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs?.length,
            }),
          );
        return response;
      }
    }
    throw new Error("WASM exceeded polling limit");
  }
  async exit() {}
}

const runtime = await createRuntime({
  loadWllama: async () => ({ Wllama, LoggerWithoutDebug: console }),
  loadWorkerCode: async () => wasmSource,
});
const descriptor = await runtime.initialize({
  modelUrl: "local:model",
  modelLabel: "Qwen3-0.6B-Q4_K_M",
  modality: "text",
  device: "wasm",
  nThreads: 4,
  jinja: true,
  warmup: false,
});
const core = await import(pathToFileURL(path.join(path.dirname(corePath), "turbojev_wasm.js")));
await core.default({ module_or_path: fs.readFileSync(corePath) });
const engine = new core.WasmEngine(runtime, JSON.stringify(descriptor), undefined, false);
const cases = [
  [
    "integer",
    {
      context: "There are exactly 37 units.",
      questions: { count: { type: "integer", instructions: "How many units?" } },
    },
    { count: 37 },
  ],
  [
    "string",
    {
      context: "The merchant is Hilton London.",
      questions: { merchant: { type: "string", instructions: "Return only the merchant name." } },
    },
    { merchant: "Hilton London" },
  ],
  [
    "nullable",
    {
      context: "No invoice reference is provided.",
      questions: {
        reference: {
          type: ["string", "null"],
          instructions: "Return the invoice reference, or null when absent.",
        },
      },
    },
    { reference: null },
  ],
  [
    "finite",
    {
      context: "The service is down.",
      questions: {
        team: {
          type: "string",
          enum: ["outage", "billing", "feature"],
          instructions: "Which team handles this?",
        },
        urgent: { type: "boolean", instructions: "Is this urgent?" },
      },
    },
    { team: "outage", urgent: true },
  ],
  [
    "24_candidates",
    {
      context: "The explicitly stated label is option_23.",
      questions: {
        label: {
          type: "string",
          enum: Array.from({ length: 24 }, (_, i) => `option_${i}`),
          instructions: "Which label is explicitly stated?",
          return_probabilities: true,
        },
      },
    },
    null,
  ],
];
cases.push([
  "structured",
  {
    context: "Customer Ana is 37 years old. Tags: red, blue. Do not invent other tags.",
    questions: {
      customer: {
        type: "object",
        properties: {
          name: { type: "string", instructions: "Extract the customer's name." },
          age: { type: "integer", instructions: "Extract the customer's age." },
        },
      },
      tags: {
        type: "array",
        instructions: "Return only the explicitly stated tags, then stop.",
        items: { type: "string" },
        maxItems: 2,
      },
    },
  },
  { customer: { name: "Ana", age: 37 }, tags: ["red", "blue"] },
]);
const results = [];
for (const [id, request, expected] of cases.filter(([id]) => option("--only", id) === id)) {
  const begin = performance.now();
  try {
    const response = JSON.parse(await engine.generate_json(JSON.stringify(request)));
    results.push({
      id,
      request,
      response,
      expected,
      matched:
        expected === null ? null : JSON.stringify(response.result) === JSON.stringify(expected),
      seconds: (performance.now() - begin) / 1000,
    });
  } catch (error) {
    results.push({
      id,
      request,
      error: String(error),
      seconds: (performance.now() - begin) / 1000,
    });
  }
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        host: "Real Node CPU WASM, not browser API qualification",
        wllama: "3.8.1",
        model: modelPath,
        results,
      },
      null,
      2,
    ),
  );
  log(JSON.stringify(results.at(-1)));
}
engine.free();
if (results.some((r) => r.error)) process.exit(1);
log("Completed real WASM typed-field cases; inspect accuracy separately.");
process.exit(0);
