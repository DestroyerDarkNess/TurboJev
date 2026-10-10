let runtime = null,
  descriptor = null,
  engine = null,
  wasmModule = null;
const reply = (id, ok, value) =>
  postMessage(ok ? { id, ok, value } : { id, ok, error: value?.message ?? String(value) });
const cancelled = new Set();
const pendingGenerations = new Set();
let activeControl = null;
function checkControl() {
  if (!activeControl) return;
  if (cancelled.has(activeControl.id))
    throw new Error("GENERATION_CANCELLED: generation cancelled");
  if (activeControl.deadlineMs !== undefined && Date.now() >= activeControl.deadlineMs) {
    throw new Error("GENERATION_TIMEOUT: generation deadline exceeded");
  }
}
function controlledRuntime(runtime) {
  const operations = new Set(["inferField", "scoreRuntimeBatch", "scoreDecisionBatch"]);
  return new Proxy(runtime, {
    get(target, name) {
      const value = target[name];
      if (!operations.has(name) || typeof value !== "function") return value;
      return async (...args) => {
        checkControl();
        const result = await value.apply(target, args);
        checkControl();
        return result;
      };
    },
  });
}

async function handleMessage({ data: { id, op, args } }) {
  try {
    if (op === "load") {
      const runtimeModule = await import(args.runtimeModuleUrl);
      runtime = controlledRuntime(await runtimeModule.createRuntime());
      descriptor = await runtime.initialize({ model: args.model, ...(args.runtimeOptions ?? {}) });
      const wasm = await import(args.wasmModuleUrl);
      wasmModule = wasm;
      if (typeof wasm.default === "function") await wasm.default();
      if (!wasm.WasmEngine) throw new Error("WASM module does not export WasmEngine");
      engine = new wasm.WasmEngine(
        runtime,
        JSON.stringify(descriptor),
        args.prompt ? JSON.stringify(args.prompt) : undefined,
        args.cyclicOrderRobustness === true,
        args.orderRobustness,
      );
      return reply(id, true, descriptor);
    }
    if (!engine) throw new Error("ENGINE_NOT_LOADED");
    if (op === "evaluate")
      return reply(id, true, JSON.parse(await engine.evaluate_json(JSON.stringify(args.request))));
    if (op === "candidateSoftmax")
      return reply(
        id,
        true,
        JSON.parse(wasmModule.candidate_softmax_json(JSON.stringify(args.input))),
      );
    if (op === "compileSchema")
      return reply(
        id,
        true,
        JSON.parse(wasmModule.compile_schema_json(JSON.stringify(args.schema))),
      );
    if (op === "generate") {
      activeControl = { id, deadlineMs: args.deadlineMs };
      try {
        checkControl();
        const result = JSON.parse(await engine.generate_json(JSON.stringify(args.request)));
        checkControl();
        return reply(id, true, result);
      } finally {
        activeControl = null;
        cancelled.delete(id);
      }
    }
    if (op === "compatible")
      return reply(
        id,
        true,
        JSON.parse(
          await engine.evaluate_compatible_json(args.profile, JSON.stringify(args.request)),
        ),
      );
    if (op === "info") return reply(id, true, descriptor);
    if (op === "close") {
      await runtime?.close?.();
      engine?.free?.();
      runtime = null;
      engine = null;
      return reply(id, true, null);
    }
    throw new Error(`unknown op ${op}`);
  } catch (e) {
    reply(id, false, e);
  }
}

// One loaded runtime is exclusive. Do not overlap generation, evaluation or close.
let queue = Promise.resolve();
onmessage = (event) => {
  if (event.data.op === "cancelGeneration") {
    if (pendingGenerations.has(event.data.args.requestId)) cancelled.add(event.data.args.requestId);
    return;
  }
  if (event.data.op === "generate") pendingGenerations.add(event.data.id);
  queue = queue.then(async () => {
    try {
      await handleMessage(event);
    } finally {
      pendingGenerations.delete(event.data.id);
      cancelled.delete(event.data.id);
    }
  });
};
