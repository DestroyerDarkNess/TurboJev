import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../worker.js", import.meta.url), "utf8");

async function createWorker() {
  const replies = [];
  const events = [];
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  const runtime = {
    initialize: async () => ({}),
    inferField: async () => {
      events.push("infer");
      await blocked;
      return "{}";
    },
    close: async () => events.push("close"),
  };
  class WasmEngine {
    constructor(provider) {
      this.runtime = provider;
    }
    async generate_json(input) {
      events.push("generate");
      await this.runtime.inferField(input);
      return '{"result":{}}';
    }
    async evaluate_json() {
      events.push("evaluate");
      return "{}";
    }
    free() {}
  }
  const context = vm.createContext({
    postMessage: (reply) => replies.push(reply),
    importRuntime: async () => ({ createRuntime: async () => runtime }),
    importCore: async () => ({ WasmEngine }),
    Date,
    Set,
    Proxy,
    Promise,
    Error,
    JSON,
  });
  vm.runInContext(
    source
      .replace("await import(args.runtimeModuleUrl)", "await importRuntime()")
      .replace("await import(args.wasmModuleUrl)", "await importCore()"),
    context,
  );
  const send = (id, op, args = {}) => context.onmessage({ data: { id, op, args } });
  const settle = async () => {
    await new Promise((resolve) => setImmediate(resolve));
  };
  send(1, "load", {});
  await settle();
  assert.equal(replies[0].ok, true);
  return { send, settle, replies, events, release };
}

test("cancellation bypasses an exclusive inference queue and permits reuse", async () => {
  const worker = await createWorker();
  worker.send(2, "generate", { request: {} });
  worker.send(3, "evaluate", { request: {} });
  await worker.settle();
  assert.deepEqual(worker.events, ["generate", "infer"]);
  worker.send(undefined, "cancelGeneration", { requestId: 2 });
  worker.release();
  await worker.settle();
  assert.match(worker.replies.find((reply) => reply.id === 2).error, /GENERATION_CANCELLED/);
  assert.equal(worker.replies.find((reply) => reply.id === 3).ok, true);
  assert.deepEqual(worker.events, ["generate", "infer", "evaluate"]);
  worker.send(4, "generate", { request: {} });
  await worker.settle();
  assert.equal(worker.replies.find((reply) => reply.id === 4).ok, true);
});

test("queued cancellation and expired deadline fail before model work", async () => {
  const worker = await createWorker();
  worker.send(2, "generate", { request: {}, deadlineMs: 0 });
  worker.send(3, "generate", { request: {} });
  worker.send(undefined, "cancelGeneration", { requestId: 3 });
  await worker.settle();
  assert.match(worker.replies.find((reply) => reply.id === 2).error, /GENERATION_TIMEOUT/);
  assert.match(worker.replies.find((reply) => reply.id === 3).error, /GENERATION_CANCELLED/);
  assert.deepEqual(worker.events, []);
  // A late cancellation for a completed ID must not poison a later operation.
  worker.send(undefined, "cancelGeneration", { requestId: 2 });
  worker.release();
  worker.send(4, "generate", { request: {} });
  await worker.settle();
  assert.equal(worker.replies.find((reply) => reply.id === 4).ok, true);
});
