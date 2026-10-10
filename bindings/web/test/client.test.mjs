import test from "node:test";
import assert from "node:assert/strict";
import { TurboJevWeb } from "../dist/index.js";

class Worker {
  messages = [];
  terminated = false;
  postMessage(message) {
    this.messages.push(message);
    if (message.op === "load" || message.op === "close") {
      queueMicrotask(() => this.onmessage({ data: { id: message.id, ok: true, value: {} } }));
    }
  }
  terminate() {
    this.terminated = true;
  }
}

test("browser client rejects a pre-aborted request without queueing cancellation", async () => {
  const previous = globalThis.Worker;
  let worker;
  globalThis.Worker = class extends Worker {
    constructor() {
      super();
      worker = this;
    }
  };
  try {
    const client = await TurboJevWeb.load({
      model: "fixture",
      runtimeModuleUrl: "fixture",
      wasmModuleUrl: "fixture",
    });
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(
      client.generate({ context: "", questions: {} }, { signal: abort.signal }),
      /GENERATION_CANCELLED/,
    );
    assert.deepEqual(
      worker.messages.map((message) => message.op),
      ["load"],
    );
    await client.close();
  } finally {
    if (previous === undefined) delete globalThis.Worker;
    else globalThis.Worker = previous;
  }
});

test("closing a browser client rejects new and outstanding operations", async () => {
  const previous = globalThis.Worker;
  let worker;
  globalThis.Worker = class extends Worker {
    constructor() {
      super();
      worker = this;
    }
  };
  try {
    const client = await TurboJevWeb.load({
      model: "fixture",
      runtimeModuleUrl: "fixture",
      wasmModuleUrl: "fixture",
    });
    const pending = assert.rejects(
      client.generate({ context: "", questions: {} }),
      /ENGINE_CLOSED/,
    );
    const closing = client.close();
    await assert.rejects(client.evaluate({}), /ENGINE_CLOSED/);
    await closing;
    await pending;
    assert.equal(worker.terminated, true);
  } finally {
    if (previous === undefined) delete globalThis.Worker;
    else globalThis.Worker = previous;
  }
});
