class TurboJevWeb {
  #worker;
  #id = 1;
  #pending = /* @__PURE__ */ new Map();
  #closed = false;
  constructor(worker) {
    this.#worker = worker;
    worker.onmessage = ({ data }) => {
      const p = this.#pending.get(data.id);
      if (!p) return;
      this.#pending.delete(data.id);
      data.ok ? p.resolve(data.value) : p.reject(new Error(data.error));
    };
    worker.onerror = (e) => {
      for (const p of this.#pending.values()) p.reject(e);
      this.#pending.clear();
    };
  }
  static async load(options) {
    const worker = new Worker(options.workerUrl ?? new URL("../worker.js", import.meta.url), { type: "module" });
    const x = new TurboJevWeb(worker);
    try {
      await x.#call("load", options);
      return x;
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }
  async classify(state, options) {
    return this.#call("classify", { state, options });
  }
  async evaluate(request) {
    return this.#call("evaluate", { request });
  }
  async evaluateCompatible(profile, request) {
    return this.#call("compatible", { profile, request });
  }
  async info() {
    return this.#call("info", {});
  }
  async close() {
    if (this.#closed) return;
    try {
      await this.#call("close", {});
    } finally {
      this.#closed = true;
      this.#worker.terminate();
    }
  }
  #call(op, args) {
    if (this.#closed) return Promise.reject(new Error("ENGINE_CLOSED"));
    const id = this.#id++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, op, args });
    });
  }
}
export {
  TurboJevWeb
};
