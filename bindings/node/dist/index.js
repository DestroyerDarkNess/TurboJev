import { Worker } from "node:worker_threads";
import { resolveModel } from "./model.js";
import { normalizeError, TurboJevError } from "./errors.js";
export * from "./types.js";
export { TurboJevError } from "./errors.js";
export class TurboJev {
    #worker;
    #next = 1;
    #pending = new Map();
    #closed = false;
    constructor(worker) { this.#worker = worker; worker.on("message", (m) => { const p = this.#pending.get(m.id); if (!p)
        return; this.#pending.delete(m.id); m.ok ? p.resolve(m.value) : p.reject(normalizeError(m.error)); }); worker.on("error", (e) => this.#fail(e)); worker.on("exit", (c) => { if (!this.#closed && c !== 0)
        this.#fail(new Error(`worker exited ${c}`)); }); }
    static async load(options) { const model = await resolveModel(options); const worker = new Worker(new URL("../worker.mjs", import.meta.url), { workerData: { options: { ...options, model } } }); const jev = new TurboJev(worker); try {
        await jev.#call("ready", {});
        return jev;
    }
    catch (error) {
        await worker.terminate();
        throw error;
    } }
    async evaluate(request) { return JSON.parse(await this.#call("evaluate", { request: JSON.stringify(request) })); }
    async generate(request) { return JSON.parse(await this.#call("generate", { request: JSON.stringify(request) })); }
    async evaluateCompatible(profile, request) { return JSON.parse(await this.#call("compatible", { profile, request: JSON.stringify(request) })); }
    async info() { return JSON.parse(await this.#call("info", {})); }
    async close() { if (this.#closed)
        return; try {
        await this.#call("close", {});
    }
    finally {
        this.#closed = true;
        await this.#worker.terminate();
    } }
    #call(op, args) { if (this.#closed)
        return Promise.reject(new TurboJevError("ENGINE_CLOSED", "engine is closed")); const id = this.#next++; return new Promise((resolve, reject) => { this.#pending.set(id, { resolve, reject }); this.#worker.postMessage({ id, op, args }); }); }
    #fail(error) { for (const p of this.#pending.values())
        p.reject(normalizeError(error)); this.#pending.clear(); }
}
