export { Dice } from "./dice.js";
export { TurboJevRemote, imageEvidence, mediaEvidence, videoFrameEvidence } from "./remote.js";
export class TurboJevWebError extends Error {
    usage;
    constructor(message) {
        const [text, usage] = message.split("\nTURBOJEV_USAGE: ");
        super(text);
        this.name = "TurboJevWebError";
        if (usage)
            this.usage = JSON.parse(usage);
    }
}
export class TurboJevWeb {
    #worker;
    #id = 1;
    #pending = new Map();
    #closed = false;
    #closing = false;
    constructor(worker) {
        this.#worker = worker;
        worker.onmessage = ({ data }) => {
            const p = this.#pending.get(data.id);
            if (!p)
                return;
            this.#pending.delete(data.id);
            data.ok ? p.resolve(data.value) : p.reject(new TurboJevWebError(data.error));
        };
        worker.onerror = (e) => {
            for (const p of this.#pending.values())
                p.reject(e);
            this.#pending.clear();
        };
    }
    static async load(options) {
        const worker = new Worker(options.workerUrl ?? new URL("../worker.js", import.meta.url), {
            type: "module",
        });
        const x = new TurboJevWeb(worker);
        try {
            await x.#call("load", options);
            return x;
        }
        catch (error) {
            worker.terminate();
            throw error;
        }
    }
    async evaluate(request) {
        return this.#call("evaluate", { request });
    }
    async candidateSoftmax(logits, temperature = 1) {
        return this.#call("candidateSoftmax", {
            input: { logits, temperature },
        });
    }
    async compileSchema(schema) {
        return this.#call("compileSchema", { schema });
    }
    async generate(request, options = {}) {
        if (request.seed === undefined &&
            (request.mode === "sample" || (request.mode !== "argmax" && (request.temperature ?? 0) > 0))) {
            const random = crypto.getRandomValues(new Uint32Array(1));
            request = { ...request, seed: random[0] };
        }
        return this.#call("generate", { request }, options);
    }
    async evaluateCompatible(profile, request) {
        return this.#call("compatible", { profile, request });
    }
    async info() {
        return this.#call("info", {});
    }
    async close() {
        if (this.#closed || this.#closing)
            return;
        this.#closing = true;
        try {
            await this.#call("close", {});
        }
        finally {
            this.#closed = true;
            this.#worker.terminate();
            for (const pending of this.#pending.values())
                pending.reject(new Error("ENGINE_CLOSED"));
            this.#pending.clear();
        }
    }
    #call(op, args, options = {}) {
        if (this.#closed || (this.#closing && op !== "close"))
            return Promise.reject(new Error("ENGINE_CLOSED"));
        if (options.timeoutMs !== undefined &&
            (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 0)) {
            return Promise.reject(new Error("Timeout must be a finite non-negative number"));
        }
        if (options.signal?.aborted)
            return Promise.reject(new Error("GENERATION_CANCELLED: generation cancelled"));
        const id = this.#id++;
        return new Promise((resolve, reject) => {
            const cancel = () => this.#worker.postMessage({ op: "cancelGeneration", args: { requestId: id } });
            const cleanup = () => options.signal?.removeEventListener("abort", cancel);
            this.#pending.set(id, {
                resolve: (value) => {
                    cleanup();
                    resolve(value);
                },
                reject: (error) => {
                    cleanup();
                    reject(error);
                },
            });
            options.signal?.addEventListener("abort", cancel, { once: true });
            this.#worker.postMessage({
                id,
                op,
                args: {
                    ...args,
                    ...(options.timeoutMs === undefined
                        ? {}
                        : { deadlineMs: Date.now() + options.timeoutMs }),
                },
            });
        });
    }
}
