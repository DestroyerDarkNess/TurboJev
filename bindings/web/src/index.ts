export type {
  GenerateRequest,
  GenerateResponse,
  FieldSchema,
  ScalarField,
  ObjectField,
  ArrayField,
  ProbabilityAnswer,
  CompiledDecision,
  JsonValue,
  ScalarValue,
} from "./fields.js";
import type { GenerateRequest, GenerateResponse, CompiledDecision } from "./fields.js";
export { TurboJevRemote, imageEvidence, mediaEvidence, videoFrameEvidence } from "./remote.js";
export type { MediaEvidence, MediaModality, VideoFrame, VideoFrameInput } from "./remote.js";
export interface BrowserRuntimeDescriptor {
  name: string;
  model: string;
  codecProfile?: string;
  scoringMode?: "tokens" | "decisions";
  modalities?: Array<"text" | "image" | "audio" | "video">;
  promptProfile?: { profile: string; systemInstruction: string; bestMatchInstruction: string };
  capabilities: {
    nextTokenLogits: boolean;
    sequenceScores: boolean;
    sharedPrefix: boolean;
    parallelSuffixes: boolean;
    selectiveOutputHead: boolean;
    localInference: boolean;
  };
}
export interface BrowserRuntime {
  initialize(options: unknown): Promise<BrowserRuntimeDescriptor>;
  encodePrompt(messagesJson: string): string;
  encodePromptWithAnswer(messagesJson: string, answer: string): string;
  decodeTokens(tokenIdsJson: string): string;
  scoreRuntimeBatch(runtimeBatchJson: string, planJson: string): Promise<string> | string;
  inferField?(taskJson: string): Promise<string> | string;
  close(): Promise<void> | void;
}
/** Model-independent scoring; the provider owns media decoding and preprocessing. */
export interface BrowserDecisionRuntime {
  initialize(options: unknown): Promise<
    BrowserRuntimeDescriptor & {
      scoringMode: "decisions";
      modalities: Array<"text" | "image" | "audio" | "video">;
    }
  >;
  scoreDecisionBatch(batchJson: string, planJson: string): Promise<string> | string;
  /** Optional bounded, schema-constrained inference; workflow semantics stay in Rust. */
  inferField?(taskJson: string): Promise<string> | string;
  close(): Promise<void> | void;
}
export interface WebLoadOptions {
  model: string;
  runtimeModuleUrl: string;
  wasmModuleUrl: string;
  runtimeOptions?: unknown;
  /** Defaults to none; balanced costs K layouts (2K for odd K). */ orderRobustness?:
    | "none"
    | "cyclic"
    | "balanced";
  /** Legacy alias; orderRobustness takes precedence. */ cyclicOrderRobustness?: boolean;
  prompt?: { profile?: string; systemInstruction?: string; bestMatchInstruction?: string };
  workerUrl?: string;
}
export class TurboJevWeb {
  #worker: Worker;
  #id = 1;
  #pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
  #closed = false;
  #closing = false;
  private constructor(worker: Worker) {
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
  static async load(options: WebLoadOptions) {
    const worker = new Worker(options.workerUrl ?? new URL("../worker.js", import.meta.url), {
      type: "module",
    });
    const x = new TurboJevWeb(worker);
    try {
      await x.#call("load", options);
      return x;
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }
  async evaluate(request: Record<string, unknown>) {
    return this.#call("evaluate", { request });
  }
  async candidateSoftmax(
    logits: Record<string, number>,
    temperature = 1,
  ): Promise<Record<string, number>> {
    return this.#call<Record<string, number>>("candidateSoftmax", {
      input: { logits, temperature },
    });
  }
  async compileSchema(schema: Record<string, unknown>): Promise<CompiledDecision[]> {
    return this.#call<CompiledDecision[]>("compileSchema", { schema });
  }
  async generate(
    request: GenerateRequest,
    options: { timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<GenerateResponse> {
    return this.#call<GenerateResponse>("generate", { request }, options);
  }
  async evaluateCompatible(profile: string, request: Record<string, unknown>) {
    return this.#call("compatible", { profile, request });
  }
  async info() {
    return this.#call("info", {});
  }
  async close() {
    if (this.#closed || this.#closing) return;
    this.#closing = true;
    try {
      await this.#call("close", {});
    } finally {
      this.#closed = true;
      this.#worker.terminate();
      for (const pending of this.#pending.values()) pending.reject(new Error("ENGINE_CLOSED"));
      this.#pending.clear();
    }
  }
  #call<T = unknown>(
    op: string,
    args: any,
    options: { timeoutMs?: number; signal?: AbortSignal } = {},
  ): Promise<T> {
    if (this.#closed || (this.#closing && op !== "close"))
      return Promise.reject(new Error("ENGINE_CLOSED"));
    if (
      options.timeoutMs !== undefined &&
      (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 0)
    ) {
      return Promise.reject(new Error("Timeout must be a finite non-negative number"));
    }
    if (options.signal?.aborted)
      return Promise.reject(new Error("GENERATION_CANCELLED: generation cancelled"));
    const id = this.#id++;
    return new Promise<T>((resolve, reject) => {
      const cancel = () =>
        this.#worker.postMessage({ op: "cancelGeneration", args: { requestId: id } });
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
