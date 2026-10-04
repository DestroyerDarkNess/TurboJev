export type PlanningPolicy = "auto" | "reference" | "deterministic-reuse";
export type MediaModality = "image" | "audio" | "video";
export interface VideoFrame { timestamp_ms: number; base64: string }
export type MediaEvidence =
  | { contract: "turbojev.media.v1"; text?: string; modality: "image" | "audio"; base64: string }
  | { contract: "turbojev.media.v1"; text?: string; modality: "video"; frames: VideoFrame[] };
export interface LoadOptions {
  model: string;
  /** Generate bounded model reasoning before scoring; defaults to false. Native text only. */
  thinking?: boolean;
  /** Reasoning budget, default 512. Increase nCtx when increasing this limit. */
  maxThinkingTokens?: number;
  /** Optional llama.cpp multimodal projector GGUF paired with `model`. */
  mmproj?: string;
  /** Defaults to CPU. GPU requires a native package built with a supported GPU backend. */
  device?: "cpu" | "gpu";
  worker?: { program: string; args?: string[]; timeoutSeconds?: number };
  cacheDir?: string;
  execution?: { policy?: PlanningPolicy; orderRobustness?: "none" | "cyclic"; maxParallelTasks?: number; memoryBudgetBytes?: number; llamacppMultiRootFused?: boolean };
  prompt?: { profile?: string; systemInstruction?: string; bestMatchInstruction?: string };
  nCtx?: number; nBatch?: number; nSeqMax?: number; nGpuLayers?: number; nThreads?: number; nThreadsBatch?: number;
  fixtureLogits?: Record<string, number[]>;
  nativePath?: string;
}
export type EvaluationRequest = Record<string, unknown>;
export type EvaluationResponse = Record<string, unknown>;
