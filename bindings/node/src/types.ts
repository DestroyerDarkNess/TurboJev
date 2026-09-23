export type PlanningPolicy = "auto" | "reference" | "deterministic-reuse";
export interface LoadOptions {
  model: string;
  worker?: { program: string; args?: string[]; timeoutSeconds?: number };
  cacheDir?: string;
  execution?: { policy?: PlanningPolicy; orderRobustness?: "none" | "cyclic"; maxParallelTasks?: number; memoryBudgetBytes?: number; llamacppMultiRootFused?: boolean };
  prompt?: { profile?: string; systemInstruction?: string; bestMatchInstruction?: string };
  nCtx?: number; nBatch?: number; nSeqMax?: number; nGpuLayers?: number; nThreads?: number; nThreadsBatch?: number;
  fixtureLogits?: Record<string, number[]>;
  nativePath?: string;
}
export interface ClassificationResult { choice: string; probabilities: Record<string, number>; }
export type EvaluationRequest = Record<string, unknown>;
export type EvaluationResponse = Record<string, unknown>;
