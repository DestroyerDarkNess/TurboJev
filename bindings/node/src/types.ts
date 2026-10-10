export type PlanningPolicy = "auto" | "reference" | "deterministic-reuse";
export type MediaModality = "image" | "audio" | "video";
export interface VideoFrame {
  timestamp_ms: number;
  base64: string;
}
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
  execution?: {
    policy?: PlanningPolicy;
    orderRobustness?: "none" | "cyclic" | "balanced";
    maxParallelTasks?: number;
    memoryBudgetBytes?: number;
    llamacppMultiRootFused?: boolean;
  };
  prompt?: { profile?: string; systemInstruction?: string; bestMatchInstruction?: string };
  nCtx?: number;
  nBatch?: number;
  nSeqMax?: number;
  nGpuLayers?: number;
  nThreads?: number;
  nThreadsBatch?: number;
  fixtureLogits?: Record<string, number[]>;
  nativePath?: string;
}
export type EvaluationRequest = Record<string, unknown>;
export type EvaluationResponse = Record<string, unknown>;

export type ScalarValue = string | number | boolean | null;
export type JsonValue = ScalarValue | JsonValue[] | { [key: string]: JsonValue };
export interface ComparisonBounds {
  gt?: number;
  gte?: number;
  lt?: number;
  lte?: number;
}
export interface ConditionOperators extends ComparisonBounds {
  in?: ScalarValue[];
  not_in?: ScalarValue[];
  ne?: ScalarValue;
  confidence?: ComparisonBounds;
}
export interface FieldBase {
  instructions?: string;
  description?: string;
  depends_on?: string[];
  when?: Record<string, ScalarValue | ScalarValue[] | ConditionOperators>;
}
export interface ScalarField extends FieldBase {
  type:
    | "string"
    | "integer"
    | "number"
    | "boolean"
    | ["string" | "integer" | "number" | "boolean", "null"]
    | ["null", "string" | "integer" | "number" | "boolean"];
  enum?: ScalarValue[];
  choices?: Array<{ value: ScalarValue; description?: string }>;
  levels?: Array<{ label: string; description?: string }>;
  permutations?: number | "auto" | "all";
  return_probabilities?: boolean;
  thinking?: boolean | "auto";
  thinking_effort?: "none" | "low" | "medium" | "high";
  thinking_budget?: number;
}
export interface ObjectField extends FieldBase {
  type: "object";
  properties: Record<string, ScalarField | ObjectField>;
  required?: string[];
  additionalProperties?: false;
}
export interface ArrayField extends FieldBase {
  type: "array";
  items: ScalarField | ObjectField;
  minItems?: number;
  maxItems?: number;
  continue_from?: JsonValue[];
}
export type FieldSchema = ScalarField | ObjectField | ArrayField;
export interface GenerateRequest {
  context?: unknown;
  state?: unknown;
  images?: string[];
  questions?: Record<string, FieldSchema>;
  schema?: {
    type: "object";
    properties: Record<string, FieldSchema>;
    required?: string[];
    additionalProperties?: false;
  };
  thinking?: boolean | "auto";
  thinking_effort?: "none" | "low" | "medium" | "high";
  thinking_budget?: number;
  text_max_tokens?: number;
  numeric_max_tokens?: number;
  seed?: number;
  temperature?: number;
}
export interface ProbabilityAnswer {
  value: ScalarValue;
  confidence: number;
  probability: number;
  probabilities: Record<string, number>;
  distribution: Array<{ value: ScalarValue; probability: number }>;
}
export interface GenerateResponse {
  result: Record<string, JsonValue | ProbabilityAnswer>;
  thinking: Record<string, string>;
  thinking_effort: Record<string, string>;
  skipped: string[];
  usage: { decisions: number; input_tokens?: number | null; generated_tokens: number };
}
export interface CompiledDecision {
  name: string;
  path: string[];
  question: string;
  output_schema: Record<string, unknown>;
  choices: ScalarValue[] | null;
  descriptions: Array<[ScalarValue, string]>;
  levels: string[];
  depends_on: string[];
  dependency_groups: string[][];
  when: Array<[string, unknown]>;
  permutations: number | "auto" | "all";
  return_probabilities: boolean;
  thinking: boolean | "auto";
  thinking_budget: number | null;
  array: {
    items: CompiledDecision[];
    item_object: boolean;
    minItems: number;
    maxItems: number | null;
    continue_from: JsonValue[];
  } | null;
}
