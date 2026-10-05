export type PlanningPolicy = "auto" | "reference" | "deterministic-reuse";
export type MediaModality = "image" | "audio" | "video";
export interface VideoFrame {
    timestamp_ms: number;
    base64: string;
}
export type MediaEvidence = {
    contract: "turbojev.media.v1";
    text?: string;
    modality: "image" | "audio";
    base64: string;
} | {
    contract: "turbojev.media.v1";
    text?: string;
    modality: "video";
    frames: VideoFrame[];
};
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
    worker?: {
        program: string;
        args?: string[];
        timeoutSeconds?: number;
    };
    cacheDir?: string;
    execution?: {
        policy?: PlanningPolicy;
        orderRobustness?: "none" | "cyclic";
        maxParallelTasks?: number;
        memoryBudgetBytes?: number;
        llamacppMultiRootFused?: boolean;
    };
    prompt?: {
        profile?: string;
        systemInstruction?: string;
        bestMatchInstruction?: string;
    };
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
export interface FieldSchema {
    type: "string" | "integer" | "number" | "boolean" | ["string" | "integer" | "number" | "boolean", "null"];
    enum?: ScalarValue[];
    instructions?: string;
    description?: string;
    depends_on?: string[];
    when?: Record<string, ScalarValue | ScalarValue[] | {
        in?: ScalarValue[];
        not_in?: ScalarValue[];
        ne?: ScalarValue;
        gt?: number;
        gte?: number;
        lt?: number;
        lte?: number;
    }>;
    permutations?: number | "auto" | "all";
    return_probabilities?: boolean;
    thinking?: boolean | "auto";
    thinking_effort?: "none" | "low" | "medium" | "high";
    thinking_budget?: number;
}
export interface GenerateRequest {
    context: unknown;
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
}
export interface GenerateResponse {
    result: Record<string, ScalarValue | {
        value: ScalarValue;
        confidence: number;
        probabilities: Record<string, number>;
        distribution: Array<{
            value: ScalarValue;
            probability: number;
        }>;
    }>;
    thinking: Record<string, string>;
    thinking_effort: Record<string, string>;
    skipped: string[];
    usage: {
        decisions: number;
        input_tokens?: number;
        generated_tokens: number;
    };
}
