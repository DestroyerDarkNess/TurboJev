export { TurboJevRemote, mediaEvidence } from "./remote.js";
export type { MediaEvidence, MediaModality } from "./remote.js";
export interface BrowserRuntimeDescriptor {
    name: string;
    model: string;
    codecProfile?: string;
    scoringMode?: "tokens" | "decisions";
    modalities?: Array<"text" | "image" | "audio" | "video">;
    promptProfile?: {
        profile: string;
        systemInstruction: string;
        bestMatchInstruction: string;
    };
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
    close(): Promise<void> | void;
}
/** Model-independent scoring; the provider owns media decoding and preprocessing. */
export interface BrowserDecisionRuntime {
    initialize(options: unknown): Promise<BrowserRuntimeDescriptor & {
        scoringMode: "decisions";
        modalities: Array<"text" | "image" | "audio" | "video">;
    }>;
    scoreDecisionBatch(batchJson: string, planJson: string): Promise<string> | string;
    close(): Promise<void> | void;
}
export interface WebLoadOptions {
    model: string;
    runtimeModuleUrl: string;
    wasmModuleUrl: string;
    runtimeOptions?: unknown;
    prompt?: {
        profile?: string;
        systemInstruction?: string;
        bestMatchInstruction?: string;
    };
    workerUrl?: string;
}
export declare class TurboJevWeb {
    #private;
    private constructor();
    static load(options: WebLoadOptions): Promise<TurboJevWeb>;
    classify(state: string, options: string[]): Promise<unknown>;
    evaluate(request: Record<string, unknown>): Promise<unknown>;
    evaluateCompatible(profile: string, request: Record<string, unknown>): Promise<unknown>;
    info(): Promise<unknown>;
    close(): Promise<void>;
}
