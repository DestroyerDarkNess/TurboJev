import type { EvaluationRequest, EvaluationResponse, CompiledDecision, GenerateRequest, GenerateResponse, LoadOptions } from "./types.js";
export * from "./types.js";
export { TurboJevError } from "./errors.js";
export declare class TurboJev {
    #private;
    private readonly nativePath?;
    private constructor();
    static load(options: LoadOptions): Promise<TurboJev>;
    evaluate(request: EvaluationRequest): Promise<EvaluationResponse>;
    candidateSoftmax(logits: Record<string, number>, temperature?: number): Promise<Record<string, number>>;
    compileSchema(schema: Record<string, unknown>): Promise<CompiledDecision[]>;
    generate(request: GenerateRequest, options?: {
        timeoutMs?: number;
        signal?: AbortSignal;
    }): Promise<GenerateResponse>;
    evaluateCompatible(profile: string, request: EvaluationRequest): Promise<Record<string, unknown>>;
    info(): Promise<Record<string, unknown>>;
    close(): Promise<void>;
}
/** Compile using the native Rust compiler without loading a model. */
export declare function compileJsonSchema(schema: Record<string, unknown>, options?: {
    nativePath?: string;
}): CompiledDecision[];
export declare function candidateSoftmax(logits: Record<string, number>, temperature?: number, options?: {
    nativePath?: string;
}): Record<string, number>;
