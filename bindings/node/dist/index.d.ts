import type { EvaluationRequest, EvaluationResponse, GenerateRequest, GenerateResponse, LoadOptions } from "./types.js";
export * from "./types.js";
export { TurboJevError } from "./errors.js";
export declare class TurboJev {
    #private;
    private constructor();
    static load(options: LoadOptions): Promise<TurboJev>;
    evaluate(request: EvaluationRequest): Promise<EvaluationResponse>;
    generate(request: GenerateRequest): Promise<GenerateResponse>;
    evaluateCompatible(profile: string, request: EvaluationRequest): Promise<Record<string, unknown>>;
    info(): Promise<Record<string, unknown>>;
    close(): Promise<void>;
}
