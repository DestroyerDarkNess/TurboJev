import type { GenerateResponse } from "./types.js";
export declare class TurboJevError extends Error {
    readonly code: string;
    readonly cause?: unknown | undefined;
    readonly usage?: GenerateResponse["usage"];
    constructor(code: string, message: string, cause?: unknown | undefined);
}
export declare function normalizeError(error: unknown): TurboJevError;
