export declare class TurboJevError extends Error {
    readonly code: string;
    readonly cause?: unknown | undefined;
    constructor(code: string, message: string, cause?: unknown | undefined);
}
export declare function normalizeError(error: unknown): TurboJevError;
