export class TurboJevError extends Error {
    code;
    cause;
    constructor(code, message, cause) {
        super(message);
        this.code = code;
        this.cause = cause;
        this.name = "TurboJevError";
    }
}
export function normalizeError(error) {
    const text = error instanceof Error ? error.message : String(error);
    const m = /^([A-Z_]+):\s*(.*)$/.exec(text);
    return new TurboJevError(m?.[1] ?? "NATIVE_ERROR", m?.[2] ?? text, error);
}
