export class TurboJevError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "TurboJevError";
  }
}
export function normalizeError(error: unknown): TurboJevError {
  const text = error instanceof Error ? error.message : String(error);
  const m = /^([A-Z_]+):\s*(.*)$/.exec(text);
  return new TurboJevError(m?.[1] ?? "NATIVE_ERROR", m?.[2] ?? text, error);
}
