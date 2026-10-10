import type { GenerateResponse } from "./types.js";

export class TurboJevError extends Error {
  public readonly usage?: GenerateResponse["usage"];
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown,
  ) {
    const [text, usage] = message.split("\nTURBOJEV_USAGE: ");
    super(text);
    if (usage) this.usage = JSON.parse(usage);
    this.name = "TurboJevError";
  }
}
export function normalizeError(error: unknown): TurboJevError {
  const text = error instanceof Error ? error.message : String(error);
  const m = /^([A-Z_]+):\s*([\s\S]*)$/.exec(text);
  return new TurboJevError(m?.[1] ?? "NATIVE_ERROR", m?.[2] ?? text, error);
}
