import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import type { LoadOptions } from "./types.js";

export interface NativeEngine {
  evaluateJson(s: string): string;
  generateJson(s: string): string;
  generateJsonControlled(s: string, controlId: string): string;
  evaluateCompatibleJson(p: string, s: string): string;
  infoJson(): string;
  close(): void;
}
export interface NativeModule {
  NativeGenerationControl: new (timeoutMs?: number) => {
    id: string;
    cancel(): void;
    close(): void;
  };
  NativeEngine: {
    open(optionsJson: string): NativeEngine;
    compileSchemaJson(schemaJson: string): string;
    candidateSoftmaxJson(input: string): string;
  };
}

export function loadNative(options: Pick<LoadOptions, "nativePath"> = {}): NativeModule {
  const require = createRequire(import.meta.url);
  const candidates = [
    options.nativePath,
    process.env.TURBOJEV_NATIVE_PATH,
    resolve(process.cwd(), "turbojev.node"),
    `@turbojev/node-${process.platform}-${process.arch}`,
  ].filter(Boolean) as string[];
  const failures: string[] = [];
  for (const candidate of candidates) {
    try {
      if (candidate.endsWith(".node") && !existsSync(candidate)) continue;
      return require(candidate) as NativeModule;
    } catch (error) {
      failures.push(`${candidate}: ${String(error)}`);
    }
  }
  throw new Error(
    `RUNTIME_UNAVAILABLE: TurboJev native addon not found. Tried: ${failures.join(" | ")}`,
  );
}
