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
export declare function loadNative(options?: Pick<LoadOptions, "nativePath">): NativeModule;
