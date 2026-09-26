import type { LoadOptions } from "./types.js";
export interface NativeEngine {
    evaluateJson(s: string): string;
    evaluateCompatibleJson(p: string, s: string): string;
    infoJson(): string;
    close(): void;
}
export interface NativeModule {
    NativeEngine: {
        open(optionsJson: string): NativeEngine;
    };
}
export declare function loadNative(options: LoadOptions): NativeModule;
