import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import type { LoadOptions } from "./types.js";

export interface NativeEngine { evaluateJson(s:string):string; evaluateCompatibleJson(p:string,s:string):string; classifyJson(s:string,o:string):string; infoJson():string; close():void; }
export interface NativeModule { NativeEngine: { open(optionsJson:string): NativeEngine } }

export function loadNative(options: LoadOptions): NativeModule {
  const require = createRequire(import.meta.url);
  const candidates = [options.nativePath, process.env.TURBOJEV_NATIVE_PATH,
    resolve(process.cwd(), "turbojev.node"),
    `@turbojev/node-${process.platform}-${process.arch}`].filter(Boolean) as string[];
  const failures:string[]=[];
  for (const candidate of candidates) {
    try {
      if (candidate.endsWith(".node") && !existsSync(candidate)) continue;
      return require(candidate) as NativeModule;
    } catch (error) { failures.push(`${candidate}: ${String(error)}`); }
  }
  throw new Error(`RUNTIME_UNAVAILABLE: TurboJev native addon not found. Tried: ${failures.join(" | ")}`);
}
