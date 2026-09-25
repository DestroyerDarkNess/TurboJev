/* tslint:disable */
/* eslint-disable */
export function turbojev_wasm_contract(): string;
/**
 * High-level browser engine. The runtime object must already be initialized.
 */
export class WasmEngine {
  free(): void;
  constructor(runtime: any, descriptor_json: string, prompt_profile_json?: string | null);
  evaluate_json(input: string): Promise<string>;
  classify_json(state: string, options_json: string): Promise<string>;
  evaluate_compatible_json(profile: string, input: string): Promise<string>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmengine_free: (a: number, b: number) => void;
  readonly wasmengine_new: (a: any, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly wasmengine_evaluate_json: (a: number, b: number, c: number) => any;
  readonly wasmengine_classify_json: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly wasmengine_evaluate_compatible_json: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly turbojev_wasm_contract: () => [number, number];
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly closure61_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure195_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
