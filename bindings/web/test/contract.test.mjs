import test from "node:test";import assert from "node:assert/strict";import { readFile } from "node:fs/promises";
test("worker delegates semantic decisions to Rust/WASM",async()=>{const t=await readFile(new URL("../worker.js",import.meta.url),"utf8");assert.match(t,/WasmEngine/);assert.doesNotMatch(t,/function softmax|Math\.exp/);assert.doesNotMatch(t,/labelBase/);});
