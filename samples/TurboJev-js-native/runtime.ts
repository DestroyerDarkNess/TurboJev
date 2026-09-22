import { env } from "@huggingface/transformers";
import { createRuntime as create } from "../../bindings/web-runtimes/transformersjs/src/index";
env.backends.onnx.wasm.wasmPaths = "/vendor/onnx/";
env.backends.onnx.wasm.numThreads = 1;
export const createRuntime = create;
