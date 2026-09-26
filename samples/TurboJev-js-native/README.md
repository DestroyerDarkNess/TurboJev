# TurboJev-js-native

HTML + vanilla JavaScript, Rust/WASM and browser-local model runtimes.

The setup panel lets you choose **Transformers.js (ONNX)** or **llama.cpp
WASM (GGUF)**. The default GGUF is Qwen3 0.6B Q4_K_M. You can replace the
model field with another ONNX model ID or a direct GGUF URL, respectively.
The text form sends its question as a typed `choice` instruction; the model is
not asked to guess the task from the message and answer list alone.

From the repository root:

```sh
python samples/build.py web
python samples/run.py web
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
