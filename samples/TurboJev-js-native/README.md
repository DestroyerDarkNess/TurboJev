# TurboJev-js-native

HTML + vanilla JavaScript, Rust/WASM and a browser-local ONNX model.

From the repository root:

```sh
python samples/build.py web
python samples/run.py web
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
