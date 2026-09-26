# TurboJev-rust

Rust with typed evaluate(), compiled crates and the llama.cpp adapter.

From the repository root:

```sh
python samples/build.py native
python samples/run.py rust --model C:/models/model.gguf
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
