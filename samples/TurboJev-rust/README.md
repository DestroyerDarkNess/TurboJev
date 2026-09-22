# TurboJev-rust

Rust with typed classify(), compiled crates and the llama.cpp adapter.

From the repository root:

```sh
python samples/build.py native --download-model
python samples/run.py rust
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
