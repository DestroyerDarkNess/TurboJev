# TurboJev-csharp-NET

C# with await using, the compiled .NET assembly and native C ABI.

From the repository root:

```sh
python samples/build.py native
python samples/run.py csharp --model C:/models/model.gguf
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
