# TurboJev-csharp-NET

C# with await using, the compiled .NET assembly and native C ABI.

From the repository root:

```sh
python samples/build.py native --download-model
python samples/run.py csharp
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
