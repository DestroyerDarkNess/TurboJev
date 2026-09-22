# TurboJev-TS-node

TypeScript with async/await, a dedicated worker and the compiled Node-API addon.

From the repository root:

```sh
python samples/build.py native --download-model
python samples/run.py ts
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
