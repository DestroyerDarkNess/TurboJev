# TurboJev-Python

Python with a context manager and the locally compiled PyO3 package.

From the repository root:

```sh
python samples/build.py native --download-model
python samples/run.py python
```

See [shared setup and architecture matrix](../README.md). Native binaries are
selected automatically for this host. The web example uses WASM instead of DLLs.
