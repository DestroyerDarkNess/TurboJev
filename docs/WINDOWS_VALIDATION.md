# Real Windows validation — 2026-09-21

Validated locally with Rust 1.82.0, Node 24.18.0, Python 3.12.7, .NET SDK
10.0.201 (net8.0 application), MSVC 2022 and the installed CMake.
This is Windows functional evidence, not the pinned Linux release gate or an
intelligence/performance benchmark. Rust native bindings used the debug profile;
llama.cpp and the .NET runner used Release.

Model: `tensorblock/SmolLM2-135M-Instruct-GGUF`, Q2_K, 88,202,816 bytes.
SHA-256: `478cca924101ed67bb2751f597be41532b2f4817ee2ab1b0d7cc379c76e79faf`.
Runtime: llama.cpp tag `b10964`, revision starting `b29c606`, CPU.
Windows builds must use `-DGGML_OPENMP=OFF`: the MSVC OpenMP build crashed on
Node worker teardown. The same test passes using ggml's native thread pool.
The CMake default now disables OpenMP on Windows; existing caches need the flag.

## Results

- Rust formatting and workspace/all-target compilation pass.
- Workspace unit/integration/doc tests: 108 passed, zero failures.
- Native C++ ABI smoke test passes with the corrected llama.cpp stub.
- Real Node/TypeScript, Python, C and .NET responses pass full semantic parity
  at absolute/relative tolerance `1e-7`, with zero generated tokens.
- Python: 17 real-model checks pass (typed answers, repeatability, Unicode,
  object/array state, instruction-like input, invalid input, recovery, lifecycle).
- Node: eight real-model checks pass, including file URLs, concurrent calls,
  failed model loading, invalid options and lifecycle.
- Strict Clippy remains incomplete: after correcting lint-group priorities,
  it stops on the existing `type_complexity` diagnostic in turbojev-core.
  No broader cleanup was undertaken for this functional-validation task.
- Multi-root fusion is **not qualified** for this configuration. Saved evidence
  has two rounds/eight requests, zero changed choices, zero isolation failures,
  zero non-finite values, but maximum absolute logit delta `0.2952880859375`
  and relative delta `0.010238857058871426`, above the `1e-5` limits.
  The generated qualification manifest correctly advertises `fusedBatch=false`.
  Do not relax the thresholds or enable this optimization based on this run.

## Fixes discovered through execution

- Pin compatible transitive dependencies in Cargo.lock (napi-sys,
  napi-derive-backend, unicode-segmentation); keep Cargo.lock under version control.
- Correct runtime error propagation, missing conformance errors, JNI lifetimes,
  an HTTP parser borrow, and missing/incorrect test dependencies and fixtures.
- Set `llama_batch.n_tokens` after every allocated batch is populated; the real
  initializer returns zero used tokens. Make the ABI stub reproduce this behavior.
- Export the shim DLL symbols on Windows and disable MSVC OpenMP by default there.
- Permit N-API's generated unsafe registration glue while retaining `deny` for
  handwritten unsafe code in the Node binding.
- Decode Node file URLs with `fileURLToPath`; stop the worker when loading fails.
- Escape C runner model paths as JSON; use the current Python interpreter and
  MSVC for the Windows cross-language harness.
- Normalize Rust formatting and enforce LF for Rust files. Ignore generated
  build/output directories; retain project context and dependency lockfiles.

## Local evidence and reruns

`artifacts/windows-validation/` contains the language, Python, Node and fusion
reports plus Rust logs. Large model/build caches remain in `.cache/real-validation/`.
These generated directories are ignored by Git. The tests and this document are
reviewable source changes; no commit or push was made by this validation task.

To repeat with the staged binaries, from the repository root in PowerShell:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$work = (Resolve-Path .cache/real-validation).Path
$env:PYTHONPATH = "$work/python"
python scripts/e2e_language_matrix.py --model "$work/SmolLM2-135M-Instruct-Q2_K.gguf" --native-lib-dir "$work/native" --node-addon "$work/native/turbojev.node" --python-root "$work/python" --report "$work/language-report.json"
python tests/e2e/python/real_smoke.py "$work/SmolLM2-135M-Instruct-Q2_K.gguf" --report "$work/smoke-report.json"
node tests/e2e/node/real_smoke.mjs "$work/SmolLM2-135M-Instruct-Q2_K.gguf" "$work/native/turbojev.node"
```

After source changes, rebuild/restage the binaries before rerunning. Build the
shim with the CMake project in `crates/turbojev-runtime-llamacpp/native`, setting
`LLAMA_CPP_SOURCE_DIR` and `GGML_OPENMP=OFF`. Point `TURBOJEV_LLAMA_SHIM_LIB_DIR`
at the staged DLL/import libraries, then build the capi/node/python crates with
their `llamacpp` features. Copy the Node DLL to `turbojev.node`, the Python DLL
to `turbojev/_native.pyd` alongside its dependency DLLs, and the C ABI DLL/import
library to the native stage. Rebuild `bindings/node` and the .NET E2E project.
