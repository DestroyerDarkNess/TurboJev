# Binding compatibility

This audit covers the source after the 0.33.0 release. Changes listed here are
not yet published. A source wrapper, matching native binaries and a compatible
model runtime are separate deployment requirements.

| Binding | Requirement | Validation on 2026-10-11 |
| --- | --- | --- |
| Python | CPython 3.9+, normal GIL-enabled `cp39-abi3` | Corrected x64 wheel passed on 3.9.13, 3.12.7, 3.14.0; published x86 binary passed on 3.12.10; real GGUF suite passed on 3.12.7 |
| Node JS | Node 18+, ESM, Node-API 8 addon | Same addon passed on Node 18.20.8 and 24.18.0 x64; Node 22.18.0 x86 passed; real GGUF suite passed on 18.20.8 |
| TypeScript | ESM-aware compiler; 5.9.2 used for builds | Consumer declarations checked with TypeScript 4.7.4; repository builds passed |
| C | C99 header, matching native ABI | C99/C11/C17 and C++11/C++17 passed on x64; MSVC C11 passed on x86 |
| Rust | Rust 1.82+, edition 2021 | Locked Node/Python/C/JNI crates compiled with Rust 1.82 |
| Browser/WASM | ESM, module workers, WebAssembly; adapter-specific requirements | Web and runtime adapter contract suites passed; this is not an all-browser inference test |
| .NET | Framework 4.8, Core 3.1, .NET 5–10 | See [.NET package-consumer results](DOTNET.md) |
| Android/Kotlin | API 24+, Java 8 bytecode; Kotlin 2.0.21 build compiler | Wrapper compiled against API 35; full Gradle build/device execution not verified |
| Swift | Tools 5.7+, source deployment targets iOS 12 / macOS 10.13 | Source audit; Apple compiler/device validation pending |

The Node native ABI can support older runtimes, but the full wrapper uses Node
18's `fetch` and Web Streams for model downloads. ESM packages can be imported
from CommonJS with dynamic `import()`. TypeScript's build compiler version is
separate from the runtime or native ABI. These facts do not extend a runtime's
vendor support lifecycle.

The real-model suites used Qwen3-0.6B Q4_K_M on CPU, with SHA-256
`9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14`.
They verified evaluation, generated fields, error recovery and engine lifetime;
these are functional checks, not intelligence or latency benchmarks.

## Fixed issues

- The released Python wheel could not import on 3.9 because `benchmark.py`
  evaluated a Python 3.10 union annotation during import. The failure was
  reproduced with the original wheel. Annotations now work both during import
  and through `typing.get_type_hints` on 3.9.
- Node did not declare its minimum wrapper runtime. The next wrapper and native
  manifests declare Node 18+, without tying the addon to Node 22/24.
- Android declared project repositories despite `FAIL_ON_PROJECT_REPOS` in its
  settings. It also bundled a desktop `org.json` dependency already supplied by
  Android and imposed API 26 despite only needing API 24. These were corrected;
  the stale Android module version was aligned with the workspace. A placeholder
  test that only asserted `true` was removed.
- Swift imposed tools 5.9 / macOS 12 / iOS 13 on a wrapper with no corresponding
  API requirement. Its source deployment settings were lowered. Opening and
  closing also read thread-local errors from the caller rather than the native
  worker, and failure could leave the worker running. Those paths were corrected.

## Architecture and publishing checks

Windows packages contain separate x86, x64 and ARM64 binaries. Python wheels,
npm native directories and NuGet runtime directories now verify each PE machine
type, so a mislabeled package fails before upload. Matching architectures do not
establish runtime execution on that architecture. ARM64 was inspected only.

The local publisher bundles inference libraries. The older GitHub release jobs
build a Linux Python artifact and a NuGet artifact without native runtime assets;
new guards reject those incomplete artifacts before upload. Those CI jobs need
complete native artifact assembly before they can publish. The local publisher
remains the supported native release route.

Native Linux/macOS registry packages, Android AARs and Apple XCFrameworks are
not currently published. Android and Swift users must build and package the
matching native runtime themselves. Their wrapper deployment targets do not
lower the OS requirement of a separately built llama.cpp or GPU library.

## Reproduce

After installing the corrected Python wheel, run:

```powershell
python tests/compat/python_sdk.py
```

Repeat with each installed interpreter. The checks cover native loading, schema
compilation, UTF-8, typed results, probabilities, Dice, timeout and annotation
introspection. The x86 run used the existing 0.33.0 extension; x64 runs used a
freshly built wheel containing the current wrapper and bundled inference DLLs.

After building the Node wrapper, pass its entry point and the native addon:

```powershell
node tests/compat/node_sdk.mjs bindings/node/dist/index.js path/to/turbojev.node
```

The C consumer in `tests/compat/c_api.c` links the C ABI library. Include
`bindings/c/include`, link the matching import library and deploy its DLLs.
Compile it as C or C++ with the desired standard; the executable checks native
schema compilation, generation, softmax, deadline errors and ABI access.

Additional checks:

```powershell
python -m unittest discover -s tests/compat -p "test_*.py"
python scripts/check-native-python-wheel.py path/to/package.whl
python scripts/check-native-node-package.py path/to/native-package
python scripts/check-native-nuget.py path/to/package.nupkg
```

Android compilation used Kotlin 2.0.21 with the installed API 35 reference JAR
and Java 8 bytecode. Gradle 9.2 failed before project evaluation with
`java.io.IOException: Unable to establish loopback connection`, including an
IPv4 retry. This does not qualify Android packaging, minimum-device behavior or
JNI execution. There is no Apple toolchain on this Windows validation host.
