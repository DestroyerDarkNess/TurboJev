# TurboJev samples

Five small projects using compiled TurboJev libraries and real local inference.

| Project | Interface | Engine |
|---|---|---|
| TurboJev-js-native | HTML + vanilla JS decision studio | Rust/WASM + Transformers.js, browser only |
| TurboJev-TS-node | Typed async console application | Node-API + llama.cpp |
| TurboJev-Python | Context-managed Python console application | PyO3 + llama.cpp |
| TurboJev-csharp-NET | C# `await using` console application | Compiled .NET wrapper + native C ABI |
| TurboJev-rust | Typed Rust classification application | Compiled Rust crates + llama.cpp |

## Native: build once, run four languages

Prerequisites: Python 3.9+, Node 22+, .NET SDK 8+, Rust/rustup, Git, CMake, and a
C/C++ toolchain (Visual Studio C++ tools on Windows, Xcode tools on macOS, GCC/Clang
on Linux). The repository pins Rust 1.82. Run commands from the repository root.

```sh
python samples/build.py native --download-model
python samples/run.py ts
python samples/run.py python
python samples/run.py csharp
python samples/run.py rust
```

## Comandos por proyecto

Todos los comandos se ejecutan desde la raíz del repositorio. El comando de
build nativo prepara una sola vez las librerías para el lenguaje elegido.

### `TurboJev-TS-node`

```sh
python samples/build.py native --download-model
python samples/run.py ts
python samples/run.py ts --text "I was charged twice."
```

El código del ejemplo está en `samples/TurboJev-TS-node/src/main.ts`. Para
compilar solamente el proyecto después de preparar los artefactos:

```sh
npm run build --prefix samples/TurboJev-TS-node
```

### `TurboJev-Python`

```sh
python samples/build.py native --download-model
python samples/run.py python
python samples/run.py python --text "My order arrived damaged."
```

El launcher configura `PYTHONPATH` y las librerías nativas automáticamente.
El ejemplo fuente está en `samples/TurboJev-Python/main.py`.

### `TurboJev-csharp-NET`

```sh
python samples/build.py native --download-model
python samples/run.py csharp
python samples/run.py csharp --text "The application crashes on startup."
```

El proyecto .NET está en `samples/TurboJev-csharp-NET/Sample.csproj`. Para
compilarlo directamente, después del build nativo:

```sh
dotnet build samples/TurboJev-csharp-NET/Sample.csproj -c Release -p:TurboJevRid=win-x64
```

Usa `linux-x64`, `linux-arm64`, `osx-x64` u otro RID compatible cuando ejecutes
en esa plataforma.

### `TurboJev-rust`

```sh
python samples/build.py native --download-model
python samples/run.py rust
python samples/run.py rust --text "Please send a replacement."
```

Para comprobar o compilar el crate desde Cargo:

```sh
cargo check -p turbojev-rust-sample --features native
```

El ejecutable enlazado con llama.cpp lo genera `python samples/build.py native`;
`samples/run.py rust` lo ejecuta con la biblioteca correcta ya preparada.

### `TurboJev-js-native`

Este ejemplo corre en el navegador usando Rust/WASM y Transformers.js; no usa
una DLL nativa. Sus comandos son:

```sh
python samples/build.py web
python samples/run.py web
```

Para trabajar solamente dentro del proyecto web, desde
`samples/TurboJev-js-native`:

```sh
npm install
npm run prepare:runtime
npm run build
npm run dev
```

Abre la URL local de Vite, pulsa **Cargar modelo** y después **Evaluar mensaje**.

The first build fetches pinned llama.cpp and optionally the SHA-256-verified 88 MB
SmolLM2 Q2_K model. Later builds reuse downloads and compilation caches. Use
`--debug` for faster Rust development builds. Use another model or message:

```sh
python samples/run.py python --model /path/model.gguf --text "I was charged twice."
```

Artifacts are placed in `samples/.artifacts/<rid>/`, with native shared libraries,
a Node package, a Python package, the managed .NET assembly and the Rust executable.
The launcher sets library paths automatically. No `pip install turbojev` is needed:
the Python example imports the package built from this checkout.

Supported build mappings: `win-x86`, `win-x64`, `linux-x86`, `linux-x64`,
`linux-arm64`, `osx-x64`, `osx-arm64`. **Only the current host architecture is built**.
Use matching Node, Python and compilers on each target host; `--arch x86`/`x64`/
`arm64` verifies the selection. This does not manufacture other-platform binaries
from Windows. Linux x86 needs the 32-bit compiler/runtime development libraries.
These mappings are build recipes, not a claim that every platform has been tested.

## Browser: HTML + vanilla JavaScript

```sh
python samples/build.py web
python samples/run.py web
```

Open the localhost URL printed by Vite. The build installs the pinned local
wasm-bindgen CLI, compiles the Rust core to WASM, and bundles the browser runtime.
Click **Cargar modelo**, then **Evaluar mensaje**. CPU is the compatible default;
WebGPU is optional. Model files are downloaded by the browser only after clicking.
Use an ONNX-compatible model ID, not a GGUF file. Native and browser models use
different formats. The page has no inference server and does not upload the input
text. Internet access is needed for the initial model/runtime installation.

WebAssembly is architecture independent: it does not need x86/x64 DLLs. `npm run
build` in the browser project produces a static `dist/` site. Serve it over HTTP;
opening the HTML with `file://` cannot start the module worker.

## Meaning of the output

The small model demonstrates loading, typed classification and probabilities.
Its predictions can be wrong. Probabilities are conditional on the supplied
options, not calibrated correctness estimates. Multi-root fusion remains disabled.
Each language owns its UI and lifecycle; TurboJev Rust owns decision semantics.
