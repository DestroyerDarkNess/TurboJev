# TurboJev samples

Small projects using compiled TurboJev libraries and local inference.

| Project | Interface | Engine |
|---|---|---|
| TurboJev-js-native | HTML + vanilla JS decision studio | Rust/WASM + Transformers.js (ONNX) or llama.cpp WASM (GGUF) |
| TurboJev-TS-node | Typed async console application, plus image/video decisions | Node-API + llama.cpp or SmolVLM2 worker |
| TurboJev-Python | Context-managed Python console application | PyO3 + llama.cpp |
| TurboJev-csharp-NET | C# `await using` console application, plus image/video decisions | Native C ABI + SmolVLM2 worker |
| TurboJev-rust | Typed Rust classification application | Compiled Rust crates + llama.cpp |
| [TurboJev-SmolVLM2](TurboJev-SmolVLM2/README.md) | Image and short-video decisions | Rust + optional SmolVLM2 CPU worker |
| [TurboJev-LFM-Audio](TurboJev-LFM-Audio/README.md) | Spoken-intent decisions | Rust + optional LFM2.5-Audio CPU worker |

## Native: build once, run four languages

Prerequisites: Python 3.9+, Node 22+, .NET SDK 8+, Rust/rustup, Git, CMake, and a
C/C++ toolchain (Visual Studio C++ tools on Windows, Xcode tools on macOS, GCC/Clang
on Linux). The repository pins Rust 1.82. Run commands from the repository root.

If you are already in `D:\Projects\TurboJev\samples`, use `py` like this:

```powershell
py .\build.py native
py .\run.py ts --model C:\models\model.gguf
py .\run.py python --model C:\models\model.gguf
py .\run.py csharp --model C:\models\model.gguf
py .\run.py rust --model C:\models\model.gguf
```

`native` and `web` are required modes. To select an architecture, write one
option without braces: `--arch x64`, `--arch x86`, or `--arch arm64`. The value
must match the architecture of Windows, Python, Node, and the installed build
tools. For the web example:

```powershell
py .\build.py web
py .\run.py web
```

```sh
python samples/build.py native
python samples/run.py ts --model C:\models\model.gguf
python samples/run.py python --model C:\models\model.gguf
python samples/run.py csharp --model C:\models\model.gguf
python samples/run.py rust --model C:\models\model.gguf
```

## Commands by project

### Multimodal examples

These optional CPU samples compile only the Python binding and run a local
model worker. They need Python 3.12; setup installs its own environment and
downloads the selected model. From the repository root:

```sh
python samples/multimodal/setup.py vision
python samples/run.py vision --file clip.mp4 --question "What is happening?" --options "Walking" "Sitting"
python samples/run.py vision --file photo.jpg --modality image --options "A cat" "A dog"

python samples/multimodal/setup.py audio
python samples/run.py audio --file speech.wav --question "What is requested?" --options "Turn on the light" "Turn off the light"
```

From `samples/`, omit the `samples/` prefix in these commands. See each
example's README for limits, licenses, and platform validation status.

Run all commands from the repository root. The native build prepares the
compiled libraries once for the selected host.

### `TurboJev-TS-node`

```sh
python samples/build.py native
python samples/run.py ts --model C:\models\model.gguf
python samples/run.py ts --model C:\models\model.gguf --text "I was charged twice."
```

After the native build above and preparing the optional vision worker, this
sample can also send an image or short video through the same native TurboJev
session and print the complete typed `EvaluationResponse`:

```sh
python samples/multimodal/setup.py vision
python samples/run.py ts --media photo.jpg --modality image \
  --question "What animal is shown?" --options "A cat" "A dog"
```

Use `--modality video` for a supported short video. The launcher configures the
local SmolVLM2 worker; the TypeScript code only constructs media evidence and
calls `evaluate`.

The sample source is in `samples/TurboJev-TS-node/src/main.ts`. To compile only
the project after preparing the artifacts:

```sh
npm run build --prefix samples/TurboJev-TS-node
```

### `TurboJev-Python`

```sh
python samples/build.py native
python samples/run.py python --model C:\models\model.gguf
python samples/run.py python --model C:\models\model.gguf --text "My order arrived damaged."
```

The launcher configures `PYTHONPATH` and native library paths automatically.
The sample source is in `samples/TurboJev-Python/main.py`.

### `TurboJev-csharp-NET`

```sh
python samples/build.py native
python samples/run.py csharp --model C:\models\model.gguf
python samples/run.py csharp --model C:\models\model.gguf --text "The application crashes on startup."
```

After the native build above, the C# sample has the same image/video path:

```sh
python samples/multimodal/setup.py vision
python samples/run.py csharp --media photo.jpg --modality image \
  --question "What animal is shown?" --options "A cat" "A dog"
```

It loads the same local worker through the stable native C ABI and prints the
canonical JSON response returned by `EvaluateJsonAsync`.

The .NET project is in `samples/TurboJev-csharp-NET/Sample.csproj`. To compile
it directly after the native build:

```sh
dotnet build samples/TurboJev-csharp-NET/Sample.csproj -c Release -p:TurboJevRid=win-x64
```

Use `linux-x64`, `linux-arm64`, `osx-x64`, or another compatible RID on that
platform.

### `TurboJev-rust`

```sh
python samples/build.py native
python samples/run.py rust --model C:\models\model.gguf
python samples/run.py rust --model C:\models\model.gguf --text "Please send a replacement."
```

To check the crate with Cargo:

```sh
cargo check -p turbojev-rust-sample --features native
```

The executable linked with llama.cpp is produced by
`python samples/build.py native`; `samples/run.py rust` runs it with the
prepared library.

### `TurboJev-js-native`

This example runs in the browser with Rust/WASM and lets you select either
Transformers.js for ONNX models or llama.cpp WASM for GGUF models. It does not
use a native DLL. Its commands are:

```sh
python samples/build.py web
python samples/run.py web
```

To work only inside the web project, from `samples/TurboJev-js-native`:

```sh
npm install
npm run prepare:runtime
npm run build
npm run dev
```

Open the local Vite URL, choose the runtime and model, click **Load model**, and
then **Evaluate message**. The default GGUF example uses Qwen3 0.6B Q4_K_M.

The first build fetches pinned llama.cpp; later builds reuse the compilation cache.
Provide your own local GGUF with `--model`. Use `--debug` for faster Rust
development builds, or pass a different model and message:

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
wasm-bindgen CLI, compiles the Rust core to WASM, and prepares both browser
adapters. Select **Transformers.js** with an ONNX model ID or **llama.cpp** with
a direct GGUF URL. CPU is the default; WebGPU can be selected explicitly.
Model files load only after you click **Load model**. Inference runs in the
browser; the page does not upload the input text.

WebAssembly is architecture independent: it does not need x86/x64 DLLs. `npm run
build` in the browser project produces a static `dist/` site. Serve it over HTTP;
opening the HTML with `file://` cannot start the module worker.

## Meaning of the output

The small model demonstrates loading, typed classification and probabilities.
Its predictions can be wrong. Probabilities are conditional on the supplied
options, not calibrated correctness estimates. Multi-root fusion remains disabled.
Each language owns its UI and lifecycle; TurboJev Rust owns decision semantics.
