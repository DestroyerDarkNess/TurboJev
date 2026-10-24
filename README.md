<h1 align="center">TurboJev</h1>

<p align="center">
  <strong>Typed decisions and structured generation with local open-weight models.</strong>
</p>

<p align="center">
  Turn context and questions into decisions, probabilities and structured data.<br>
  Built in Rust, with Python, JavaScript, TypeScript, C#, C and browser bindings,<br>
  plus source integrations for Android and Swift.
</p>

<p align="center">
  Native packages for <strong>Windows · Linux · macOS</strong><br>
  Windows / Linux: x86 · x64 · ARM64 · macOS: Intel · Apple Silicon<br>
  Portable Rust core · Browser WebAssembly · CPU and supported GPU runtimes
</p>

<p align="center">
  <a href="https://pypi.org/project/turbojev/"><img alt="Python package on PyPI" src="https://img.shields.io/badge/PYTHON-PyPI-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="https://www.npmjs.com/package/@turbojev/node"><img alt="JavaScript and TypeScript packages on npm" src="https://img.shields.io/badge/JS%20%2F%20TS-npm-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="https://www.nuget.org/packages/TurboJev"><img alt=".NET package on NuGet" src="https://img.shields.io/badge/.NET-NuGet-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="https://crates.io/crates/turbojev"><img alt="Rust package on crates.io" src="https://img.shields.io/badge/RUST-crates.io-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@turbojev/web"><img alt="Browser and WebAssembly packages" src="https://img.shields.io/badge/BROWSER-WASM-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="https://github.com/DestroyerDarkNess/TurboJev.Demos"><img alt="Runnable demos using published packages" src="https://img.shields.io/badge/DEMOS-multiple%20languages-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="evals/README.md"><img alt="Reproducible evaluation tools" src="https://img.shields.io/badge/EVALUATIONS-reproducible-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/LICENSE-Apache--2.0-30363d?style=for-the-badge&amp;labelColor=161b22" height="28"></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href=".agents/skills/turbojev/SKILL.md"><strong>SDK guide</strong></a> ·
  <a href="docs/COOKBOOK.md"><strong>Cookbook</strong></a> ·
  <a href="samples/README.md"><strong>Samples</strong></a> ·
  <a href="https://github.com/DestroyerDarkNess/TurboJev.Demos"><strong>Demos</strong></a>
</p>

---

TurboJev turns context and questions into typed answers. Use it to route support
tickets, assess urgency, extract invoice fields, or build workflows where one
answer determines the next question. It runs compatible models locally through
llama.cpp or a browser runtime, with no mandatory hosted service or telemetry.

Bindings share the Rust engine's validation, schemas and response contracts.
Choose your language and runtime while keeping the same decision semantics.

| Capability | Use it for |
| --- | --- |
| **Typed decisions** | Select an allowed answer, assess a boolean condition or score ordered criteria. |
| **Structured generation** | Extract text, numbers, nullable values, objects and bounded arrays. |
| **Workflows** | Run dependent questions and include fields only when their conditions hold. |
| **Probability controls** | Inspect distributions, sample answers and configure option-order averaging. |
| **Multimodal input** | Evaluate images, audio and timestamped video frames with a compatible runtime and model. |

## Install

| Language | Registry | Install |
| --- | --- | --- |
| **Python** | [PyPI](https://pypi.org/project/turbojev/) | `python -m pip install --upgrade turbojev` |
| **Node.js / TypeScript** | [npm](https://www.npmjs.com/package/@turbojev/node) | `npm install @turbojev/node` |
| **C# / .NET** | [NuGet](https://www.nuget.org/packages/TurboJev) | `dotnet add package TurboJev` |
| **Rust** | [crates.io](https://crates.io/crates/turbojev) | `cargo add turbojev` |
| **Browser** | [npm / CDN](https://www.npmjs.com/package/@turbojev/web) | `npm install @turbojev/web @turbojev/wasm @turbojev/runtime-llamacpp-wasm` |

Native packages include the model runtime and its library dependencies:

| Platform | Architectures | Requirement |
| --- | --- | --- |
| Windows | x86, x64, ARM64 | Matching process architecture |
| Linux | x86, x64, ARM64 | glibc 2.28+; Alpine/musl needs a source build |
| macOS | x64, ARM64 | macOS 11+; Intel or Apple Silicon |

Python and npm select the native package for the running interpreter. NuGet
contains all eight runtime directories. Browser WebAssembly is architecture
independent. Android and iOS integrations are available as source.

The Rust core does not load models on its own. Add
`turbojev-runtime-llamacpp` with its `ffi` feature and provide the matching native
shim for GGUF inference. C links the C ABI library; Android and Swift need native
libraries built for their targets. The [self-contained SDK guide](.agents/skills/turbojev/SKILL.md)
includes installation and examples for every binding, including browser CDN use.

The NuGet package includes .NET Standard 2.0, .NET Framework 4.8 and .NET 8
assemblies, supporting .NET Core 3.1, .NET 5–7 and newer .NET versions. See
[.NET compatibility](docs/DOTNET.md) for installation and validation details.
See the [binding compatibility matrix](docs/BINDING_COMPATIBILITY.md) for runtime
requirements, architecture coverage and the status of mobile/Apple bindings.

Release verification is recorded in
[the 0.34.0 registry report](docs/validation/published-release-0.34.0.json).
That release is verified on npm, PyPI and crates.io. NuGet 0.34.0 is still
pending; do not assume `dotnet add package TurboJev` installs that version yet.

## Quick start

Download a compatible text GGUF to `model.gguf`, then ask a bounded question:

```python
from turbojev import TurboJev

with TurboJev.load("model.gguf", device="cpu") as client:
    response = client.generate(
        context="My package arrived broken. Please send a replacement.",
        questions={
            "team": {
                "type": "string",
                "instructions": "Which team should handle this request?",
                "choices": [
                    {"value": "shipping", "description": "Delivery damage or replacement"},
                    {"value": "billing", "description": "Payments, charges or refunds"},
                    {"value": "technical", "description": "Software bugs or outages"},
                ],
                "return_probabilities": True,
                "permutations": 1,
            },
            "urgent": {
                "type": "boolean",
                "instructions": "Does this require a reply within the hour?",
            },
        },
    )

print(response["result"]["team"]["value"])
print(response["result"]["team"]["distribution"])
print(response["result"]["urgent"])
print(response["usage"])
```

Finite fields score permitted answers. Open text and numeric fields generate
bounded values. The SDK validates output types; the model remains responsible
for the answer's meaning. Invalid or unsupported outputs raise errors.

## Output types

| Output           | Field definition                                                                        | Returned value                        |
| ---------------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| Text             | `{"type": "string"}`                                                                    | String                                |
| Integer          | `{"type": "integer"}`                                                                   | Integer                               |
| Number           | `{"type": "number"}`                                                                    | Number                                |
| Boolean          | `{"type": "boolean"}`                                                                   | Boolean                               |
| Nullable value   | `{"type": ["string", "null"]}`                                                          | Value or actual null                  |
| Enum             | `{"type": "string", "enum": ["meal", "travel"]}`                                        | One permitted value                   |
| Described choice | `{"type": "string", "choices": [{"value": "billing", "description": "Payments"}, ...]}` | One permitted value                   |
| Ordered score    | `{"type": "number", "levels": [{"label": "low"}, {"label": "high"}]}`                   | Probability-weighted zero-based index |
| Object           | `{"type": "object", "properties": {...}}`                                               | Object                                |
| Array            | `{"type": "array", "items": {...}, "maxItems": 10}`                                     | Bounded array                         |

Numeric enums are unordered choices. Ordered `levels` preserve the original
scale and can return fractional scores. Schemas support nested objects and
top-level arrays of scalars or objects; arrays inside objects and nested arrays
are not supported. This is a defined JSON Schema subset.

Use `questions` for a field map or `schema` for an object schema, never both.
See [typed fields](docs/TYPED_FIELDS.md) for the complete contract.

## Dependencies and conditional fields

Declare `depends_on` when a field needs previous answers. Use `when` to omit a
field unless its condition holds:

```python
with TurboJev.load("model.gguf") as client:
    response = client.generate(
        context="Payments are failing after this morning's deployment.",
        questions={
            "category": {
                "type": "string",
                "enum": ["incident", "billing", "feature_request"],
                "instructions": "What kind of request is this?",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "instructions": "Assess the incident severity.",
                "when": {"category": "incident"},
            },
            "page_on_call": {
                "type": "boolean",
                "instructions": "Should the on-call engineer be paged?",
                "depends_on": ["severity"],
            },
        },
    )

print(response["result"])
print(response["skipped"])
```

Conditions support equality, membership, numeric comparisons and confidence
bounds. Skipped fields are absent from `result` and listed in `skipped`; they
are distinct from an actual null answer. Dependencies are validated for unknown
fields and cycles. Fields currently execute serially in dependency order.

## Probabilities and option ordering

`return_probabilities` adds the selected value, its probability, a confidence
statistic and the full typed distribution. These are conditional scores over
the supplied answers, not automatically calibrated confidence.

`permutations` controls extra model evaluations for unordered fields:

- `1`: one pass in the supplied order.
- `"auto"`: canonical balanced layouts; K passes for even K, 2K for odd K.
- A positive integer: that many distinct sampled layouts, capped by the number available.
- `"all"`: every layout, subject to a 720-evaluation limit.

Small unordered probability fields default to automatic averaging. Large enums
and ordinal levels default to one pass. Averaging costs real inference time and
does not guarantee a correct answer. Ordered scores keep their scale order.

```python
from turbojev import Dice

with TurboJev.load("model.gguf") as client:
    response = client.generate(
        context="A single roll of a fair six-sided die.",
        questions={"roll": Dice(permutations=1).to_schema()},
        temperature=1,
        seed=42,
    )

print(response["result"]["roll"])
```

Positive temperature enables sampling; the default selects the most likely
answer. `Dice` uses the model's distribution and does not force fairness.
See [order robustness](docs/ORDER_ROBUSTNESS.md) and [calibration](docs/CALIBRATION.md).

## Thinking

Thinking is off by default. Native text GGUF inference can generate reasoning
before a field's answer using `thinking=True`, `thinking_budget`, or
`thinking_effort` (`none`, `low`, `medium`, `high`). Set `thinking="auto"` to let
the model select effort as an additional decision. Fields can override the
request policy.

Reasoning adds tokens and latency. It requires a compatible model and enough
context; unfinished reasoning raises an error. Built-in browser runtimes and
native multimodal reasoning currently report unsupported mode errors.
Actual reasoning and selected effort appear in `thinking` and `thinking_effort`.

## Images, audio and video

Use a compatible multimodal model and its matching projector. For GGUF image
extraction:

```python
from pathlib import Path

with TurboJev.load(
    "vision.gguf",
    mmproj="mmproj.gguf",
    device="cpu",
    n_ctx=8192,
) as client:
    response = client.generate(
        context="Read the attached receipt.",
        images=[Path("receipt.png")],
        questions={
            "merchant": {"type": "string", "instructions": "Extract the merchant name."},
            "total": {"type": "number", "instructions": "Extract the printed total."},
        },
    )

print(response["result"])
```

Python accepts image paths, bytes, base64/data URIs and PIL-like images.
Other bindings use the same base64 JSON contract; browser helpers encode
Files/Blobs. Audio uses an audio-capable checkpoint and projector. Video evidence
is an ordered sequence of timestamped image frames, decoded by the application.
TurboJev does not require external ffmpeg/ffprobe executables.

The current OCR demos use Qwen3.5-2B with its vision projector. Text demos use
QJev3.5-0.8B v19c Q8_0; that decision-trained checkpoint is not the OCR model or
a general extraction/reasoning recommendation. See [multimodal input](docs/MULTIMODAL.md)
and [measured validation](docs/MULTIMODAL_VALIDATION.md).

## Direct typed decisions

`evaluate` provides the original Jev-style `choice`, `noul` and ordered `score`
contracts over one state:

```python
with TurboJev.load("model.gguf") as client:
    response = client.evaluate({
        "state": "Everything is down and our client demo starts at noon.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle it?",
                "criteria": {
                    "outage": "Service unavailable",
                    "billing": "Charges and refunds",
                    "feature": "Requests and how-to",
                },
            },
            "urgent": {
                "type": "noul",
                "instructions": "Does this require a reply within the hour?",
            },
            "tone": {
                "type": "score",
                "instructions": "How upset is the customer?",
                "criteria": ["calm", "annoyed", "furious"],
            },
        },
    })

print(response["answers"])
```

Direct scoring generates no answer tokens unless reasoning is enabled.
Its load-time `orderRobustness` policy is separate from `generate.permutations`;
`none` is the single-pass default. Compatibility profiles translate supported
TypeSafe-style and AI SDK wire formats without substituting their models.

## Models and execution

Native GGUF inference uses the bundled llama.cpp runtime. A text checkpoint must
have an architecture, tokenizer and prompt template supported by that build.
Loading a GGUF does not reproduce a specialized model's custom head, scorer or
training prompt automatically.

CPU is the default. Explicit GPU selection reports an error if the requested
backend/device is unavailable. Windows x64 includes Vulkan and macOS includes
Metal. Windows x86/ARM64 and Linux packages use CPU inference. Browser applications can
choose GGUF through llama.cpp/WASM or supported ONNX through Transformers.js,
with explicit WebGPU selection where supported.

Keep models loaded for repeated requests. Prefix reuse and other optimizations
depend on runtime capabilities and numerical qualification; structured generation
does not promise parallel field execution. Timeouts and cancellation are
cooperative between model operations. Failed generation retains known usage.

## Documentation and examples

- [Complete SDK skill](.agents/skills/turbojev/SKILL.md): installation, APIs, configuration and examples for every language, all in one file.
- [Cookbook](docs/COOKBOOK.md): routing, extraction, dependencies and continuation.
- [Samples](samples/README.md): applications built from this repository.
- [Standalone demos](https://github.com/DestroyerDarkNess/TurboJev.Demos): applications using published packages and CDN assets.
- [Decision design](docs/DECISION_DESIGN.md): prompts, option descriptions and honest model-controlled decisions.
- [Architecture](docs/ARCHITECTURE.md): portable Rust core, runtimes and bindings.
- [Evaluations](evals/README.md): reproducible task sets and real-model runs.
- [Testing](docs/TESTING.md): contracts, fixtures and validation limits.

Published assets and actual inference are different checks. See the
[native release matrix](docs/NATIVE_RELEASES.md) for build commands, package
contents and execution evidence. Cross-compilation is not a hardware execution
test, and a package release does not establish accuracy for every model.

## License

[Apache License 2.0](LICENSE).
