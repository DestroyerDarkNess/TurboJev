<div align="center">

# TurboJev

### Typed decisions from local open-weight models

**Turn text, structured state, images, audio, or video into a typed decision with auditable probabilities.**

[Website and docs](website/) · [Samples](samples/README.md) · [TurboJev skill](.agents/skills/turbojev/SKILL.md) · [Decision design](docs/DECISION_DESIGN.md) · [Architecture](docs/ARCHITECTURE.md) · [Multimodal input](docs/MULTIMODAL.md) · [Order robustness](docs/ORDER_ROBUSTNESS.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/core-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Local inference](https://img.shields.io/badge/inference-local-2563eb.svg)](docs/PORTABILITY.md)
[![GitHub Actions](https://github.com/DestroyerDarkNess/TurboJev/actions/workflows/ci.yml/badge.svg)](https://github.com/DestroyerDarkNess/TurboJev/actions/workflows/ci.yml)

</div>

```text
state + typed questions → candidate scores → probabilities → typed answers
```

TurboJev is a Rust-first decision layer for applications that need a model to
choose, route, assess, or approve. It scores the allowed candidates directly
and returns structured `choice`, `noul`, and `score` answers. The optimized
path does not ask a model to generate JSON and does not rely on parsing a
natural-language reply.

It runs locally with open-weight models, keeps inference and decision semantics
separate, and exposes the same contract through Rust, Node/TypeScript, Python,
C, .NET, and browser/WASM hosts.

For extraction and dependent workflows, `generate` also supports JSON-typed
strings, integers, decimals, booleans, nullable/described enums, nested objects,
bounded arrays, ordinal scales, confidence gates and dependencies,
per-field reasoning and optional probability averaging across option orders.
Unrestricted values use bounded model generation; finite decisions keep direct
candidate scoring. See [typed fields](docs/TYPED_FIELDS.md) and the
[real-model evals](evals/README.md) for the contract and runtime limits.
The [cookbook](docs/COOKBOOK.md) covers continuation, hierarchical routing and
category discovery; the [TypeLLM 0.6.9 port matrix](docs/TYPELLM_0_6_9_PORT.md) records
feature equivalents and validation.

## Get a real model running

The fastest path is one of the runnable samples. It builds the native libraries,
downloads a small verified GGUF model, then runs the same local model through
four language bindings.

```powershell
py samples/build.py native --download-model
py samples/run.py ts
py samples/run.py python
py samples/run.py csharp
py samples/run.py rust
```

On Linux or macOS, use `python` in place of `py`. See the complete
[samples guide](samples/README.md), including architecture selection and the
browser-only JavaScript example.

## Ask a typed question

```python
from turbojev import TurboJev

with TurboJev.load("./model.gguf") as jev:
    result = jev.evaluate({
        "state": {"message": "My package arrived broken."},
        "questions": {"department": {"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": "Payments", "shipping": "Delivery", "technical": "Product issue", "other": "Other"}}},
    })["answers"]["department"]

print(result["choice"])
print(result["probabilities"])
```

Native runtimes use CPU by default. To request GPU offload, set `device="gpu"`
in Python or `device: "gpu"` in Node/.NET load options. The Windows x64 native
package includes Vulkan support; x86 and ARM64 are CPU-only. A GPU request
fails with the runtime error if no supported GPU backend or device is available.

Use `evaluate` when one state needs several typed decisions at once:

```json
{
  "state": { "message": "My package arrived broken.", "customerTier": "plus" },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoices, or refunds",
        "shipping": "Delivery, damage, or replacement",
        "technical": "Product or application problem"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this need urgent human intervention?"
    },
    "resolution": {
      "type": "score",
      "instructions": "How complete is the proposed resolution?",
      "criteria": ["Not started", "Partly resolved", "Resolved"]
    }
  }
}
```

The response preserves each question's type, candidate distribution, selected
answer, zero-generation usage, execution path, and calibration provenance.

## Why TurboJev

| Need                        | TurboJev approach                                                              |
| --------------------------- | ------------------------------------------------------------------------------ |
| Route a request             | `choice` returns a stable named option and its full distribution.              |
| Gate an action              | `noul` represents a typed yes/no decision without string parsing.              |
| Evaluate a state            | `score` returns an ordered, expected-value assessment.                         |
| Avoid output parsing        | Backends score candidates; TurboJev owns validation and response construction. |
| Keep data local             | Local inference is supported with no mandatory telemetry or service.           |
| Use one contract everywhere | Native bindings and browser/WASM share the same decision schema.               |
| Bring a new model           | Implement a small scoring adapter instead of rewriting decision logic.         |

### Stable option ordering when order is irrelevant

Many models favor the first or last presented option. For unordered `choice`
questions, opt into balanced order robustness:

```json
{
  "model": "./model.gguf",
  "execution": { "orderRobustness": "balanced" }
}
```

TurboJev evaluates canonical layouts that balance option positions and neighbours,
calibrates each layout once, and averages probabilities by original key. With K
choices this costs K evaluations for even K, or 2K for odd K. `none` remains the
single-pass default; `cyclic` retains its original cyclic logit averaging. The response
records both the active policy and the actual number of model evaluations.
The feature applies to choice questions in `evaluate`; it deliberately leaves
ordered scores and boolean decisions unchanged. Read the [full contract and
cost model](docs/ORDER_ROBUSTNESS.md).

### Calibrated confidence is explicit

A softmax distribution is useful for ranking but is not automatically a
calibrated probability. TurboJev keeps calibration separate from inference and
records whether a fitted calibration profile was used. This lets applications
set thresholds based on evidence instead of treating every high model score as
a guarantee. See [calibration tooling](docs/CALIBRATION.md).

## Text and multimodal models

Text models can use the built-in llama.cpp/GGUF route or a custom runtime
adapter. For models that accept media, TurboJev carries a model-independent
envelope for text, image, audio, and video evidence. The selected runtime owns
the processor and candidate scoring; TurboJev still validates the input and
produces the same typed response.

The repository includes local CPU examples for:

- [SmolVLM2](samples/TurboJev-SmolVLM2/README.md) image and short-video decisions.
- [LFM2.5-Audio](samples/TurboJev-LFM-Audio/README.md) spoken-intent decisions.

The generic protocol does not claim that every GGUF, ONNX, or safetensors model
is multimodal. Runtime support is declared by the adapter and documented in
[multimodal compatibility](docs/MULTIMODAL.md).

## Pick your host

| Host                 | Starting point                                  | Runtime path                |
| -------------------- | ----------------------------------------------- | --------------------------- |
| Rust                 | [Rust sample](samples/TurboJev-rust)            | Rust crates + llama.cpp     |
| Node / TypeScript    | [Node sample](samples/TurboJev-TS-node)         | Node-API + llama.cpp        |
| Python               | [Python sample](samples/TurboJev-Python)        | PyO3 + llama.cpp or worker  |
| C / .NET             | [C# sample](samples/TurboJev-csharp-NET)        | Stable C ABI + P/Invoke     |
| Browser              | [Vanilla JS sample](samples/TurboJev-js-native) | Rust/WASM + Transformers.js |
| Custom model runtime | [Adapter contract](docs/ARCHITECTURE.md)        | Candidate-scoring backend   |

## Built for evidence, not opaque claims

The Windows validation run used a real small GGUF model with complete response
parity across Node/TypeScript, Python, C, and .NET, including typed answers and
the zero-generation invariant. The multimodal SmolVLM2 CPU sample also has a
recorded real image/video validation run. Reproduce the commands and inspect the
limits in [Windows validation](docs/WINDOWS_VALIDATION.md) and [multimodal
validation](docs/MULTIMODAL_VALIDATION.md).

TurboJev keeps an independent reference path and treats runtime optimizations as
capabilities that require parity evidence. It does not advertise cross-request
fusion merely because a runtime can batch tokens.

## Design commitments

- **Portable core:** semantic logic has no operating-system, filesystem,
  network, hardware, tokenizer, or model dependency.
- **Truthful runtime boundaries:** adapters use the real model template and
  tokenizer; they never invent candidate token IDs.
- **Local by default:** no mandatory hosted service and no mandatory telemetry.
- **Typed output:** model backends return scores; TurboJev validates, normalizes,
  calibrates, and builds one canonical response.
- **Open integration:** custom runtimes and model families plug into the scoring
  contract without forking the decision engine.

## Documentation

- [Language guides and API reference](website/docs/quickstart.html)
- [Typed fields, dependencies and conditions](docs/TYPED_FIELDS.md)
- [Self-contained TurboJev skill](.agents/skills/turbojev/SKILL.md)
- [Decision design](docs/DECISION_DESIGN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Samples and build commands](samples/README.md)
- [Multimodal input contract](docs/MULTIMODAL.md)
- [Candidate-order robustness](docs/ORDER_ROBUSTNESS.md)
- [Calibration tooling](docs/CALIBRATION.md)
- [Runtime qualification](docs/RUNTIME_QUALIFICATION.md)
- [Testing and validation](docs/TESTING.md)
- [Release and package publishing](docs/RELEASING.md)

## License

TurboJev is available under the [Apache License 2.0](LICENSE), which permits
commercial use and redistribution subject to its terms.
