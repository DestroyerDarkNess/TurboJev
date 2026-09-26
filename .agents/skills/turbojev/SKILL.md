---
name: turbojev
description: Use when installing, integrating, designing decisions for, or troubleshooting TurboJev in Rust, Node.js/TypeScript, Python, C, .NET, browser/WASM, Android, or Swift applications.
---

# TurboJev

Help build applications with TurboJev's typed decision API. Use the repository's current source and docs as the authority: package versions, platform assets, and runtime capabilities change. Do not repeat old README version numbers or claim validation that has not been run.

## First establish the integration

Identify the host language, target OS/architecture, model format, and intended inference runtime. TurboJev has a portable decision core and separate runtime adapters. The core does not load every model by itself. A checkpoint is usable only when the selected runtime supports its architecture, tokenizer, prompt template, and required modality.

## Install by language

Check the current package registry, release assets, target OS, and CPU architecture before promising availability. These packages expose TurboJev's typed-decision API; they do not all include the same model runtime.

| Host | Install | What it provides |
|---|---|---|
| Rust | `cargo add turbojev` | Portable typed-decision API. For GGUF, add `cargo add turbojev-runtime-llamacpp --features ffi`. |
| Node.js / TypeScript | `npm install @turbojev/node` | Worker-backed native binding. |
| Python | `python -m pip install turbojev` | Context-managed native binding. |
| C# / .NET | `dotnet add package TurboJev` | .NET binding over the stable C ABI. |
| C | Build native artifacts, include `bindings/c/include/turbojev.h`, and link the matching C ABI library. | Stable UTF-8 JSON ABI. |
| Browser JavaScript / TypeScript | `npm install @turbojev/web` | Browser host, worker, and WASM integration. |
| Android / Kotlin | Include the Android binding source/module and package matching native libraries. | JNI wrapper over the C ABI. |
| Swift / iOS / macOS | Add the local Swift package at `bindings/swift` and package the matching C ABI library. | Swift wrapper over the C ABI. |

For native samples, build artifacts for the same architecture as the host compiler/interpreter. The samples builder takes one architecture per run (for example, `--arch x64`); it does not mean “all architectures.” C consumers can start with `cargo build -p turbojev-capi --release`; build/link the llama.cpp shim only when GGUF inference is needed.

## Request and decision model

Use `evaluate` for choices, boolean gates (`noul`), ordered scores, and multiple typed questions over one state. Do not use the removed `classify` operation. A normal request has `state` and `questions`; each question specifies `type`, task-specific `instructions`, and, for `choice`, a `criteria` map from stable answer keys to meaningful descriptions.

```json
{
  "state": "A customer says the package arrived broken and asks for a replacement.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices, or refunds",
        "shipping": "Delivery, damage, or replacement",
        "technical": "Product or application problems"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this require immediate human intervention?"
    },
    "quality": {
      "type": "score",
      "instructions": "How complete is the proposed resolution?",
      "criteria": ["Incomplete", "Partly complete", "Complete"]
    }
  }
}
```

Response probabilities are relative candidate scores, not calibrated confidence by default. Only describe them as calibrated when a matching calibration profile was fitted and applied. Keep the model's choice, host policy, and calibration provenance distinct.

## Design decisions models can answer

- State the relevant relationship in plain language when a small model would otherwise need to infer it from raw JSON, coordinates, or arithmetic. Structured state is valid; make its decision-relevant meaning explicit too.
- Keep prompts compact and avoid repeating the same facts in the state, instructions, and criteria. More prompt tokens can increase prefill latency; remove only redundant text, then remeasure decision quality and latency.
- Ask for one bounded decision at a time. Do not expect a small checkpoint to plan a long sequence of actions just because the API accepts a complex state.
- Give candidates meaningful keys and descriptions that expose the distinction being scored. Opaque keys such as `A/B/C` add a mapping the model must infer.
- Filter illegal or unsafe candidates in host code. Let TurboJev score the semantic choice, then use deterministic host rules for hard constraints and ties between equivalent actions.
- Test for order sensitivity by permuting candidates. A model that repeatedly chooses the first candidate is order-biased, even when one fixed ordering appears accurate. `orderRobustness: "cyclic"` evaluates multiple rotations, costs additional model evaluations, and is not a guarantee of correctness. Keep single-pass mode by default unless reproducible tests justify the added latency.
- Validate with the exact model, quantization, runtime, and prompt path intended for deployment. Use reproducible cases and a simple reference policy where possible; report both decision quality and latency.

For sequential tasks such as Snake, expose a one-step decision rather than asking a small model to plan a whole path. Describe the food's direction and distance in the state, describe each move by its resulting progress, filter illegal moves in the host, and resolve equivalent moves deterministically. Test with multiple starting states and permuted candidate order; one successful game is not evidence of general reliability.

## Choose the right runtime

- **Native text inference:** use the llama.cpp adapter for GGUF models it can load. CPU is the normal default. GPU is opt-in; an unsupported GPU request should surface the runtime error, not silently change devices.
- **Node/TypeScript, Python, C, and .NET:** these bindings expose the shared native host. Their package or binary must match the target OS, CPU architecture, and language ABI. Check actual release assets before promising x86, x64, or ARM64 support.
- **Rust:** compose the core crate with the runtime adapter the application selects; adding the core crate alone does not provide GGUF inference.
- **Browser/WASM:** choose a supported browser runtime, such as Transformers.js with ONNX or the llama.cpp WASM path with compatible GGUF. Browser hardware, memory, model-format, tokenizer, and template constraints apply.
- **Android and Swift:** these bindings call native code through JNI/C ABI and need the corresponding host libraries packaged for the app. Do not present them as registry packages unless the current repository/release explicitly provides one.
- **Images, audio, and video:** the media envelope is a transport contract, not an inference model. Select a multimodal checkpoint and runtime that implement preprocessing and candidate scoring for that modality.

Browser runtime choices are separate integrations: Transformers.js for supported ONNX models, llama.cpp WebAssembly for GGUF models supported by that WASM build, or a custom implementation of the browser runtime interface. WebGPU must be requested explicitly; expose CPU/WASM where available and report unsupported-device errors honestly. Browser memory, hardware, model format, tokenizer, and template support can differ from the native host.

TurboJev's media envelope only standardizes how evidence enters `evaluate`; it does not include a vision/audio model, processor, or inference runtime. Verify that both the selected checkpoint and adapter support the modality and task. A GGUF filename alone does not guarantee compatibility.

TurboJev is useful for classification, routing, typed gates, and bounded rubric scoring when an application needs a structured choice from an open-weight model. It is not a general text generator, chat replacement, route planner, or promise that every model/runtime combination works.

## Troubleshooting

1. Reproduce with the same request, model file, quantization, runtime, and device.
2. Separate model-load errors from evaluation errors; preserve the complete native/runtime error and check model architecture, tokenizer/template compatibility, file path, and platform assets.
3. Inspect candidate descriptions, candidate order, and the returned distribution. A high score on one candidate is not a guarantee of a correct semantic answer.
4. Compare one-pass and cyclic order scoring only when order bias is suspected, and include the extra latency in the result.
5. Change the smallest responsible layer: request design for ambiguous context, host code for legality, runtime adapter for execution support, or TurboJev core only for a demonstrated shared semantic defect.

Do not claim all models are supported merely because they use GGUF, or all hardware/architectures are available because a language binding exists. State the tested model, runtime, platform, and limitations.
