# TurboJev Project Context

Summary saved at the user's request on 2026-09-21. Based on README.md, SPEC.md,
Cargo.toml, and CHANGELOG.md; this is not a code audit or evidence that every
test has passed. Revalidate the current state when making changes.

## Purpose

TurboJev is a cross-platform library, primarily written in Rust, for making
typed semantic decisions with open-weight models. Its flow is:
state + typed questions -> candidate scores -> probabilities -> typed answers.
It supports classification, routing, and state evaluation. The optimized path
reads candidate logits without generating model text or JSON.

Core types are `choice` (selection among options), `noul` (a boolean decision;
`boolean` is an input alias), and `score` (evaluation with ordered criteria).
The public API uses `evaluate` for typed choices, boolean decisions, scores, and media.

## Architecture and principles

- Semantic logic, validation, planning, calibration, and response construction
  live exactly once in small, separate Rust crates.
- The portable core is independent of the model, operating system, hardware,
  runtime, language, prompt template, and async executor. It does not access
  files, networks, processes, or the environment directly; hosts and adapters own
  those concerns.
- Adapters use the model's real template and tokenizer. They never guess or fix
  candidate IDs for a model family.
- Thin bindings for Node/TypeScript, browser/WASM, Python, C, .NET, Android,
  and Swift/iOS share Rust and JSON contracts. Optional integrations include
  llama.cpp/GGUF, a Transformers.js web adapter, and custom runtimes.
- Optimizations include prefix reuse, parallel suffixes, selective candidate
  projection, and request fusion when the runtime supports and demonstrates them.
  Keep an independent reference path and verify numerical parity.
- Conditional softmax is not calibrated confidence. Calibration and provenance
  are explicit. Do not advertise capabilities that a runtime does not implement.
- Local inference and no mandatory telemetry are project principles.
- The project is inspired by ideas from SemIf and Nimble; it is not a fork.
  Jev compatibility concerns contracts and semantics, without claiming equality
  with private weights or calibration.
- The project includes contracts, datasets, artifacts, and reproducible evaluation
  for specialized models; LoRA/QLoRA training happens outside TurboJev.

## Observed state

Cargo.toml and CHANGELOG.md report v0.28.2. The README v0.20 heading is stale
relative to its later sections and the manifest.
Recent work focuses on reproducible validation with real models and complete
response parity across Node, Python, C, and .NET, plus llama.cpp multi-root
qualification with evidence bound to the model, runtime, and hardware.
v0.28.2 fixes .NET affinity-thread release on failure and uses the prebuilt
Release binary in E2E. Do not assume 1.0 readiness without the real report:
fixture or stub tests do not replace evidence from a real model.

## Orientation

- README.md: product, API, and integrations.
- SPEC.md: portable contract; its own version is distinct from the workspace version.
- CHANGELOG.md and Cargo.toml: workspace changes and version.
- crates/: core, adapters, hosts, and tools.
- bindings/: packages for other languages and platforms.
- docs/, schemas/, tests/, scripts/, and benchmarks/: contracts, documentation,
  and validation.

For structural code questions, follow the CodeGraph instructions in the
environment and prefer its tools when available.

## Real validation after the previous milestone (2026-09-21)

SmolLM2-135M-Instruct Q2_K ran on Windows with complete parity across Node,
Python, C, and .NET; 108 Rust tests and real robustness checks passed.
Compilation failures, llama.cpp token counters, Windows paths, and worker
shutdown were fixed. See docs/WINDOWS_VALIDATION.md for results, limitations,
and reproduction. Strict Clippy remains pending. Multi-root fusion did not meet
the numerical tolerance and must remain disabled in this configuration.
This does not constitute Linux gate approval or automatic 1.0 readiness.
