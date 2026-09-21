# Cross-language real-model end-to-end matrix

v0.28 is the final pre-1.0 release gate. One exact open-weight GGUF artifact and one canonical `EvaluationRequest` are executed through every mandatory native host binding that is expected to share the same Rust semantics.

The required matrix is Node/TypeScript (`@turbojev/node` + Node-API addon), Python (`turbojev` + PyO3 extension), C (the stable `turbojev.h` / `turbojev_capi` ABI), and C#/.NET (P/Invoke over that same C ABI). Swift remains an optional real-execution row because it requires an Apple host; its package still builds in normal release gates.

`tests/e2e/request.json` deliberately exercises all three public answer families in one request: `choice`, boolean/`noul`, and ordered `score`. No binding is allowed to replace the canonical Rust semantics with host-language logic.

`scripts/e2e_language_matrix.py` accepts the exact model artifact, native-library directory and Node addon. Every runner emits the complete canonical `EvaluationResponse`. The harness validates typed answer shapes, finite normalized probabilities, the zero-generation invariant and TurboJev metadata, then recursively compares the complete responses against the Node/TypeScript reference using explicit absolute/relative floating-point tolerances.

The emitted contract is `turbojev.e2e-language-report.v2`. It records model/request SHA-256, the reference language, tolerances, semantic-parity status, bounded mismatch diagnostics, per-language process status/duration/stdout hash and canonical response SHA-256. The mandatory matrix passes only when Node/TypeScript, Python, C and .NET/C# all execute successfully **and** their canonical responses are semantically equivalent.

The matrix is a behavioral/conformance test, not an intelligence benchmark. A tiny model may choose any schema-valid option. The release gate proves real model loading, tokenizer/chat-template boundary handling, candidate scoring, calibration, typed-response construction, ABI/binding correctness and cross-language semantic parity.

## Real llama.cpp qualification

`crates/turbojev-runtime-llamacpp/examples/real_fused_qualification.rs` is the evidence producer for the multi-root optimization. It opens the real GGUF through the pinned llama.cpp build, discovers safe candidate codes using the checkpoint tokenizer/chat template, scores independent roots, executes the same roots through the multi-root path, compares logits and argmax, then re-runs independent scoring after fusion to detect KV/sequence-state contamination.

That raw evidence is converted into the versioned runtime-qualification manifest. A passing language matrix does not by itself authorize fused execution, and a passing fixture/stub test does not substitute for real target qualification.

## One-command final gate

On the supported Linux release host with network access and the pinned toolchains installed:

```sh
python scripts/final_e2e_linux.py \
  --work-dir .e2e-final \
  --report artifacts/final-validation-report.json
```

The script verifies exact Rust/Node/Python/.NET toolchain versions, the GGUF SHA-256 and exact byte size, pins/checks the llama.cpp revision, freezes Cargo/npm dependency resolution for the run, builds the TurboJev shim and native host artifacts, runs runtime qualification, executes the mandatory language matrix and writes `turbojev.final-validation-report.v2`. `.github/workflows/real-e2e.yml` performs the same release gate reproducibly in CI.
