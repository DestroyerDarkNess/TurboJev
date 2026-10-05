# Typed-field development validation

These are actual model results from **unpublished development builds**, not
claims about registry packages or TypeLLM's private model. The full benchmark
contains every attempt, including wrong answers; gold labels were never given
to the model. See `manifest.json` for the frozen native library, model and
dataset hashes, configuration and resumed segments.

| Suite | Cases | Correct | Accuracy | Execution errors |
|---|---:|---:|---:|---:|
| TypeLLM numeric | 128 | 112 | 87.5% | 0 |
| JevBench | 231 | 125 | 54.1% | 0 |

Model: Qwen3 0.6B Q4_K_M; thinking disabled. This small model still makes
semantic and arithmetic mistakes. The two suites test different tasks; their
combined accuracy should not replace the separate measurements.

**Timing limitation:** runs overlapped native compilation and other machine
activity, with interruptions and resumes. The frozen runtime predates the final
CPU offload guard. `device=cpu` records the requested configuration; these
latencies are not an isolated CPU performance qualification or a comparison
against another engine. Raw measured times remain in the report.

Release source checks on Windows x64 with Rust 1.82 also passed:
`cargo fmt --all -- --check`, workspace Clippy with `-D warnings`, and the
complete workspace test run (147 tests passed, zero failures). The 19 Wllama
adapter regression tests and compiled Rust WASM field-contract gate passed.
Fixture/static checks are contract evidence, not model-quality evidence.

Additional probes:

- `native-fields.json`: real scalar text/numeric/null fields, booleans,
  dependencies, conditions and finite distributions. Six labeled cases matched;
  two distribution cases were not semantic accuracy tests. This earlier probe
  does not contain an exact native binary fingerprint.
- `native-reasoning.json`: actual bounded reasoning, automatic effort selection
  and restoration to reasoning disabled, using Vulkan GPU and the same text
  model. All three expected answers matched. Snapshot DLL hashes are included.
- `wasm-24-options.json`: real Rust WASM plus Wllama 3.8.1 and Qwen3 GGUF under
  the Node CPU/JSPI harness. All 24 option probabilities were returned; the
  selected value was `option_23`, explicitly stated by the input. The harness
  recorded `matched=null` because this case was originally a capacity probe;
  the archived response establishes the selected value directly. This is not
  browser UI or WebGPU qualification.
- `vision-one-field.json`: real Qwen3.5 2B GGUF with its matching F16 projector,
  requesting GPU. The receipt's first item matched `ICE BLACKCOFFE`. It tests
  one image field, not the full 14-field receipt demo or other modalities.

Reproduce the benchmark with the published package once available:

```powershell
python -m pip install -r evals/requirements.txt
python evals/run.py --suite all --model C:/models/Qwen3-0.6B-Q4_K_M.gguf --output evals/results/my-run
```

Reproduce native reasoning with a built C ABI runtime and its bundled DLLs:

```powershell
python scripts/validate-fields-native.py --suite reasoning --device gpu --model C:/models/Qwen3-0.6B-Q4_K_M.gguf --library C:/runtime/turbojev_capi.dll --library-dir C:/runtime --output artifacts/reasoning.json
```

`evals/data/PROVENANCE.json` records dataset sources and licenses. Every JSON
result contains actual requests and responses; none substitutes host-generated
answers for model output.
