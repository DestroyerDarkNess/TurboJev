# TurboJev 0.31.0 validation

## Option ordering with a real GGUF

[Unmodified responses](real-option-order.json) record Qwen3-0.6B Q4_K_M on
Windows x64 CPU, four threads, thinking disabled, through the release Node
addon. Three support requests were evaluated with three different candidate
orders under each policy. The model saw the complete option set and descriptions.
The host neither selected nor replaced answers.

| Policy | Correct choices | Largest keyed probability change after reordering |
| --- | --- | --- |
| `none` | 9/9 | 0.1372772363 |
| `balanced` | 9/9 | 0 |

Balanced used four model evaluations per four-way choice. These three cases
demonstrate real-runtime stability, not general reasoning accuracy or a Snake
benchmark. Both modes selected the right answer on this small sample. Recorded
latencies overlap publication work and are unsuitable for an isolated speed
comparison. Conditional probabilities are not calibrated confidence.

Reproduce after building the Node release addon:

```powershell
node scripts/validate-real-option-order.mjs C:/path/to/Qwen3-0.6B-Q4_K_M.gguf
```

The report records GGUF and addon SHA-256 hashes. Pass a matching native addon
path and output JSON path as the second and third arguments if needed.

## Regression gates

49 Rust tests passed across the core, native host, training-data, specialized
artifact, conformance and compatibility crates. The new fixture exercises all
24 input orders of a four-way set-interaction task, sync/async parity,
per-layout calibration, stable exact ties and preserved ordinal scores.
Strict Clippy passed for core, host-native, training-data and WASM; formatting,
Node and web wrapper tests and the real WASM contract gate also passed.

Package publication runs real GGUF evaluate/generate gates on Windows x64.
Windows x86 and ARM64 payloads are cross-compiled and inspected; this does not
constitute executing them on matching hardware. Cargo publishes portable
source. This release does not supply Linux/macOS native binary packages.
