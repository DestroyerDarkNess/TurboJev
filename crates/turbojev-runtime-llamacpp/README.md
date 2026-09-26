# turbojev-runtime-llamacpp

Optional GGUF/llama.cpp adapter for TurboJev. It is an outer runtime plugin, not a dependency of the portable TurboJev core.

## v0.9 execution paths and selective projection contract

The adapter now exposes three paths over the same model-facing prompt/token contract:

1. **Reference** — every decision recomputes its complete prompt independently.
2. **Deterministic reuse** — one shared prefix is prefetched and each suffix is evaluated serially from an isolated copy.
3. **Bounded parallel suffixes** — the shared prefix remains immutable while independent suffix branches are copied into distinct llama.cpp sequence IDs and submitted in bounded batches.

The stock FFI implementation still uses llama.cpp's ordinary **full-vocabulary logits** and then gathers TurboJev's candidate token IDs. v0.9 extends `LlamaCppApi` with explicit candidate-only methods, but `FfiLlamaCpp` keeps `selective_output_head = false` until it can genuinely bypass the complete vocabulary head. Custom/future implementations may advertise the capability only after full-vocabulary parity passes.

The parallel path is bounded by portable planner data:

- number of decision tasks;
- the runtime's `n_seq_max` capacity;
- an optional caller `max_parallel_tasks` cap;
- an optional caller memory budget and runtime-provided per-branch estimate.

TurboJev does not inspect CUDA, Metal, Vulkan, an operating system, or a device brand to make that decision.

## Native boundary

The Rust adapter does **not** expose llama.cpp ABI structs. `native/turbojev_llama_shim.*` defines a small versioned C ABI owned by TurboJev. Upstream llama.cpp ABI changes are absorbed there instead of leaking into the Rust/core contracts.

The current TurboJev-owned shim ABI is `5`. In addition to the existing independent/shared-prefix/sequence-scoring paths, v0.25 adds a distinct multi-root operation for cross-request execution:

```text
tj_llama_multi_root_supported
tj_llama_max_multi_root_tasks
tj_llama_max_multi_root_tokens
tj_llama_score_root_batch
```

Each fused decision task receives its own sequence ID and starts at position zero. All prompt roots are submitted in one `llama_decode`, only each root's final prompt token requests logits, and sequence memory is cleared before and after the operation. This is deliberately separate from shared-prefix parallel suffixes.

`FfiLlamaCpp` keeps this capability **disabled by default**. A native host must set `execution.llamacppMultiRootFused=true` only after the exact llama.cpp revision/backend/hardware/model combination passes TurboJev fused-vs-independent conformance. Multi-token candidate shapes remain on the exact serial sequence-scoring path.

## Native build

Build llama.cpp for the target platform, compile the shim with `native/CMakeLists.txt`, link `turbojev_llama_shim`, and enable this crate's `ffi` feature.

The shim uses llama.cpp's Jinja renderer (`llama-common`) to execute the GGUF
chat template with `enable_thinking=false`. Candidate logits are read after
the template's direct-answer prefix. The legacy `llama_chat_apply_template`
API only approximates template families; for Qwen3 it omits the closed
reasoning prefix and therefore scores candidates at the wrong boundary.
No model-specific token IDs or reasoning markers are inserted by TurboJev.

Native packages must include `llama-common` alongside `llama` and the shim.
The CMake build enables that dependency automatically. The model-free
regression test checks Qwen-style and non-reasoning boundaries using the real
Jinja renderer:

```sh
cmake -S crates/turbojev-runtime-llamacpp/native -B artifacts/chat-check -DLLAMA_CPP_SOURCE_DIR=/path/to/llama.cpp -DTURBOJEV_BUILD_NATIVE_TESTS=ON
cmake --build artifacts/chat-check --config Release --target turbojev_chat_template_test
ctest --test-dir artifacts/chat-check -C Release --output-on-failure
```

The shim contains no Windows/Linux/macOS/Android/iOS policy branches. Accelerator/device selection remains the responsibility of the selected llama.cpp build.

Examples:

```text
reference_gguf          independent correctness baseline
shared_prefix_gguf      deterministic cached-serial reuse
parallel_suffix_gguf    bounded parallel suffix execution
```


## Candidate-only projection

The high-level `LlamaCppApi` now has selected-path operations for independent, shared-prefix, and bounded-parallel scoring. The executor dispatches to them only when the API explicitly reports `selective_output_head_supported() == true`. Optimized results report `candidate_rows_projected`, and conformance verifies numerical parity plus row accounting. See `../../docs/CANDIDATE_PROJECTION.md`.
