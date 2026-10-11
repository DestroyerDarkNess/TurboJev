# Changelog

## 0.34.0

- Add bundled Linux x86/x64/ARM64 and macOS Intel/Apple Silicon native packages alongside Windows x86/x64/ARM64.
- Use a shared eight-target release matrix for npm optional dependencies, Python stable-ABI wheels and NuGet runtime assets.
- Verify PE, ELF and Mach-O architectures before packaging; check relocatable library paths, Linux glibc 2.28 compatibility and macOS 11 deployment targets.
- Include Metal with embedded kernels in macOS builds and retain Windows x64 Vulkan support. Linux builds use CPU inference.
- Add local manylinux and LLVM/macOS cross-build scripts, with real-model validation where executable and explicit build-only evidence for cross-built macOS artifacts.
- Update installation guidance, source sample staging and standalone registry demos for the expanded desktop platform matrix.

## 0.33.1

- Restore Python 3.9 imports and annotation introspection; verify the packaged stable-ABI extension on CPython 3.9, 3.12 and 3.14, and CPython 3.12 x86.
- Declare the Node wrapper's actual Node 18+ requirement and retain Node-API 8 compatibility across runtime versions. Validate native package architecture before publication.
- Correct Android repository configuration, remove a duplicate platform JSON dependency and lower the wrapper minimum to API 24. Align the Android module version with the workspace.
- Lower the source Swift wrapper's manifest requirements to Swift 5.7, iOS 12 and macOS 10.13; read native open/close errors on their affinity thread and stop that thread on failure. Apple runtime validation remains pending.
- Reject Python/NuGet CI uploads missing packaged inference libraries and check PE architectures in Windows wheels, npm addons and NuGet assets.
- Multi-target the .NET SDK for .NET Standard 2.0, .NET Framework 4.8 and .NET 8; keep the same typed SDK and native runtime contracts across targets.
- Copy packaged native runtime dependencies for .NET Framework and load the directory matching the running process without changing its PATH.
- Add portable UTF-8 interop, synchronous disposal and context-independent SDK continuations for WinForms/WPF consumers.
- Gate NuGet publishing on installed-package compatibility checks, including legacy .NET consumers, UTF-8, schema/result helpers, cancellation, disposal and real GGUF inference.

## Unreleased

No unreleased changes.

## 0.33.0

- Add the Dice schema helper in Rust, Python, Node/browser and .NET, with a real fair-die example and explicit permutation costs.
- Add argmax/sample decoding compatibility, validated plain-decimal digit limits, and opt-in actual formatted input diagnostics.
- Include transitive dependency answers while isolating unrelated branches, key ordinal distributions by level names, and accept equivalent integer condition operands.
- Preserve reported partial usage on generation failure/cancellation; use fresh host entropy for sample requests without an explicit seed.
- Expand contract tests, shared sample recipes, registry demos and self-contained SDK guidance.

## 0.32.0

- Extend the Rust schema compiler with nested objects, inherited dependency gates, described choices, confidence conditions and ordered levels. Preserve JSON property order and distinguish normalized confidence from raw probability.
- Add bounded scalar/object arrays, continuation, nullable items, duplicate stopping and seeded sampling. Every appended item comes from actual model inference; unsupported combinations fail before inference.
- Raise finite enum schemas to 512 values and control codebooks to 26. Native GGUF and ONNX score complete value continuations for large text enums; browser GGUF accepts an explicit value-scorer plugin.
- Add ordered multi-image evidence, model-free schema inspection, shared double-precision candidate normalization, typed SDK result helpers and cooperative generation cancellation/deadlines. Native engines remain thread-affine.
- Port the four current TypeLLM cookbook workflows to documentation and runnable request samples. Add Rust, C ABI, Python, Node, .NET and WASM contract checks plus real Qwen3-0.6B extraction/large-enum validation. See `docs/TYPELLM_0_6_9_PORT.md` for runtime limits and measured coverage.

## 0.31.0

- Add opt-in `orderRobustness: "balanced"` to native and browser `evaluate`: canonical Williams layouts retain the full option set, calibrate each layout once and average probabilities by semantic key. Single-pass `none` remains the default; existing cyclic logit averaging is retained. Ordered scores and nouls stay single-pass.
- Resolve exact choice/enum probability ties by stable semantic key/value rather than caller position. Unequal probabilities are never treated as ties. Preserve the browser's legacy cyclic boolean and expose the new string policy.

- Preserve ordered score scales during training-data materialization while retaining independent output-code augmentation and source-weight conservation. Choice and boolean augmentation are unchanged.
- Document the source audit of alternative Jev implementations, the distinction between unordered choices and ordered scores, and the limits of cyclic versus canonical balanced permutation policies. Add regression checks for score training scales and canonical enum layouts.

## 0.30.1

- Negotiate shared-prefix scoring only when the loaded llama.cpp context has at least two sequences. A context configured with `nSeqMax=1` now evaluates multiple questions through the independent reference path instead of decoding an invalid sequence ID.
- Add real-model regression checks for single-sequence, multi-question requests in Node, Python and the NuGet package gate. Preserve shared-prefix reuse when two or more sequences are available.

## 0.30.0

- Add `generate` across Rust, Python, Node/TypeScript, browser/WASM, C, .NET, Android and Swift with a shared Rust workflow: scalar JSON types, nullable fields, typed enums, dependencies, conditions, plain prior answers and explicit skipped fields.
- Add bounded native JSON grammar generation and browser GGUF JSON Schema generation for open strings/numbers; Transformers.js validates bounded text generation afterward. Finite fields retain direct candidate scoring.
- Add per-field thinking flags, budgets, efforts and actual model-selected automatic effort; unsupported runtimes return errors. Add opt-in seeded, balanced and exhaustive permutation probability averaging with a 720-evaluation cap.
- Preserve semantic option labels for boolean/numeric fields and distinguish actual JSON null from absent output or the string "null". Return typed probability distributions without key collisions.
- Disable automatic llama.cpp operation offload for CPU requests, in addition to setting zero GPU weight layers. Native shim ABI advances to 10.
- Accept integrated GPU devices for explicit GPU requests instead of rejecting a working Vulkan iGPU before model loading.
- Add field regression coverage and reproducible real-model numeric/JevBench evals with licenses, hashes, raw outputs, error accounting and latency metrics. Port fair-die and receipt examples to published-package demos.

## 0.29.2

- Score native option tokens after the structured answer prefill, using the same prefix during candidate tokenization. This fixes the measured first-option bias in simple text decisions; spatial reasoning still depends on the model and prompt.
- Add optional native text `thinking` (default false) with a bounded token budget, explicit unsupported-mode errors, and actual generated-token accounting. Browser runtimes reject thinking requests until implemented.
- Add native GGUF image/audio projector integration and browser GGUF media workers. Report observed runtime capabilities and propagate unsupported-media failures; see multimodal validation for measured coverage and remaining limitations.

- Fix the native GGUF scoring boundary by evaluating the model's actual Jinja chat template with `enable_thinking=false`, rather than the legacy template-family approximation. This preserves Qwen3's closed reasoning prefix before candidate logits are read.
- Render native choices as explicit labeled options and include both each option key and description. This matches the direct WASM/SemIf prompt structure instead of asking the model to interpret opaque codes inside a JSON array.
- Keep native choice scoring single-pass by default; cyclic robustness remains opt-in. Remove the unvalidated experimental binary policy.
- Include the Jinja runtime library in native packages and invalidate cached native builds when shim sources change.

## 0.29.1

- Align native GGUF choice prompts with direct labeled scoring to improve parity across runtimes.

## 0.29.0

- Remove the legacy `classify` API from Rust, native, C, Node, Python, .NET, Android, Swift, and browser/WASM bindings. Submit text choices through the canonical typed `evaluate` request like every other decision.
- Update samples, smoke tests, website documentation, and package versions for the breaking API change.

## 0.28.6

- Make CPU the native runtime default and add an explicit `device: "cpu" | "gpu"` load option across bindings.
- Build the Windows x64 llama.cpp package with Vulkan GPU support; x86 and ARM64 remain CPU-only.
- Return a clear error when GPU is requested but the native build has no GPU backend/device.

## 0.28.5

- Build the published Node, Python, and .NET native bindings with the optional llama.cpp runtime enabled.
- Bundle the pinned llama.cpp shim and its runtime DLLs with each architecture-specific package for Windows x86, x64, and ARM64.
- Add wheel and NuGet asset checks so packages missing a concrete GGUF runtime fail before publication.

## 0.28.4

- Make cyclic candidate-order correction opt-in; single-pass scoring is now the default across the Rust core and browser/WASM text playground.
- Add a browser playground toggle for cyclic correction and regenerate the WASM interface and published site assets.

## 0.28.3

- Made cyclic candidate-order averaging the default for unordered `choice`
  questions across sync and async engines and native language bindings. This
  removes candidate-position changes in the browser's text path and keeps
  option probabilities mapped to their original keys.
- Added cyclic order correction to the async core used by browser/WASM.
- Added an explicit `orderRobustness: "none"` override for single-pass latency.

## 0.28.2
- Project licensing and package metadata now use Apache-2.0 for commercial-friendly distribution.
- `.NET DisposeAsync` now tears down the dedicated affinity worker in a `finally` block even when native close reports an error.

- Final-gate bugfix only; no architecture changes.
- .NET/C# E2E now executes the already-built Release configuration with `--no-build` instead of silently rebuilding Debug during the parity run.
- `TurboJevEngine.LoadAsync` now tears down its dedicated native-affinity thread when native engine opening fails, preventing a leaked background worker on bad model/ABI/configuration errors.
- Offline binding/E2E gates now assert both invariants before the real-model release run.

## 0.28.1

- Hardened the final real-model gate without adding a new architecture phase.
- Corrected the pinned TensorBlock SmolLM2 Q2_K byte size to 88,202,816 while retaining the verified SHA-256.
- Final Linux E2E now enforces Node 22.x, Python 3.12.x and .NET SDK 8.0.x in addition to exact Rust 1.82.0.
- Final E2E freezes one Cargo and npm dependency resolution before compilation, executes Cargo with `--locked`, uses `npm ci`, records both lock SHA-256 values, and uploads the frozen lockfiles with release evidence.
- Pinned TypeScript / Node type development dependencies used by host packages instead of floating caret ranges.
- Upgraded the final validation report to `turbojev.final-validation-report.v2` with exact model size and dependency-resolution provenance.
- Exact-pinned the workspace serialization/error/map baseline plus ABI/toolchain-sensitive binding dependencies (N-API, PyO3, wasm-bindgen stack and JNI); the generated Cargo.lock freezes the remaining transitive graph for the evidence run.
- Real-model CI now runs workspace fmt/check/clippy/test only after lock generation and through the same final orchestrator, eliminating the previous pre-lock Cargo resolution.
- CPU qualification fingerprints now record the detected CPU model plus Linux kernel identity instead of using architecture alone as the device identity.

## 0.28.0

- Added the final real-model release plan pinned to Rust 1.82.0, Node 22, Python 3.12, .NET 8, llama.cpp b10964/b29c606 and a verified small open-weight SmolLM2-135M-Instruct Q2_K GGUF artifact.
- Upgraded the cross-language report to `turbojev.e2e-language-report.v2`: Node/TypeScript, Python, C and .NET/C# now emit the full canonical `EvaluationResponse`, and the harness proves semantic parity with explicit floating-point tolerance rather than checking process exit alone.
- Expanded the canonical E2E request to exercise `choice`, `boolean/noul` and ordered `score` in one model invocation while preserving the zero-generation invariant.
- Added a real llama.cpp fused qualification executable that tokenizes with the checkpoint's real chat template, compares independent-vs-multi-root logits/argmax, and re-scores after fusion to detect sequence/KV leakage.
- Added a native CMake integration for pinned llama.cpp + TurboJev shim, plus `scripts/final_e2e_linux.py` to build all native bindings, qualify the runtime and execute the full language matrix in one command.
- Added `.github/workflows/real-e2e.yml` as the reproducible release host for the real-model gate and versioned final-plan/final-validation schemas.
- Kept 1.0 readiness evidence separate from implementation: fixture/offline gates cannot substitute for a passing real-model final-validation report.

## 0.27.0

- Added the final cross-language real-model E2E harness for Node/TypeScript, Python, C and .NET/C#, with Swift remaining an optional host build gate.
- Added versioned E2E matrix/report schemas and shared canonical request fixtures.
- Added `scripts/e2e_language_matrix.py` to run one exact GGUF artifact through every native binding and record model/request SHA-256 plus per-language outcomes.
- Synchronized native host package metadata to v0.27.0.

## 0.26.0

- Added portable `turbojev-runtime-qualification` and `turbojev.runtime-qualification.v1`.
- llama.cpp multi-root opt-in now requires evidence plus an exact runtime/model/backend/hardware fingerprint; a boolean alone no longer authorizes fused execution.
- Added deterministic qualification/fingerprint tooling, schemas, fixtures, docs and release gates.
- Preserved truthful cancellation semantics: upstream llama.cpp abort callbacks are CPU-only, so TurboJev still does not claim generic cooperative cancellation or hard deadlines.

## 0.25.0

- Bumped the TurboJev-owned llama.cpp shim to ABI v5 with a true multi-root operation: independent prompt roots use distinct llama.cpp sequence IDs inside one `llama_decode` call.
- Added `FusedRuntimeExecutor` support to `LlamaCppExecutor` and shape-aware eligibility for single-token candidate tasks, total root count and total prompt-token capacity.
- Wired llama.cpp fused execution through `TurboJevEngine::evaluate_fused` and `NativeSession`, while preserving automatic serial fallback for ineligible request groups.
- Corrected batched `llama_get_logits_ith` handling to use requested batch-token indices and hardened the native stub to model current upstream output-index semantics.
- Added 2-root independent-vs-fused parity plus 48-round multi-root cache/output-isolation stress coverage in the native ABI smoke test.
- Kept llama.cpp multi-root capability disabled by default behind `execution.llamacppMultiRootFused`; target model/backend/hardware conformance is required before a host advertises it.
- Added a versioned llama.cpp multi-root contract fixture/schema, ADR 0027, updated server/runtime gates and synchronized host package metadata to v0.25.0.
- Cooperative/hard cancellation remains disabled for llama.cpp; v0.25 does not generalize CPU abort callbacks into a false accelerator-wide guarantee.

## 0.24.0

- Added portable `FusedScoringBackend` / `FusedBackendCapabilities` so independent-request fusion is a distinct backend operation rather than a loop over ordinary scoring.
- Added `TurboJevEngine::evaluate_fused`, preserving request validation, per-request execution planning, calibration, typed answers and response ordering through one fused backend call.
- Added `FusedRuntimeExecutor` to `turbojev-runtime`; runtime adapters now have an explicit multi-root batch hook instead of reusing intra-request parallel suffix semantics.
- Added `verify_fused_runtime_executor` to runtime conformance for independent-reference-vs-fused candidate-logit parity.
- Enabled true fused execution for the deterministic fixture backend only, giving the server a complete scheduler -> NativeSession -> engine -> backend conformance path. Stock llama.cpp remains `fusedBatch=false`.
- Added `estimatedFusedItemBytes` capability metadata and optional `fusedMemoryBudgetBytes` host policy so cross-request batch size is bounded by fairness, host/runtime item limits and memory estimates.
- Added deterministic server benchmark provenance bundles with SHA-256 references to config/load-plan/load-report artifacts.
- Added ADR 0026, updated schemas/fixtures/server gates, and synchronized host package metadata to v0.24.0.

## 0.23.0

- Added a native-host cross-request capability bridge; `turbojev-server` now consumes `NativeSession::execution_capabilities()` instead of hardcoding runtime capability truth.
- Added `NativeSession::evaluate_fused` as the concrete future fused-batch entry point while stock fixture and llama.cpp sessions continue to reject it with `UNSUPPORTED_CAPABILITY` rather than simulating fusion with serial loops.
- Added `GET /readyz` and an explicit server lifecycle (`ready -> draining -> stopped`) separate from liveness at `GET /healthz`.
- Added programmatic graceful shutdown through `ShutdownToken`; the listener stops accepting, readiness drops before drain, admitted work receives a bounded `drainTimeoutMs` grace period, and new inference is rejected with `SERVER_DRAINING`.
- Added structured request IDs with safe client-provided `X-TurboJev-Request-Id` / `X-Request-Id` support and `X-TurboJev-Request-Id` response correlation.
- Added a loopback-only `reverse-proxy` deployment mode with optional enforcement of `X-Forwarded-Proto: https`; TurboJev still does not terminate TLS or provide public authentication itself.
- Added a stdlib-only reproducible server load harness with versioned load-plan/load-report contracts and loopback-only execution by default.
- Added native execution, lifecycle and load schemas/fixtures, reverse-proxy config coverage, ADR 0025, and updated server/release verification gates.
- Stock llama.cpp remains `fusedBatch=false`, `cooperativeCancellation=false`, and `hardDeadlines=false` until a runtime-specific implementation proves those capabilities with exact conformance.

## 0.22.0

- Added portable `turbojev-server-execution` for cross-request capability negotiation and execution planning.
- Added `CrossRequestExecutor`, serial/fused modes, host/backend fused-size limits and exact serial-vs-fused response parity conformance.
- Reworked the native session worker into a round-robin scheduler so large admitted batches cannot monopolize the thread-affine model session.
- Added `fairnessQuantum` and `maxFusedItems` server controls while preserving ordered batch responses.
- Added `X-TurboJev-Timeout-Ms`, clamped to the configured server timeout, and soft cancellation of work that has not started after a caller deadline.
- Added local scheduler/overload metrics at `GET /v1/server/metrics`.
- Batch metadata now reports the execution mode actually used (`serial-round-robin` or `fused-batch`).
- The stock `NativeSession` remains truthful: `fusedBatch=false`, `cooperativeCancellation=false`, `hardDeadlines=false` until a concrete runtime proves those capabilities.
- Added server execution/metrics schemas, fixtures, ADR 0024 and CI/WASM portability wiring.

## 0.21.0

- Started Phase 11 with portable `turbojev-server-protocol` and host-only `turbojev-server`.
- Added canonical, TypeSafe/System One and AI SDK/Vercel evaluation routes plus gateway-friendly OpenRouter aliases without claiming Chat Completions compatibility.
- Added bounded batch envelopes, stable server errors, request/body limits and deterministic response-order preservation.
- Added one dedicated thread-affine `NativeSession` worker behind a bounded queue so concurrent HTTP clients do not duplicate model state.
- Added loopback-only HTTP/1.0/1.1 transport, connection limits, I/O/session timeouts and fail-fast 503 backpressure.
- Added typed `NativeSession::evaluate` so server transports do not round-trip canonical requests through JSON.
- Added server schemas, fixtures, docs, ADR 0023 and CI/offline gates.
- Cross-request fused model batching remains intentionally unclaimed; v0.21 batch execution is explicit session-thread serial.

## 0.20.0

- Added portable `turbojev-specialized-eval` with versioned external-run, paired-report and reproducible experiment-bundle contracts.
- External runs carry semantic candidate keys, raw logits, calibration provenance, backend/host/quantization/execution-path metadata, latency and peak-memory measurements.
- Generic-vs-specialized reports now fail fast on benchmark, observation, prompt/tokenizer or measurement-environment drift.
- Headline accuracy/NLL/Brier/ECE are computed on baseline cases only through the existing calibration math.
- Added separate candidate-order and paraphrase invariance + accuracy metrics using semantic keys rather than logit positions.
- Added deterministic latency summary, decisions/sec, peak memory and explicit `specialized - generic` deltas without an aggregate winner score.
- Added `turbojev-specialized-report` as a host-only CLI for turning external JSON runs into canonical reports.
- Added SHA-256 experiment bundles, schemas, fixtures, ADR 0022, documentation and a dedicated offline v0.20 gate.
- Kept all model training/fine-tuning frameworks external and optional.

## 0.19.0

- Added portable `turbojev-specialized-artifact`; TurboJev still contains no trainer/optimizer framework.
- Added `BoundaryVerifiedCodeProvider` using the real `PromptRenderer` + `PromptCodec` answer boundary for deterministic single-token code discovery.
- Added `turbojev.model-bound-training-dataset.v1` with exact rendered messages, prompt token ids, candidate token ids and target token id.
- Candidate-logit CE v1 now fails loudly when a verified candidate code is not exactly one token; multi-token inference remains unchanged.
- Added context-limit enforcement and model/recipe/materialized provenance checks during model binding.
- Added specialized manifest construction, strict calibration-profile handoff and runtime compatibility validation.
- Added model-bound schema/fixtures, specialized calibration provenance fixture, docs, ADR 0021 and offline gates.
- Synchronized Node/Web/Transformers.js/Python/.NET/Android package metadata to v0.19.0 and restored the missing specialized fixture step in CI.

## 0.18.0

- Added portable `turbojev-training-data` with versioned semantic source-dataset and materialized candidate-logit dataset contracts.
- Made the canonical `EvaluationRequest -> DecisionBatch` compiler public so inference and specialization share one candidate-normalization path.
- Added typed training targets for choice/noul/score plus recipe language/data-policy validation.
- Added deterministic unique candidate-order and candidate-code permutations with identity retention and `n!` capping.
- Added per-decision augmentation weight conservation and materialized target key/index/code self-validation.
- Added model/tokenizer-aware `CandidateCodeProvider` boundary instead of hard-coding token ids or pretending generic ASCII codes are universally safe.
- Added semantic/materialized dataset schemas, multilingual fixtures, training-data documentation, ADR 0020 and offline verification gates.

## 0.17.0

- Started Phase 10 with portable specialized-model training contracts.
- Added `turbojev-specialized` for LoRA/QLoRA recipes, candidate-logit CE, debiasing/data policy and split-leakage validation.
- Added optional `ModelSpec.decision_contract` while keeping generic models unchanged.
- Added specialized checkpoint/evaluation manifests, 0.6B/0.8B/2B/4B fixtures, schemas, ADR 0019 and offline gates.

## 0.16.0

- Added versioned `CalibrationDataset` and `CalibrationProfile` artifacts with model/tokenizer/prompt/renderer provenance.
- Added deterministic scalar temperature fitting in log-temperature space using weighted multiclass NLL.
- Added weighted NLL, multiclass Brier, top-label ECE, reliability bins and accuracy reporting.
- Added `TemperatureCalibrator::from_profile`; fitted calibration is now explicit and separate from plain conditional softmax.
- Added the host-side `turbojev-calibrate` CLI with `fit` and `report` commands while keeping the calibration math portable.
- Added calibration dataset/profile JSON schemas, reproducible numeric fixtures, ADR 0018 and calibration fixture verification in CI/release gates.

## 0.15.0

- Updated the verification policy so WASM builds target only portable crates and host SDKs have independent build/test/package gates.
- Added deterministic source-manifest verification and reproducible source ZIP tooling for release artifacts.
- Closed the v0.15 release gates for host bindings: direct Python `pytest`, Node/Web/Transformers.js TypeScript builds/tests/package dry-runs, C11 header smoke, and Swift package compilation.
- Fixed the Swift wrapper to use the C importer's `OpaquePointer` handle, propagate throwing work across its dedicated native thread, and expose `info()` without duplicating semantics.
- Added CI jobs for native shim/C header, JavaScript packages, Python wrapper, and Swift package surface so future runtime changes cannot silently break host bindings.
- Added `turbojev-sequence`, a portable prefix-sharing trie for exact multi-token candidate log-likelihood scoring.
- Multi-token scores now use `logit(token) - logsumexp(full vocabulary logits)` at every continuation step; raw logits are never summed as if they were sequence probabilities.
- Candidate trie traversal evaluates each unique candidate prefix once and reports token-work savings through backend metrics.
- `turbojev-runtime` now keeps single-token codes as the fast path and automatically downgrades unsafe multi-token plans to serial full-vocabulary scoring while preserving valid shared-state prefix reuse.
- Added `BackendMetrics.effective_plan` so response execution metadata reflects a runtime correctness downgrade instead of the originally requested optimization.
- Extended the llama.cpp adapter with exact independent and shared-prefix multi-token sequence scoring.
- Bumped the TurboJev llama.cpp shim ABI to v4 and added `tj_llama_score_sequences` / `tj_llama_score_suffix_sequences`.
- Added native ABI smoke coverage for exact multi-token log-likelihood, shared-prefix parity, trie prefix reuse, and single-token distribution parity.
- Extended `turbojev-runtime-conformance` with an exact full-vocabulary sequence-scoring oracle and prefix-work report for third-party runtime adapters.
- Candidate-only projection remains intentionally disabled for exact multi-token scoring until a backend can provide the full-vocabulary normalizer (or an equivalent exact `logZ`) at every continuation state.
- Fixed WASM CI to compile only portable crates and added `turbojev-sequence` to the portability audit.

## 0.14.0

- Added `turbojev-jni` as the Android host ABI over `turbojev-host-native`.
- Added Kotlin Android wrapper using one dedicated `HandlerThread` for open/evaluate/classify/close.
- Added Swift Package wrapper over the stable C ABI using a dedicated Foundation `Thread` to preserve native affinity.
- Added mobile binding documentation, structural release gates, and ADR 0016.
- Reconstructed the v0.10-v0.13 host binding source tree in the delivered workspace so Node/Web/Python/C/.NET/mobile all ship together.


## 0.13.0

- Added `turbojev-host-native` so Node, Python and C/.NET native bindings share one model/session implementation instead of duplicating runtime configuration and decision calls.
- Refactored `turbojev-node` to a thin Node-API session map over the shared native host layer.
- Added `turbojev-capi` plus `bindings/c/include/turbojev.h`, with opaque thread-affine engine handles, UTF-8 JSON transport, explicit string ownership and thread-local stable error reporting.
- Added panic containment at the C ABI boundary and a strict C11 header smoke gate.
- Added `turbojev-python` (PyO3 ABI3) and the `turbojev` Python package with thin JSON/lifetime/error wrappers and unit tests.
- Added the .NET `TurboJev` wrapper over the stable C ABI. Its dedicated managed thread preserves native session affinity across async Task continuations.
- Added Python/.NET/C-ABI verification scripts, CI jobs, documentation, and ADR 0015.

## 0.12.0

- Added `@turbojev/runtime-transformersjs`, the first concrete optional browser inference adapter.
- Added runtime-owned opaque model references so repository/cache-aware runtimes can resolve their own checkpoints.
- Added exact checkpoint chat-template/tokenizer boundary encoding with dynamic candidate-code verification in Rust/WASM.
- Added local causal-LM next-token scoring through Transformers.js with configurable runtime device/dtype.
- Kept capabilities conservative and truthful: reference serial full-vocabulary scoring only; no false shared-prefix, multi-token, parallel, or selective-head claims.
- Added last-position-logit optimization probing with automatic fallback when unsupported.
- Added disposal of temporary tokenizer/input/logit tensors and model cleanup on close.
- Added adapter unit tests, build/package gates, CI, `docs/TRANSFORMERSJS_RUNTIME.md`, and ADR 0014.

## 0.11.0

- Added `turbojev-wasm`, a browser host binding over `TurboJevAsync` and the universal runtime bridge.
- Added `@turbojev/web` with worker isolation, typed `load/classify/evaluate/evaluateCompatible/close`, benchmark helper, and typed errors.
- Added a low-level browser runtime contract for exact prompt encoding/answer-boundary encoding/token decoding/runtime-batch scoring.
- Kept dynamic candidate discovery, round-trip verification, planning, calibration and typed answers in Rust/WASM rather than JavaScript.
- Added browser-owned CacheStorage model resolution plus runtime-owned opaque model resolution.
- Changed remote cache population to consume a response clone concurrently instead of delaying runtime consumption of large weights.
- Added web binding TypeScript tests/build/package gates, dedicated WASM CI, `docs/WEB_BINDING.md`, and ADR 0013.

## 0.10.0

- Added `turbojev-node`, a thin Node-API binding over the canonical Rust/wire contracts.
- Added `@turbojev/node` with async `TurboJev.load`, `classify`, `evaluate`, compatibility evaluation, engine info, and typed errors.
- Native inference runs in one dedicated `worker_threads` worker per engine, preserving thread affinity without blocking the Node main event loop.
- Added local path, `file://`, HTTP(S), and `hf://` model resolution plus binding-owned model caching and download progress.
- Added platform-native addon discovery with `TURBOJEV_NATIVE_PATH` override and platform package fallbacks.
- TypeScript request/response declarations are generated from the canonical TurboJev JSON schemas rather than maintained as a second semantic contract.
- Added worker/native protocol tests, local-model tests, and atomic HTTP download/cache tests.
- Added `docs/NODE_BINDING.md`, ADR 0012, and Node binding checks to the repository verification pipeline.


## 0.9.0

- Added `turbojev-projection`, a portable candidate-only output-head reference implementation with dense candidate rows, full-vocabulary oracle math, candidate-row gathering, and parity diagnostics.
- Added `turbojev-runtime-projection`, a runtime-neutral selective executor over final post-normalization hidden states plus requested output rows.
- Added `candidate_rows_projected` backend accounting so selective work is auditable.
- Added `verify_selective_projection`, which requires both numerical parity against the full-vocabulary baseline and exact candidate-row accounting.
- Extended `LlamaCppApi` with parity-gated selected paths for independent, shared-prefix, and bounded-parallel execution.
- Added fixture coverage proving candidate-only + parallel suffixes preserve reference logits and do not call the full projection path.
- Kept the stock llama.cpp FFI capability disabled rather than mislabel full-vocabulary + gather as candidate-only.
- Added `docs/CANDIDATE_PROJECTION.md` and ADR 0011.
- Expanded portability auditing to cover projection/runtime-conformance crates.

## 0.8.0

- Added bounded parallel suffix execution to the optional llama.cpp/GGUF adapter.
- Added portable `ExecutionBudget` controls for maximum parallel tasks and branch-memory budgeting.
- Added quantitative `BackendResourceHints`; runtimes can report safe parallel capacity and an optional per-branch memory estimate without exposing OS/device APIs.
- Extended `ExecutionPlan` with `parallel_task_limit` and response execution metadata with the selected bound.
- Bumped the TurboJev llama.cpp shim ABI to v3 and added batched isolated suffix scoring over distinct llama.cpp sequence IDs.
- Kept sequence 0 immutable while ephemeral branches are cloned, scored in one decode batch, and removed after every batch.
- Added mixed empty/non-empty suffix handling and exact output-row ordering.
- Added runtime-reported suffix-token submission bounds so large task groups are split before native `n_batch`/context limits are crossed.
- Added 48-round native cache-isolation stress coverage over 2..7 parallel branches.
- Added reference vs cached-serial vs parallel exact-logit parity tests and portable-budget batching tests.
- Added `parallel_suffix_gguf` as the concrete v0.8 example.
- Advanced the native core response contract to `turbojev.core.v4` to expose bounded scheduling metadata.
- Candidate-only output projection remains intentionally disabled until v0.9.

## 0.7.0

- Added deterministic shared-prefix reuse to the optional llama.cpp/GGUF adapter.
- Added runtime capability probing so `shared_prefix` is advertised only when sequence memory is available.
- Added explicit prefix-prefill and suffix-score lifecycle methods to `LlamaCppApi`.
- Bumped the TurboJev llama.cpp shim ABI to v2.
- The native shim preserves sequence 0 as an immutable prefix and recreates ephemeral sequence 1 for every serial suffix.
- Added retained prefix logits for valid empty-suffix tasks.
- Added exact shared-vs-independent logit parity and typed-answer parity tests.
- Strengthened the native test stub so logits depend on emulated sequence state.
- Native shim smoke compilation now treats warnings as errors.
- Fixed `PlanningPolicy::Reference` so it always selects full-vocabulary projection.
- Parallel suffixes and selective-head projection remain disabled.

## 0.6.0

- Added optional `turbojev-runtime-llamacpp` as the first concrete local model runtime.
- Added a TurboJev-owned C shim isolating llama.cpp ABI details from Rust and portable core crates.
- Added exact model chat-template/tokenizer answer-boundary handling through llama.cpp.
- Added independent serial full-vocabulary candidate scoring as the correctness reference path.
- Added `turbojev-benchmark` and multilingual benchmark fixtures.
- Added C++ compile/link/execute smoke testing against an ABI stub.

## 0.5.0

- Added executor-neutral asynchronous backend/runtime contracts with no Tokio/async-std dependency.
- Added `TurboJevAsyncEngine` and `TurboJevAsync` while preserving the synchronous zero-overhead path.
- Added `AsyncRuntimeExecutor` and async composition through the existing `RuntimeBackend`.
- Added `AsyncCustomBackend` for host-provided async runtimes.
- Added `turbojev-runtime-conformance` with answer-boundary, capability, shape, finite-score, and reference-parity checks.
- Added async/runtime conformance ADRs and documentation.
- Preserved non-`Send` futures so browser/mobile thread-affine runtimes remain first-class.

## 0.4.0

- Added `turbojev-runtime`, a portable universal token-level runtime bridge.
- Added `PromptCodec` so candidate validation occurs at the checkpoint's exact chat-template answer boundary.
- Added `RuntimeExecutor`, `RuntimeCapabilities`, token-level batches, exact candidate sequences, and raw runtime outputs.
- Added `RuntimeBackend` composition implementing the stable `ScoringBackend` SPI without coupling the core to any model format/runtime/OS.
- Added deterministic candidate symbol-set probing, single-token preference, multi-token sequence support, and prefix-free verification.
- Added exact longest-common-token-prefix factoring for shared-state execution.
- Added ADR 0006 and extended portability auditing to the runtime bridge.
- Removed mandatory `Send + Sync` from portable runtime traits; added `ThreadSafeScoringBackend` as an opt-in host constraint.

## 0.3.0

- Added `turbojev-compat` with TurboJev, TypeSafe System One and AI SDK evaluation wire profiles.
- Added `turbojev-conformance` for reference-vs-optimized parity reports.
- Added optional `input_tokens` to portable usage metadata.
- Bumped the native core contract to `turbojev.core.v3`.
- Added golden request fixtures and compatibility documentation.
- Added ADRs enforcing wire/runtime separation and measurable optimized-path parity.

## 0.2.0 — universal portable-core milestone

### Added

- `turbojev-engine` with capability-based execution planning.
- `ExecutionPlan` covering prefix reuse, task scheduling, and candidate projection.
- `turbojev-model` with opaque `ModelSource` and model-family-neutral `ModelSpec`.
- `turbojev-wire` as the canonical UTF-8 JSON boundary for future bindings.
- `turbojev-backend-custom` for integrating arbitrary application-owned model runtimes.
- Fixed-width, prefix-free multi-token candidate-code fallback.
- Language-neutral `PromptDocument` IR and model-owned `PromptRenderer` contract.
- Portability audit rejecting OS/browser/runtime coupling in portable crates.
- WASM portable-core CI check.
- Execution metadata in native responses.

### Changed

- `ScoringBackend::score` now receives an explicit semantic `ExecutionPlan`.
- Prompt construction no longer contains a mandatory English system prompt.
- Core contract advanced to `turbojev.core.v2`.
- Prompt IR contract advanced to `turbojev.prompt-ir.v2`.
- Workspace version advanced to 0.2.0.

### Architecture guarantees

- No concrete model runtime is a dependency of `turbojev-core`.
- No model architecture allowlist exists in portable core code.
- No operating-system branching is allowed in portable crates.
- Model loading/downloading is delegated to host/runtime adapters.

## 0.1.0 — architecture milestone

- Canonical `choice`, `noul`/`boolean`, and `score` schema.
- Backend SPI returning raw candidate logits.
- Core-owned softmax/calibration and typed answer construction.
- Dynamic single-token candidate discovery.
- Deterministic fixture backend.
- Initial specification, architecture, and CI structure.
