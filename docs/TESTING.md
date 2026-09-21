# TurboJev verification

TurboJev uses layered verification so portable contracts remain testable in restricted/offline environments while releases still require a real Rust toolchain.

## Tier 0 — offline structural and ABI checks

```sh
python scripts/check_structure.py
python scripts/check_dependency_graph.py
python scripts/check_rust_sources.py
python scripts/audit_portability.py
python scripts/check_compat_fixtures.py
python scripts/check_benchmark_fixtures.py
python scripts/check_calibration_fixtures.py
python scripts/check_specialized_fixtures.py
python scripts/check_specialized_evaluation.py
python scripts/check_native_shims.py
python scripts/check_bindings.py
python scripts/source_manifest.py
```

These checks validate manifests/schemas, internal dependency cycles, Rust-source structural sanity, forbidden OS/browser coupling in portable crates, compatibility and benchmark fixtures, the llama.cpp C++ shim smoke executable, host-binding invariants, and release-source integrity. They do **not** replace Rust compilation.

The public C header is also compiled as strict C11:

```sh
cc -std=c11 -Wall -Wextra -Werror \
  -Ibindings/c/include \
  -c bindings/c/smoke.c \
  -o /tmp/turbojev-c-smoke.o
```

## Tier 1 — Rust correctness

```sh
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The pinned toolchain is Rust 1.82.0.

## Tier 2 — portable WebAssembly target

Only portable crates are compiled for `wasm32-unknown-unknown`. Host/runtime adapter crates are intentionally excluded because they may depend on Node-API, JNI, PyO3, native C/C++, or other host facilities.

The exact crate list is kept in `.github/workflows/ci.yml` and `scripts/verify_all.sh`.

## Tier 3 — host binding gates

Node/Web/Transformers.js packages must build, test and package cleanly:

```sh
cd bindings/node && npm run build && npm test && npm pack --dry-run
cd bindings/web && npm run build && npm test && npm pack --dry-run
cd bindings/web-runtimes/transformersjs && npm run build && npm test && npm pack --dry-run
```

The Python wrapper must pass independently of a compiled native extension by using its fixture native module:

```sh
cd bindings/python && pytest -q
```

The Swift package surface must compile against the public C header and preserve a dedicated native execution thread:

```sh
cd bindings/swift && swift build -Xswiftc -warnings-as-errors
```

Android currently has structural/thread-affinity guards in `scripts/check_bindings.py`; full Android toolchain builds belong in the mobile release matrix.

## Tier 4 — concrete runtime gates

The llama.cpp adapter must pass its TurboJev-owned ABI smoke executable and runtime-conformance fixtures. A release that claims a real GGUF model run must record the exact model, revision, quantization and runtime revision; fixture-only conformance must never be described as a real-model benchmark.

For v0.15+, exact multi-token scoring additionally requires parity between the independent and shared-prefix sequence paths and full-vocabulary normalization at every continuation step.

## Source artifact integrity

Regenerate the source manifest after any source change:

```sh
python scripts/source_manifest.py --write
python scripts/source_manifest.py
```

Create the deterministic release archive with:

```sh
python scripts/package_source.py --output ../TurboJev-v0.20.0-source.zip
```

Transient build/cache directories (`target`, `node_modules`, `.build`, `.pytest_cache`, `__pycache__`) are excluded from both the manifest and source archive.

## One command

On a machine with Rust installed:

```sh
./scripts/verify_all.sh
```

If Rust is absent and normal network access is available, `./scripts/bootstrap_rust.sh` installs the pinned toolchain. The bootstrap script is development tooling only and is not part of the runtime.


## v0.24 fused cross-request conformance

A runtime must not advertise cross-request `fusedBatch=true` merely because it can batch tokens or suffixes inside one request. Production adapters must implement the explicit fused runtime operation and pass `verify_fused_runtime_executor` against the independent reference path. Core conformance additionally compares `TurboJevEngine::evaluate_fused` with independent typed evaluation, while server fixtures verify scheduling and reported execution metadata. The deterministic fixture exists only to exercise this pipeline; it is not evidence that stock llama.cpp or another production runtime supports multi-root fusion.

## v0.25 llama.cpp multi-root conformance

The native shim smoke gate validates ABI v5 without requiring a downloaded GGUF by compiling against the stateful llama.cpp stub. The stub mirrors current `llama_get_logits_ith` batch-token indexing so output-row mistakes are observable. Coverage includes:

- two independent roots with unequal prompt lengths and independent reference parity;
- 48 rounds over 2..8 roots with changing prompts/candidate sets;
- exact candidate-logit row ordering;
- memory clear before/after fused execution and post-fused independent-score isolation;
- the pre-existing shared-prefix parallel-suffix path under the same corrected output-index semantics.

This native smoke test proves TurboJev's shim logic, not every accelerator backend. Production llama.cpp fused capability therefore stays disabled by default. A real deployment must run the reusable fused runtime/core conformance against the exact model + llama.cpp revision + backend + hardware before opting in.


## v0.26 runtime qualification

Run `python scripts/check_runtime_qualification.py`. Real deployments additionally create an exact fingerprint with `scripts/runtime_qualification.py fingerprint`, run independent-vs-fused conformance on that target, and create a qualification manifest from the raw observation. Changing the model hash, llama.cpp revision, backend, driver or hardware invalidates the evidence.

## v0.27 cross-language E2E gate

Run `python scripts/check_e2e_fixtures.py` for offline harness/schema validation. The final v0.28.2 release gate invokes `scripts/e2e_language_matrix.py` with a real small GGUF, built `turbojev_capi` library, Node addon and Python extension. Node/TS, Python, C and .NET/C# are mandatory pass rows.

## v0.28/v0.28.1/v0.28.2 final real-model release gate

Offline validation of the final harness is:

```sh
python scripts/check_runtime_qualification.py
python scripts/check_e2e_fixtures.py
```

Those checks only prove the contracts and harness structure. The pre-1.0 evidence gate requires a real GGUF, pinned llama.cpp source and the native language toolchains. On the Linux release host run:

```sh
python scripts/final_e2e_linux.py \
  --work-dir .e2e-final \
  --report artifacts/final-validation-report.json
```

The final gate requires Rust 1.82.0, Node 22, Python 3.12 and .NET 8; verifies the pinned model SHA-256 and llama.cpp revision; builds the native shim/C ABI/Node/Python artifacts; produces real independent-vs-multi-root qualification evidence; and requires semantic parity of the complete canonical response across Node/TypeScript, Python, C and .NET/C#. A fixture-only result must never be reported as 1.0 readiness evidence.
