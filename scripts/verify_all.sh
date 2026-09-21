#!/usr/bin/env sh
set -eu
python scripts/check_structure.py
python scripts/check_dependency_graph.py
python scripts/check_rust_sources.py
python scripts/audit_portability.py
python scripts/check_compat_fixtures.py
python scripts/check_benchmark_fixtures.py
python scripts/check_calibration_fixtures.py
python scripts/check_specialized_fixtures.py
python scripts/check_specialized_evaluation.py
python scripts/check_server_fixtures.py
python scripts/check_runtime_qualification.py
python scripts/check_e2e_fixtures.py
python scripts/check_native_shims.py
python scripts/check_bindings.py
python scripts/source_manifest.py
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo check --target wasm32-unknown-unknown \
  -p turbojev-schema \
  -p turbojev-calibration \
  -p turbojev-tokenizer \
  -p turbojev-prompt \
  -p turbojev-model \
  -p turbojev-specialized \
  -p turbojev-training-data \
  -p turbojev-specialized-artifact \
  -p turbojev-specialized-eval \
  -p turbojev-server-protocol \
  -p turbojev-server-execution \
  -p turbojev-backend \
  -p turbojev-engine \
  -p turbojev-core \
  -p turbojev-wire \
  -p turbojev-backend-mock \
  -p turbojev-backend-custom \
  -p turbojev-compat \
  -p turbojev-conformance \
  -p turbojev-runtime \
  -p turbojev-sequence \
  -p turbojev-runtime-conformance \
  -p turbojev-runtime-qualification \
  -p turbojev-projection \
  -p turbojev-runtime-projection \
  -p turbojev-benchmark \
  -p turbojev
