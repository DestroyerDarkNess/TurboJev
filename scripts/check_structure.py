#!/usr/bin/env python3
"""Offline structural validation that does not require a Rust toolchain."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    ROOT / "Cargo.toml",
    ROOT / "SPEC.md",
    ROOT / "docs/ARCHITECTURE.md",
    ROOT / "docs/PORTABILITY.md",
    ROOT / "docs/TESTING.md",
    ROOT / "crates/turbojev/src/lib.rs",
    ROOT / "crates/turbojev-core/src/lib.rs",
    ROOT / "crates/turbojev-schema/src/lib.rs",
    ROOT / "crates/turbojev-backend/src/lib.rs",
    ROOT / "crates/turbojev-engine/src/lib.rs",
    ROOT / "crates/turbojev-model/src/lib.rs",
    ROOT / "crates/turbojev-specialized/src/lib.rs",
    ROOT / "crates/turbojev-training-data/src/lib.rs",
    ROOT / "crates/turbojev-specialized-artifact/src/lib.rs",
    ROOT / "crates/turbojev-specialized-eval/src/lib.rs",
    ROOT / "crates/turbojev-specialized-report/src/main.rs",
    ROOT / "crates/turbojev-server-protocol/src/lib.rs",
    ROOT / "crates/turbojev-server-execution/src/lib.rs",
    ROOT / "crates/turbojev-server/src/lib.rs",
    ROOT / "crates/turbojev-server/src/main.rs",
    ROOT / "crates/turbojev-wire/src/lib.rs",
    ROOT / "crates/turbojev-runtime/src/lib.rs",
    ROOT / "crates/turbojev-runtime-llamacpp/src/lib.rs",
    ROOT / "crates/turbojev-runtime-qualification/src/lib.rs",
    ROOT / "crates/turbojev-benchmark/src/lib.rs",
    ROOT / "crates/turbojev-host-native/src/lib.rs",
    ROOT / "crates/turbojev-capi/src/lib.rs",
    ROOT / "crates/turbojev-python/src/lib.rs",
    ROOT / "crates/turbojev-jni/src/lib.rs",
    ROOT / "bindings/c/include/turbojev.h",
    ROOT / "docs/MOBILE_BINDINGS.md",
    ROOT / "docs/LLAMA_CPP_RUNTIME.md",
    ROOT / "docs/BENCHMARK_PROTOCOL.md",
    ROOT / "docs/CALIBRATION.md",
    ROOT / "docs/SPECIALIZED_MODELS.md",
    ROOT / "docs/TRAINING_DATA.md",
    ROOT / "docs/SPECIALIZED_ARTIFACTS.md",
    ROOT / "docs/SPECIALIZED_EVALUATION.md",
    ROOT / "docs/SERVER.md",
    ROOT / "docs/adr/0025-server-deployment-lifecycle-and-runtime-batch-bridge.md",
    ROOT / "docs/adr/0026-fused-runtime-batch-conformance.md",
    ROOT / "scripts/server_load_harness.py",
    ROOT / "scripts/server_benchmark_bundle.py",
    ROOT / "scripts/runtime_qualification.py",
    ROOT / "scripts/e2e_language_matrix.py",
    ROOT / "scripts/final_e2e_linux.py",
    ROOT / "tests/e2e/final-plan.json",
    ROOT / "crates/turbojev-runtime-llamacpp/native/CMakeLists.txt",
    ROOT / "crates/turbojev-runtime-llamacpp/examples/real_fused_qualification.rs",
    ROOT / ".github/workflows/real-e2e.yml",
    ROOT / "schemas/turbojev.runtime-qualification.schema.json",
    ROOT / "schemas/turbojev.e2e-language-matrix.schema.json",
    ROOT / "schemas/turbojev.e2e-language-report.schema.json",
    ROOT / "schemas/turbojev.final-e2e-plan.schema.json",
    ROOT / "schemas/turbojev.final-validation-report.schema.json",
    ROOT / "schemas/turbojev.native-execution-capabilities.schema.json",
    ROOT / "schemas/turbojev.server-lifecycle.schema.json",
    ROOT / "schemas/turbojev.server-load-plan.schema.json",
    ROOT / "schemas/turbojev.server-load-report.schema.json",
    ROOT / "schemas/turbojev.server-benchmark-bundle.schema.json",
    ROOT / "crates/turbojev-calibrate/src/main.rs",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    manifests = list(ROOT.rglob("Cargo.toml"))
    for manifest in manifests:
        tomllib.loads(manifest.read_text(encoding="utf-8"))

    schemas = list((ROOT / "schemas").glob("*.json"))
    for schema in schemas:
        json.loads(schema.read_text(encoding="utf-8"))

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    members = set(root_manifest["workspace"]["members"])
    actual = {
        str(path.parent.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "crates").glob("*/Cargo.toml")
    }
    actual.update(
        str(path.parent.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "samples").glob("*/Cargo.toml")
    )
    if members != actual:
        raise SystemExit(
            "workspace member mismatch:\n"
            f"  missing from Cargo.toml: {sorted(actual - members)}\n"
            f"  missing on disk: {sorted(members - actual)}"
        )

    print(
        f"OK: {len(manifests)} Cargo manifests, {len(schemas)} JSON schemas, "
        "workspace members match"
    )


if __name__ == "__main__":
    main()
