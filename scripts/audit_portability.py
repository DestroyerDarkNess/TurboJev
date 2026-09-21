#!/usr/bin/env python3
"""Reject platform coupling in TurboJev portable crates.

This is intentionally conservative. Platform bindings/runtime adapters may use
OS/browser APIs later, but the semantic core and public contracts may not.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_CRATES = [
    "turbojev",
    "turbojev-schema",
    "turbojev-calibration",
    "turbojev-tokenizer",
    "turbojev-prompt",
    "turbojev-model",
    "turbojev-specialized",
    "turbojev-training-data",
    "turbojev-specialized-artifact",
    "turbojev-specialized-eval",
    "turbojev-server-protocol",
    "turbojev-server-execution",
    "turbojev-backend",
    "turbojev-engine",
    "turbojev-core",
    "turbojev-wire",
    "turbojev-backend-mock",
    "turbojev-backend-custom",
    "turbojev-compat",
    "turbojev-conformance",
    "turbojev-runtime",
    "turbojev-sequence",
    "turbojev-runtime-conformance",
    "turbojev-runtime-qualification",
    "turbojev-projection",
    "turbojev-runtime-projection",
    "turbojev-benchmark",
]

FORBIDDEN_SOURCE_PATTERNS = {
    r"\bstd::fs\b": "filesystem access belongs in host/runtime adapters",
    r"\bstd::net\b": "network access belongs in host/runtime adapters",
    r"\bstd::process\b": "process execution is not portable core behavior",
    r"\bstd::env\b": "environment variables belong in host bindings",
    r"\bstd::os::": "OS-specific APIs are forbidden in portable crates",
    r"target_os\s*=": "portable crates must not branch on operating system",
    r"target_arch\s*=": "portable crates must not branch on architecture",
    r"\bwasm_bindgen\b": "WASM bindings belong in a dedicated adapter crate",
    r"\bweb_sys\b": "browser APIs belong in a dedicated adapter crate",
    r"\bwindows_sys\b": "Windows APIs belong in a dedicated adapter crate",
    r"\blibc\b": "native ABI dependencies belong in runtime adapters",
}

FORBIDDEN_DEPENDENCIES = {
    "tokio",
    "reqwest",
    "ureq",
    "libc",
    "winapi",
    "windows",
    "windows-sys",
    "wasm-bindgen",
    "web-sys",
    "js-sys",
    "jni",
    "objc",
    "core-foundation",
}


def parse_dependency_names(manifest: Path) -> set[str]:
    import tomllib

    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    names: set[str] = set()
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        names.update(data.get(section, {}).keys())
    for key, value in data.get("target", {}).items():
        del key
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            names.update(value.get(section, {}).keys())
    return names


def main() -> None:
    failures: list[str] = []
    for crate in PORTABLE_CRATES:
        crate_root = ROOT / "crates" / crate
        manifest = crate_root / "Cargo.toml"
        if not manifest.is_file():
            failures.append(f"missing portable crate manifest: {crate}")
            continue

        deps = parse_dependency_names(manifest)
        forbidden = sorted(deps & FORBIDDEN_DEPENDENCIES)
        if forbidden:
            failures.append(f"{crate}/Cargo.toml: forbidden dependencies {forbidden}")

        for source in (crate_root / "src").rglob("*.rs"):
            text = source.read_text(encoding="utf-8")
            for pattern, reason in FORBIDDEN_SOURCE_PATTERNS.items():
                if re.search(pattern, text):
                    failures.append(
                        f"{source.relative_to(ROOT)}: /{pattern}/ -> {reason}"
                    )

    if failures:
        raise SystemExit("PORTABILITY AUDIT FAILED\n" + "\n".join(f"- {x}" for x in failures))

    print(f"OK: {len(PORTABLE_CRATES)} portable crates contain no forbidden OS/browser coupling")


if __name__ == "__main__":
    main()
