#!/usr/bin/env python3
"""Generate or verify TurboJev's deterministic source manifest."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"
EXCLUDED_DIRS = {
    ".git",
    ".build",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "target",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".codegraph",
    ".playwright-cli",
    "other",
    ".e2e-final",
    "artifacts",
    ".artifacts",
    "bin",
    "obj",
}
EXCLUDED_NAMES = {".DS_Store", "SOURCE_MANIFEST.sha256"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tgz", ".gguf", ".safetensors", ".bin",
                     ".dll", ".pyd", ".node", ".lib", ".obj", ".pdb", ".exp", ".exe"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for directory, directories, filenames in os.walk(ROOT):
        directories[:] = [name for name in directories if name not in EXCLUDED_DIRS]
        for filename in filenames:
            relative = (Path(directory) / filename).relative_to(ROOT)
            if relative.parts[:3] in {
                ("bindings", "node", "dist"),
                ("bindings", "python", "dist"),
            }:
                continue
            if relative.parts[:1] == ("samples",) and any(part in {"vendor", "dist"} for part in relative.parts[1:-1]):
                continue
            if relative.name in EXCLUDED_NAMES or relative.name == ".env" or relative.suffix in EXCLUDED_SUFFIXES:
                continue
            files.append(relative)
    return sorted(files, key=lambda p: p.as_posix())


def digest(path: Path) -> str:
    content = path.read_bytes()
    # Git may check text out as CRLF on Windows and LF on Linux.
    # Hash the same canonical text on both hosts; preserve binary bytes.
    if b"\0" not in content:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def render() -> str:
    return "".join(f"{digest(ROOT / path)}  {path.as_posix()}\n" for path in source_files())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="regenerate SOURCE_MANIFEST.sha256")
    args = parser.parse_args()
    expected = render()
    if args.write:
        MANIFEST.write_text(expected, encoding="utf-8")
        print(f"OK: wrote {len(source_files())} source hashes to {MANIFEST.name}")
        return 0
    if not MANIFEST.is_file():
        raise SystemExit("SOURCE MANIFEST CHECK FAILED: SOURCE_MANIFEST.sha256 is missing")
    actual = MANIFEST.read_text(encoding="utf-8")
    if actual != expected:
        raise SystemExit("SOURCE MANIFEST CHECK FAILED: regenerate with scripts/source_manifest.py --write")
    print(f"OK: {len(source_files())} source hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
