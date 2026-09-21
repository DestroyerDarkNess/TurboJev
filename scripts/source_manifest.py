#!/usr/bin/env python3
"""Generate or verify TurboJev's deterministic source manifest."""
from __future__ import annotations

import argparse
import hashlib
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
}
EXCLUDED_NAMES = {".DS_Store", "SOURCE_MANIFEST.sha256"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tgz"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if relative.name in EXCLUDED_NAMES or relative.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(relative)
    return sorted(files, key=lambda p: p.as_posix())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
