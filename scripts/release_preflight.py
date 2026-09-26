"""Validate that a release tag and every published manifest use one version."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def json_version(path: str) -> str:
    return str(json.loads(read(path))["version"])


def require(name: str, actual: str, expected: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{name}: expected {expected}, found {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.28.6")
    args = parser.parse_args()

    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        print(f"invalid release tag: {args.tag!r}; expected vMAJOR.MINOR.PATCH", file=sys.stderr)
        return 2
    expected = args.tag[1:]
    errors: list[str] = []

    cargo = read("Cargo.toml")
    match = re.search(r"(?ms)^\[workspace\.package\].*?^version\s*=\s*\"([^\"]+)\"", cargo)
    if not match:
        errors.append("Cargo.toml: workspace package version is missing")
    else:
        require("Cargo.toml", match.group(1), expected, errors)

    for path in (
        "bindings/wasm/package.json",
        "bindings/node/package.json",
        "bindings/web/package.json",
        "bindings/web-runtimes/transformersjs/package.json",
        "bindings/web-runtimes/llamacpp-wasm/package.json",
    ):
        require(path, json_version(path), expected, errors)

    pyproject = read("bindings/python/pyproject.toml")
    match = re.search(r"(?m)^version\s*=\s*\"([^\"]+)\"", pyproject)
    if not match:
        errors.append("bindings/python/pyproject.toml: project version is missing")
    else:
        require("bindings/python/pyproject.toml", match.group(1), expected, errors)

    csproj = read("bindings/dotnet/TurboJev/TurboJev.csproj")
    match = re.search(r"<Version>([^<]+)</Version>", csproj)
    if not match:
        errors.append("TurboJev.csproj: package version is missing")
    else:
        require("TurboJev.csproj", match.group(1), expected, errors)

    if errors:
        print("release preflight failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    print(f"release preflight passed for {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
