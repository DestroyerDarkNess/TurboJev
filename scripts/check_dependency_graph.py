#!/usr/bin/env python3
"""Validate workspace-local dependency edges without Cargo."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def local_dependencies(manifest: Path) -> list[tuple[str, Path]]:
    data = load(manifest)
    result: list[tuple[str, Path]] = []
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        for name, spec in data.get(section, {}).items():
            if isinstance(spec, dict) and "path" in spec:
                result.append((name, (manifest.parent / spec["path"]).resolve()))
    for target in data.get("target", {}).values():
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, spec in target.get(section, {}).items():
                if isinstance(spec, dict) and "path" in spec:
                    result.append((name, (manifest.parent / spec["path"]).resolve()))
    return result


def main() -> None:
    manifests = list((ROOT / "crates").glob("*/Cargo.toml"))
    by_dir = {manifest.parent.resolve(): manifest for manifest in manifests}
    package_names = {
        manifest.parent.resolve(): load(manifest)["package"]["name"] for manifest in manifests
    }
    graph: dict[str, set[str]] = {name: set() for name in package_names.values()}
    failures: list[str] = []

    for manifest in manifests:
        owner = package_names[manifest.parent.resolve()]
        for declared_name, dependency_dir in local_dependencies(manifest):
            dependency_manifest = by_dir.get(dependency_dir)
            if dependency_manifest is None:
                failures.append(
                    f"{manifest.relative_to(ROOT)}: path dependency {declared_name!r} "
                    f"does not resolve to a workspace crate"
                )
                continue
            actual_name = package_names[dependency_dir]
            if declared_name != actual_name:
                failures.append(
                    f"{manifest.relative_to(ROOT)}: dependency key {declared_name!r} "
                    f"does not match package {actual_name!r}"
                )
            graph[owner].add(actual_name)

    temporary: set[str] = set()
    permanent: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            start = stack.index(node)
            cycle = stack[start:] + [node]
            failures.append("workspace dependency cycle: " + " -> ".join(cycle))
            return
        temporary.add(node)
        stack.append(node)
        for child in sorted(graph[node]):
            visit(child)
        stack.pop()
        temporary.remove(node)
        permanent.add(node)

    for node in sorted(graph):
        visit(node)

    if failures:
        raise SystemExit("DEPENDENCY GRAPH CHECK FAILED\n" + "\n".join(f"- {x}" for x in failures))

    edges = sum(len(value) for value in graph.values())
    print(f"OK: {len(graph)} workspace crates, {edges} local dependency edges, no cycles")


if __name__ == "__main__":
    main()
