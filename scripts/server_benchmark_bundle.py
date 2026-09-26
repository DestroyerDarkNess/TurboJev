#!/usr/bin/env python3
"""Create a deterministic provenance bundle for TurboJev server load evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONTRACT = "turbojev.server-benchmark-bundle.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, base: Path) -> dict[str, str]:
    try:
        display = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        display = path.name
    return {"path": display, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("load_plan", type=Path)
    parser.add_argument("load_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server-version", default="0.29.0")
    args = parser.parse_args()

    for source in (args.config, args.load_plan, args.load_report):
        if not source.is_file():
            raise SystemExit(f"missing benchmark artifact: {source}")
        json.loads(source.read_text(encoding="utf-8"))

    base = Path.cwd()
    bundle = {
        "contract": CONTRACT,
        "serverVersion": args.server_version,
        "artifacts": {
            "config": artifact(args.config, base),
            "loadPlan": artifact(args.load_plan, base),
            "loadReport": artifact(args.load_report, base),
        },
    }
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
