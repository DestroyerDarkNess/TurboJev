"""Download the pinned real-model validation artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("tests/e2e/final-plan.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model = plan["model"]
    destination = Path(".cache/real-validation") / model["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == model["sha256"]:
        print(f"Model cache hit: {destination}")
        return 0

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(model["url"], headers={"User-Agent": "TurboJev local publisher"})
    digest = hashlib.sha256()
    total = int(model["sizeBytes"])
    received = 0
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                ratio = min(1.0, received / total)
                filled = round(ratio * 30)
                sys.stdout.write(f"\rDownloading {model['filename']} [{'#' * filled}{'-' * (30 - filled)}] {ratio:5.1%}")
                sys.stdout.flush()
        print()
        if received != total or digest.hexdigest() != model["sha256"]:
            raise RuntimeError("Downloaded model size or SHA-256 did not match the pinned plan")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
