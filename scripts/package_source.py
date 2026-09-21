#!/usr/bin/env python3
"""Create a deterministic TurboJev source archive from the verified manifest."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from source_manifest import MANIFEST, ROOT, source_files

ZIP_EPOCH = (2020, 1, 1, 0, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not MANIFEST.is_file():
        raise SystemExit("SOURCE_MANIFEST.sha256 is missing; run source_manifest.py --write")
    files = source_files() + [Path("SOURCE_MANIFEST.sha256")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            info = zipfile.ZipInfo(f"TurboJev/{relative.as_posix()}", ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sha_path = args.output.with_suffix(args.output.suffix + ".sha256")
    sha_path.write_text(f"{sha}  {args.output}\n", encoding="utf-8")
    print(f"OK: wrote {args.output} ({len(files)} files)")
    print(f"SHA-256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
