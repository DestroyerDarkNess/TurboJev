"""Fail if a platform npm addon lacks its llama.cpp runtime DLLs."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    directory = Path(sys.argv[1])
    manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    required = {"turbojev.node", "turbojev_llama_shim.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"}
    if "x64" in manifest.get("name", ""):
        required.add("ggml-vulkan.dll")
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        print(f"{manifest.get('name', directory.name)} missing native runtime files: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"Native Node package check passed: {manifest['name']}@{manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
