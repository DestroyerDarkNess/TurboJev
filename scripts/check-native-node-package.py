"""Fail if a platform npm addon lacks its llama.cpp runtime DLLs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from native_assets import required_runtime, verify_machine


def main() -> int:
    directory = Path(sys.argv[1])
    manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    cpu = manifest.get("cpu", [])
    architecture = {"ia32": "x86", "x64": "x64", "arm64": "arm64"}.get(cpu[0] if len(cpu) == 1 else "")
    systems = manifest.get("os", [])
    if architecture is None or len(systems) != 1 or systems[0] not in ("win32", "linux", "darwin"):
        raise SystemExit("Native Node package must declare one supported OS and CPU architecture")
    required = {"turbojev.node"} | required_runtime(systems[0], architecture)
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        print(f"{manifest.get('name', directory.name)} missing native runtime files: {', '.join(missing)}", file=sys.stderr)
        return 1
    for binary in directory.iterdir():
        if not binary.is_file() or not binary.name.endswith((".dll", ".node", ".dylib")) and ".so" not in binary.name:
            continue
        verify_machine(binary.read_bytes(), architecture, binary.name)
    print(f"Native Node package check passed: {manifest['name']}@{manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
