"""Fail if a NuGet package is missing model-runtime files for any Windows RID."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> int:
    package = Path(sys.argv[1])
    with zipfile.ZipFile(package) as archive:
        files = {name.lower() for name in archive.namelist()}
    errors = []
    for rid in ("win-x86", "win-x64", "win-arm64"):
        prefix = f"runtimes/{rid}/native/"
        runtime_dlls = {Path(name).name for name in files if name.startswith(prefix)}
        required = {"turbojev_capi.dll", "turbojev_llama_shim.dll", "llama.dll"}
        missing = required - runtime_dlls
        if missing:
            errors.append(f"{rid}: missing {', '.join(sorted(missing))}")
        if not any(name.startswith("ggml") and name.endswith(".dll") for name in runtime_dlls):
            errors.append(f"{rid}: missing ggml runtime DLL")
    if errors:
        print("NuGet runtime asset check failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print(f"NuGet runtime asset check passed: {package.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
