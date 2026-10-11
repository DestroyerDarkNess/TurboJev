"""Fail if a NuGet package is missing model-runtime files for any Windows RID."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from native_assets import verify_machine


def main() -> int:
    package = Path(sys.argv[1])
    with zipfile.ZipFile(package) as archive:
        files = {name.lower() for name in archive.namelist()}
        for name in archive.namelist():
            for architecture in ("x86", "x64", "arm64"):
                if name.lower().startswith(f"runtimes/win-{architecture}/native/") and name.lower().endswith(".dll"):
                    verify_machine(archive.read(name), architecture, name)
    errors = []
    for framework in ("netstandard2.0", "net48", "net8.0"):
        if f"lib/{framework}/turbojev.dll" not in files:
            errors.append(f"{framework}: missing managed SDK assembly")
    for folder in ("build", "buildtransitive"):
        if f"{folder}/turbojev.targets" not in files:
            errors.append(f"{folder}: missing .NET Framework native asset deployment targets")
    for rid in ("win-x86", "win-x64", "win-arm64"):
        prefix = f"runtimes/{rid}/native/"
        runtime_dlls = {Path(name).name for name in files if name.startswith(prefix)}
        required = {"turbojev_capi.dll", "turbojev_llama_shim.dll", "mtmd.dll", "llama-common.dll", "llama.dll",
                    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"}
        if rid == "win-x64":
            required.add("ggml-vulkan.dll")
        missing = required - runtime_dlls
        if missing:
            errors.append(f"{rid}: missing {', '.join(sorted(missing))}")
    if errors:
        print("NuGet runtime asset check failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print(f"NuGet runtime asset check passed: {package.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
