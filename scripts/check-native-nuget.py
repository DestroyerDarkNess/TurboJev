"""Fail if a NuGet package is missing model-runtime files for any Windows RID."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from native_assets import architecture, required_runtime, targets, verify_machine


def main() -> int:
    package = Path(sys.argv[1])
    with zipfile.ZipFile(package) as archive:
        files = {name.lower() for name in archive.namelist()}
        for name in archive.namelist():
            for target in targets():
                if name.lower().startswith(f"runtimes/{target['rid']}/native/") and (name.endswith((".dll", ".dylib")) or ".so" in name):
                    verify_machine(archive.read(name), architecture(target), name)
    errors = []
    for framework in ("netstandard2.0", "net48", "net8.0"):
        if f"lib/{framework}/turbojev.dll" not in files:
            errors.append(f"{framework}: missing managed SDK assembly")
    for folder in ("build", "buildtransitive"):
        if f"{folder}/turbojev.targets" not in files:
            errors.append(f"{folder}: missing .NET Framework native asset deployment targets")
    selected = [target for target in targets() if '--windows-only' not in sys.argv or target['os'] == 'win32']
    for target in selected:
        rid = target["rid"]
        prefix = f"runtimes/{rid}/native/"
        runtime_dlls = {Path(name).name for name in files if name.startswith(prefix)}
        capi = "turbojev_capi.dll" if target["os"] == "win32" else "libturbojev_capi" + (".dylib" if target["os"] == "darwin" else ".so")
        required = {capi} | required_runtime(target["os"], architecture(target))
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
