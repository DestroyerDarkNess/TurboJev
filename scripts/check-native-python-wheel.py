"""Reject wheels without the declared stable ABI or matching model-runtime DLLs."""
import sys
import zipfile
from pathlib import Path

from native_assets import architecture, required_runtime, verify_machine, wheel_target


def main() -> None:
    wheel = Path(sys.argv[1])
    target = wheel_target(wheel.name)
    arch = architecture(target)
    if "-cp39-abi3-" not in wheel.name:
        raise SystemExit("Expected a CPython 3.9+ stable-ABI wheel")
    required = required_runtime(target["os"], arch)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        missing = {"turbojev/" + name for name in required} - set(names)
        if missing:
            raise SystemExit("Wheel lacks native inference dependencies: " + ", ".join(sorted(missing)))
        extensions = [name for name in names if name.startswith("turbojev/_native") and name.endswith((".pyd", ".so"))]
        if not extensions:
            raise SystemExit("Wheel lacks the Python native extension")
        for name in names:
            if name.startswith("turbojev/") and (name.endswith((".dll", ".pyd", ".dylib")) or ".so" in name):
                verify_machine(archive.read(name), arch, name)
    print(f"Native Python wheel check passed: {wheel.name}")


if __name__ == "__main__":
    main()
