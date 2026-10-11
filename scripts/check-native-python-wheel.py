"""Reject wheels without the declared stable ABI or matching model-runtime DLLs."""
import sys
import zipfile
from pathlib import Path

from native_assets import verify_machine


def main() -> None:
    wheel = Path(sys.argv[1])
    platforms = {"win32": "x86", "win_amd64": "x64", "win_arm64": "arm64"}
    tag = wheel.stem.rsplit("-", 1)[-1]
    if tag not in platforms or "-cp39-abi3-" not in wheel.name:
        raise SystemExit("Expected a Windows CPython 3.9+ stable-ABI wheel")
    architecture = platforms[tag]
    required = {"turbojev_llama_shim.dll", "mtmd.dll", "llama-common.dll", "llama.dll",
                "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"}
    if architecture == "x64":
        required.add("ggml-vulkan.dll")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        missing = {"turbojev/" + name for name in required} - set(names)
        if missing:
            raise SystemExit("Wheel lacks native inference dependencies: " + ", ".join(sorted(missing)))
        extensions = [name for name in names if name.startswith("turbojev/") and name.endswith(".pyd")]
        if not extensions:
            raise SystemExit("Wheel lacks the Python native extension")
        for name in names:
            if name.startswith("turbojev/") and name.endswith((".dll", ".pyd")):
                verify_machine(archive.read(name), architecture, name)
    print(f"Native Python wheel check passed: {wheel.name}")


if __name__ == "__main__":
    main()
