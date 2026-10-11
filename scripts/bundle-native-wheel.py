"""Add architecture-matched llama.cpp DLLs to a built TurboJev wheel."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path
from native_assets import verify_machine


def digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bundle-native-wheel.py WHEEL NATIVE_DLL_DIRECTORY")
    wheel = Path(sys.argv[1])
    native = Path(sys.argv[2])
    architecture = next((arch for tag, arch in (("win32", "x86"), ("win_amd64", "x64"), ("win_arm64", "arm64"))
                         if wheel.name.endswith(f"-{tag}.whl")), None)
    if architecture is None:
        raise SystemExit("Native DLL bundling requires a Windows wheel with a supported architecture tag")
    dlls = sorted(native.glob("*.dll"))
    required = {"turbojev_llama_shim.dll", "mtmd.dll", "llama-common.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"}
    if architecture == "x64":
        required.add("ggml-vulkan.dll")
    missing = required - {path.name.lower() for path in dlls}
    if missing:
        raise SystemExit(f"Native directory {native} is missing: {', '.join(sorted(missing))}")
    for dll in dlls:
        verify_machine(dll.read_bytes(), architecture, dll.name)

    temporary = wheel.with_suffix(".bundling.whl")
    with zipfile.ZipFile(wheel, "r") as source:
        entries = [(item, source.read(item.filename)) for item in source.infolist() if not item.filename.endswith(".dist-info/RECORD")]
        record_name = next(item.filename for item in source.infolist() if item.filename.endswith(".dist-info/RECORD"))
        record_bytes = source.read(record_name)

    rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
    extensions = [(item, data) for item, data in entries if item.filename.endswith(".pyd")]
    if not extensions:
        raise SystemExit("Wheel is missing its native Python extension")
    for item, data in extensions:
        verify_machine(data, architecture, item.filename)
    names = {item.filename for item, _ in entries}
    additions: list[tuple[str, bytes]] = []
    for dll in dlls:
        archive_name = f"turbojev/{dll.name}"
        if archive_name in names:
            continue
        data = dll.read_bytes()
        additions.append((archive_name, data))
        rows.append([archive_name, digest(data), str(len(data))])
    bundled_names = {item.filename.lower() for item, _ in entries} | {
        name.lower() for name, _ in additions
    }
    missing = {f"turbojev/{name}" for name in required} - bundled_names
    if missing:
        raise SystemExit(f"Wheel failed to include required runtime DLLs: {', '.join(sorted(missing))}")
    rows.append([record_name, "", ""])
    record_output = io.StringIO(newline="")
    csv.writer(record_output, lineterminator="\n").writerows(rows)

    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for item, data in entries:
            output.writestr(item, data)
        for name, data in additions:
            output.writestr(name, data)
        output.writestr(record_name, record_output.getvalue())
    temporary.replace(wheel)
    print(f"Bundled {len(additions)} llama.cpp runtime DLLs into {wheel.name}")


if __name__ == "__main__":
    main()
