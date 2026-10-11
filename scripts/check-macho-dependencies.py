"""Validate cross-built macOS library paths, deployment target and public ABI."""
from pathlib import Path
import struct
import subprocess
import sys


def check(path, bundled):
    data = path.read_bytes()
    if data[:4] != b"\xcf\xfa\xed\xfe":
        raise ValueError(f"{path.name}: expected a little-endian 64-bit Mach-O")
    offset = 32
    dependencies = []
    rpaths = []
    for _ in range(struct.unpack_from("<I", data, 16)[0]):
        command, size = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError(f"{path.name}: invalid Mach-O load command")
        if command in (0xC, 0xD, 0x80000018, 0x8000001F, 0x8000001C):
            start = offset + struct.unpack_from("<I", data, offset + 8)[0]
            text = data[start:offset + size].split(b"\0", 1)[0].decode()
            if command == 0x8000001C:
                rpaths.append(text)
            else:
                dependencies.append(text)
        if command in (0x32, 0x24):
            version_offset = 12 if command == 0x32 else 8
            minimum = struct.unpack_from("<I", data, offset + version_offset)[0]
            if minimum > 11 << 16:
                raise ValueError(f"{path.name}: deployment target exceeds macOS 11")
        offset += size
    for dependency in dependencies:
        if dependency.startswith(("/usr/lib/", "/System/Library/")):
            continue
        if not dependency.startswith(("@rpath/", "@loader_path/")):
            raise ValueError(f"{path.name}: non-relocatable dependency {dependency}")
        if Path(dependency).name not in bundled:
            raise ValueError(f"{path.name}: missing dependency {dependency}")
        if dependency.startswith("@rpath/") and "@loader_path" not in rpaths:
            raise ValueError(f"{path.name}: missing @loader_path runtime search path")
    required = {
        "libturbojev_capi.dylib": {"_tj_engine_open_json", "_tj_engine_evaluate_json", "_tj_engine_generate_json", "_tj_engine_close"},
        "libturbojev_llama_shim.dylib": {"_tj_llama_abi_version", "_tj_llama_open", "_tj_llama_generate_field", "_tj_llama_score_media_next"},
        "turbojev.node": {"_napi_register_module_v1"},
    }.get(path.name, {"_PyInit__native"} if path.name.startswith("_native") else set())
    if required:
        output = subprocess.check_output(["llvm-nm-18", "--defined-only", "--extern-only", str(path)], text=True)
        exports = {line.split()[-1] for line in output.splitlines() if line.strip()}
        if required - exports:
            raise ValueError(f"{path.name}: missing ABI exports {sorted(required - exports)}")


def main():
    directory = Path(sys.argv[1])
    names = {path.name for path in directory.iterdir()}
    for path in directory.iterdir():
        if path.suffix in (".dylib", ".node", ".so"):
            check(path, names)
    print("Mach-O dependency closure, macOS 11 floor and public ABI verified")


if __name__ == "__main__":
    main()
