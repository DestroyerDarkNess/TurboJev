"""Validate cross-built macOS library paths, deployment target and public ABI."""
from pathlib import Path
import hashlib
import struct
import subprocess
import sys


def check_signature(data, name):
    offset = 32
    signature = None
    for _ in range(struct.unpack_from("<I", data, 16)[0]):
        command, size = struct.unpack_from("<II", data, offset)
        if command == 0x1D:
            signature = struct.unpack_from("<II", data, offset + 8)
        offset += size
    if signature is None:
        if struct.unpack_from("<I", data, 4)[0] == 0x0100000C:
            raise ValueError(f"{name}: ARM64 Mach-O lacks its required code signature")
        return
    start, length = signature
    blob = data[start:start + length]
    magic, _, count = struct.unpack_from(">III", blob)
    if magic != 0xFADE0CC0:
        raise ValueError(f"{name}: invalid code signature container")
    directory = None
    for index in range(count):
        kind, position = struct.unpack_from(">II", blob, 12 + index * 8)
        if kind == 0:
            directory = blob[position:]
    if directory is None or struct.unpack_from(">I", directory)[0] != 0xFADE0C02:
        raise ValueError(f"{name}: missing code directory")
    hash_offset = struct.unpack_from(">I", directory, 16)[0]
    slots, limit = struct.unpack_from(">II", directory, 28)
    hash_size, hash_type, _, page_shift = directory[36:40]
    if hash_size != 32 or hash_type != 2 or page_shift != 12 or limit != start:
        raise ValueError(f"{name}: unsupported linker signature layout")
    if slots != (limit + 4095) // 4096:
        raise ValueError(f"{name}: incorrect code signature page count")
    for index in range(slots):
        actual = hashlib.sha256(data[index * 4096:min((index + 1) * 4096, limit)]).digest()
        expected = directory[hash_offset + index * 32:hash_offset + (index + 1) * 32]
        if actual != expected:
            raise ValueError(f"{name}: code signature invalid after binary modification")


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
            elif command != 0xD:
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
            check_signature(path.read_bytes(), path.name)
    print("Mach-O dependencies, macOS 11 floor, public ABI and code signatures verified")


if __name__ == "__main__":
    main()
