"""Shared native release matrix and PE/ELF/Mach-O architecture validation."""
import json
import struct
from pathlib import Path

MACHINES = {"x86": 0x14C, "x64": 0x8664, "arm64": 0xAA64}


def targets():
    return json.loads(Path(__file__).with_name("native-targets.json").read_text())


def architecture(target):
    return {"ia32": "x86", "x64": "x64", "arm64": "arm64"}[target["arch"]]


def node_package(target):
    return f"@turbojev/node-{target['os']}-{target['arch']}"


def required_runtime(platform, arch):
    libraries = ("turbojev_llama_shim", "mtmd", "llama-common", "llama", "ggml", "ggml-base", "ggml-cpu")
    if platform == "win32":
        result = {name + ".dll" for name in libraries}
        if arch == "x64":
            result.add("ggml-vulkan.dll")
        return result
    suffix = ".dylib" if platform == "darwin" else ".so"
    return {"lib" + name + suffix for name in libraries}


def wheel_target(filename):
    tag = filename.removesuffix(".whl").rsplit("-", 1)[-1]
    for target in targets():
        if target["wheel"] in tag.split("."):
            return target
    raise ValueError(f"Unsupported native wheel platform: {tag}")


def verify_machine(data: bytes, architecture: str, name: str) -> None:
    if data.startswith(b"\x7fELF"):
        if len(data) < 52 or data[4] not in (1, 2) or data[5] not in (1, 2):
            raise ValueError(f"{name}: invalid ELF header")
        actual = struct.unpack_from("<H" if data[5] == 1 else ">H", data, 18)[0]
        expected = {"x86": 3, "x64": 62, "arm64": 183}[architecture]
        width = 1 if architecture == "x86" else 2
        if actual != expected or data[4] != width:
            raise ValueError(f"{name}: expected {architecture} ELF, found machine {actual} / class {data[4]}")
        return
    if data[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"):
        if len(data) < 32:
            raise ValueError(f"{name}: invalid Mach-O header")
        actual = struct.unpack_from("<I" if data[:4] == b"\xcf\xfa\xed\xfe" else ">I", data, 4)[0]
        expected = {"x64": 0x01000007, "arm64": 0x0100000C}.get(architecture)
        if actual != expected:
            raise ValueError(f"{name}: expected {architecture} Mach-O, found CPU 0x{actual:x}")
        return
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError(f"{name}: not a supported PE, ELF or Mach-O binary")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 6 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise ValueError(f"{name}: invalid Windows PE header")
    actual = struct.unpack_from("<H", data, offset + 4)[0]
    expected = MACHINES[architecture]
    if actual != expected:
        raise ValueError(f"{name}: expected {architecture} (0x{expected:x}), found 0x{actual:x}")
