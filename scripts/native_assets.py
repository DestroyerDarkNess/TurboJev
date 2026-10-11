"""Validate Windows binary architecture without executing foreign machine code."""
import struct

MACHINES = {"x86": 0x14C, "x64": 0x8664, "arm64": 0xAA64}


def verify_machine(data: bytes, architecture: str, name: str) -> None:
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError(f"{name}: not a Windows PE binary")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 6 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise ValueError(f"{name}: invalid Windows PE header")
    actual = struct.unpack_from("<H", data, offset + 4)[0]
    expected = MACHINES[architecture]
    if actual != expected:
        raise ValueError(f"{name}: expected {architecture} (0x{expected:x}), found 0x{actual:x}")
