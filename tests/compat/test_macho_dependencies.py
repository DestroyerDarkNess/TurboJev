"""Cross-build guards must reject dependencies that cannot travel with a package."""
import importlib.util
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/check-macho-dependencies.py"
SPEC = importlib.util.spec_from_file_location("macho_checks", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def text_command(command, value):
    body = value.encode() + b"\0"
    size = (12 + len(body) + 7) & ~7
    return struct.pack("<III", command, size, 12) + body.ljust(size - 12, b"\0")


def signed_arm64():
    """A minimal linker-style signature covering one page of Mach-O bytes."""
    limit = 4096
    directory = bytearray(44 + 32)
    struct.pack_into(">IIIIIIIII", directory, 0,
                     0xFADE0C02, len(directory), 0x20400, 2, 44, 0, 0, 1, limit)
    directory[36:40] = bytes((32, 2, 0, 12))
    length = 20 + len(directory)
    header = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, 1, 16, 0, 0)
    command = struct.pack("<IIII", 0x1D, 16, limit, length)
    code = (header + command).ljust(limit, b"\0")
    directory[44:] = hashlib.sha256(code).digest()
    signature = struct.pack(">IIIII", 0xFADE0CC0, length, 1, 0, 20)
    return code + signature + directory


class MachODependencies(unittest.TestCase):
    def inspect(self, dependency, *, rpath="@loader_path", minimum=11):
        commands = [text_command(0xC, dependency),
                    text_command(0x8000001C, rpath),
                    struct.pack("<IIIIII", 0x32, 24, 1, minimum << 16, 14 << 16, 0)]
        header = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 6,
                             len(commands), sum(map(len, commands)), 0, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "libexample.dylib"
            path.write_bytes(header + b"".join(commands))
            MODULE.check(path, {"libexample.dylib", "libdependency.dylib"})

    def test_relocatable_dependency(self):
        self.inspect("@rpath/libdependency.dylib")

    def test_absolute_build_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-relocatable"):
            self.inspect("/tmp/build/libdependency.dylib")

    def test_missing_library_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing dependency"):
            self.inspect("@rpath/libmissing.dylib")

    def test_missing_relative_search_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime search path"):
            self.inspect("@rpath/libdependency.dylib", rpath="/tmp/build")

    def test_excessive_deployment_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "macOS 11"):
            self.inspect("/usr/lib/libSystem.B.dylib", minimum=14)

    def test_valid_arm64_signature(self):
        MODULE.check_signature(signed_arm64(), "libexample.dylib")

    def test_modification_after_signing_is_rejected(self):
        binary = bytearray(signed_arm64())
        binary[200] ^= 1
        with self.assertRaisesRegex(ValueError, "signature invalid"):
            MODULE.check_signature(binary, "libexample.dylib")

    def test_unsigned_arm64_is_rejected(self):
        header = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "required code signature"):
            MODULE.check_signature(header, "libexample.dylib")


if __name__ == "__main__":
    unittest.main()
