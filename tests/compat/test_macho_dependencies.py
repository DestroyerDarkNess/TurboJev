"""Cross-build guards must reject dependencies that cannot travel with a package."""
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
