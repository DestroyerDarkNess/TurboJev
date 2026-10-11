import struct
import unittest

from scripts.native_assets import verify_machine


def image(machine):
    data = bytearray(96)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    return bytes(data)


class NativeArchitectureTests(unittest.TestCase):
    def test_elf_uses_machine_and_pointer_width(self):
        data = bytearray(64)
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", data, 18, 183)
        verify_machine(bytes(data), "arm64", "libllama.so")
        with self.assertRaisesRegex(ValueError, "expected x64 ELF"):
            verify_machine(bytes(data), "x64", "libllama.so")
        data[4] = 1
        with self.assertRaisesRegex(ValueError, "expected arm64 ELF"):
            verify_machine(bytes(data), "arm64", "libllama.so")

    def test_macho_rejects_wrong_cpu(self):
        data = bytearray(32)
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", data, 4, 0x01000007)
        verify_machine(bytes(data), "x64", "libllama.dylib")
        with self.assertRaisesRegex(ValueError, "expected arm64 Mach-O"):
            verify_machine(bytes(data), "arm64", "libllama.dylib")

    def test_labeled_x64_package_rejects_x86_binary(self):
        with self.assertRaisesRegex(ValueError, "expected x64"):
            verify_machine(image(0x14C), "x64", "runtime.dll")

    def test_arm64_package_rejects_x64_binary(self):
        with self.assertRaisesRegex(ValueError, "expected arm64"):
            verify_machine(image(0x8664), "arm64", "runtime.dll")

    def test_non_binary_artifacts_are_rejected(self):
        for data in (b"", b"not a DLL", b"MZ" + b"\xff" * 70):
            with self.assertRaises(ValueError):
                verify_machine(data, "x64", "runtime.dll")


if __name__ == "__main__":
    unittest.main()
