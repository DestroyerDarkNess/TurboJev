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
