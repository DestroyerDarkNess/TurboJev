"""Validation tests; no model downloads or GPU required."""
import unittest
from jev_omni_worker import decode_evidence


class MediaTests(unittest.TestCase):
    def test_all_modalities_preserve_bytes(self):
        for modality in ("image", "audio", "video"):
            self.assertEqual(decode_evidence({"contract": "turbojev.media.v1", "modality": modality,
                                             "text": "context", "base64": "AP8="}),
                             ("context", modality, b"\x00\xff"))

    def test_invalid_base64_and_paths_are_rejected(self):
        for extra in ({"base64": "??"}, {"path": "/secret"}, {"modality": "pdf"}, {"base64": ""}):
            state = {"contract": "turbojev.media.v1", "modality": "image", "base64": "AA==", **extra}
            with self.assertRaises(ValueError):
                decode_evidence(state)


if __name__ == "__main__":
    unittest.main()
