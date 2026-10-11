"""Stable wheel RECORD digests shared by native wheel tooling."""
import base64
import hashlib


def digest(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
