"""Host-side image encoding. The Rust core never reads files or networks."""

import base64
import io
from pathlib import Path
from typing import Any


def _is_file(value: str) -> bool:
    try:
        return Path(value).is_file()
    except OSError:
        return False


def encode_image(image: Any) -> str:
    """Encode bytes, a path, a base64/data URI string or a PIL-like image.

    URLs are intentionally not fetched implicitly. Fetch them explicitly and pass bytes.
    """
    if isinstance(image, str) and image.startswith("data:image/"):
        if ";base64," not in image:
            raise ValueError("Image data URI must use base64")
        value = image.split(",", 1)[1]
    elif isinstance(image, str) and "://" in image:
        raise ValueError("Fetch image URLs explicitly and pass the downloaded bytes")
    elif isinstance(image, Path) or isinstance(image, str) and _is_file(image):
        value = base64.b64encode(Path(image).read_bytes()).decode("ascii")
    elif isinstance(image, (bytes, bytearray, memoryview)):
        value = base64.b64encode(image).decode("ascii")
    elif isinstance(image, str):
        value = image
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        value = base64.b64encode(buffer.getvalue()).decode("ascii")
    else:
        raise TypeError("Image must be bytes, a path, base64/data URI or a PIL-like image")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Invalid image base64") from error
    if not payload or len(payload) > 24 * 1024 * 1024:
        raise ValueError("Image must contain 1 byte..24 MiB")
    return value
