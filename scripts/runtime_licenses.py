"""License texts that accompany the bundled native inference libraries."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def license_entries():
    source = ROOT / ".cache/real-validation/llama.cpp"
    entries = {
        "LICENSE": ROOT / "LICENSE",
        "NOTICE": ROOT / "NOTICE",
        "licenses/llama.cpp-LICENSE": source / "LICENSE",
    }
    for path in (source / "vendor").rglob("*"):
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
            entries["licenses/llama.cpp-vendor/" + path.relative_to(source / "vendor").as_posix()] = path
    return entries


def copy_licenses(directory):
    for name, source in license_entries().items():
        destination = Path(directory) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
