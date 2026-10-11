"""Reject wheels without a complete, matching runtime and valid wheel metadata."""
import sys
import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

from native_assets import architecture, required_runtime, verify_machine, wheel_target


def main() -> None:
    wheel = Path(sys.argv[1])
    target = wheel_target(wheel.name)
    arch = architecture(target)
    if "-cp39-abi3-" not in wheel.name:
        raise SystemExit("Expected a CPython 3.9+ stable-ABI wheel")
    required = required_runtime(target["os"], arch)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise SystemExit("Wheel contains duplicate archive entries")
        record = next(name for name in names if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        if len(rows) != len({row[0] for row in rows}) or {row[0] for row in rows} != set(names):
            raise SystemExit("Wheel RECORD must list every file exactly once")
        for name, expected_hash, expected_size in rows:
            if name == record:
                continue
            data = archive.read(name)
            digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if expected_hash != digest or expected_size != str(len(data)):
                raise SystemExit(f"Wheel RECORD differs for {name}")
        missing = {"turbojev/" + name for name in required} - set(names)
        if missing:
            raise SystemExit("Wheel lacks native inference dependencies: " + ", ".join(sorted(missing)))
        extensions = [name for name in names if name.startswith("turbojev/_native") and name.endswith((".pyd", ".so"))]
        if not extensions:
            raise SystemExit("Wheel lacks the Python native extension")
        for name in names:
            if name.startswith("turbojev/") and (name.endswith((".dll", ".pyd", ".dylib")) or ".so" in name):
                verify_machine(archive.read(name), arch, name)
    print(f"Native Python wheel check passed: {wheel.name}")


if __name__ == "__main__":
    main()
