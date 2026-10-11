"""Bundle architecture-matched inference libraries and licenses into a wheel."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path
from native_assets import architecture, required_runtime, verify_machine, wheel_target
from runtime_licenses import license_entries


def digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bundle-native-wheel.py WHEEL NATIVE_LIBRARY_DIRECTORY")
    wheel = Path(sys.argv[1])
    native = Path(sys.argv[2])
    target = wheel_target(wheel.name)
    arch = architecture(target)
    dlls = sorted(path for path in native.iterdir() if path.is_file() and
                  (path.name.endswith((".dll", ".dylib")) or ".so" in path.name))
    required = required_runtime(target["os"], arch)
    missing = required - {path.name.lower() for path in dlls}
    if missing:
        raise SystemExit(f"Native directory {native} is missing: {', '.join(sorted(missing))}")
    for dll in dlls:
        verify_machine(dll.read_bytes(), arch, dll.name)

    temporary = wheel.with_suffix(".bundling.whl")
    with zipfile.ZipFile(wheel, "r") as source:
        entries = [(item, source.read(item.filename)) for item in source.infolist() if not item.filename.endswith(".dist-info/RECORD")]
        record_name = next(item.filename for item in source.infolist() if item.filename.endswith(".dist-info/RECORD"))
        record_bytes = source.read(record_name)

    rows = [row for row in csv.reader(io.StringIO(record_bytes.decode("utf-8"))) if row[0] != record_name]
    extensions = [(item, data) for item, data in entries if item.filename.startswith("turbojev/_native") and item.filename.endswith((".pyd", ".so"))]
    if not extensions:
        raise SystemExit("Wheel is missing its native Python extension")
    for item, data in extensions:
        verify_machine(data, arch, item.filename)
    names = {item.filename for item, _ in entries}
    additions: list[tuple[str, bytes]] = []
    for dll in dlls:
        archive_name = f"turbojev/{dll.name}"
        if archive_name in names:
            continue
        data = dll.read_bytes()
        additions.append((archive_name, data))
        rows.append([archive_name, digest(data), str(len(data))])
    for name, license_path in license_entries().items():
        archive_name = "turbojev/" + name
        if archive_name not in names:
            data = license_path.read_bytes()
            additions.append((archive_name, data))
            rows.append([archive_name, digest(data), str(len(data))])
    bundled_names = {item.filename.lower() for item, _ in entries} | {
        name.lower() for name, _ in additions
    }
    missing = {f"turbojev/{name}" for name in required} - bundled_names
    if missing:
        raise SystemExit(f"Wheel failed to include required runtime DLLs: {', '.join(sorted(missing))}")
    rows.append([record_name, "", ""])
    record_output = io.StringIO(newline="")
    csv.writer(record_output, lineterminator="\n").writerows(rows)

    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for item, data in entries:
            output.writestr(item, data)
        for name, data in additions:
            output.writestr(name, data)
        output.writestr(record_name, record_output.getvalue())
    temporary.replace(wheel)
    print(f"Bundled native libraries and runtime notices into {wheel.name}")


if __name__ == "__main__":
    main()
