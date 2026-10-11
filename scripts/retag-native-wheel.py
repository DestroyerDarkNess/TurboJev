"""Set the platform tag and RECORD for a wheel built in a policy-matching image."""
import csv
import io
from pathlib import Path
import sys
import zipfile

from bundle_native_wheel_support import digest


def main():
    wheel = Path(sys.argv[1])
    platform = sys.argv[2]
    destination = wheel.with_name(wheel.name.rsplit("-", 1)[0] + "-" + platform + ".whl")
    with zipfile.ZipFile(wheel) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in data if name.endswith(".dist-info/WHEEL"))
    lines = data[metadata].decode().splitlines()
    data[metadata] = ("\n".join("Tag: cp39-abi3-" + platform if line.startswith("Tag:") else line for line in lines) + "\n").encode()
    record = next(name for name in data if name.endswith(".dist-info/RECORD"))
    stream = io.StringIO(newline="")
    rows = [[name, digest(value), len(value)] for name, value in data.items() if name != record]
    rows.append([record, "", ""])
    csv.writer(stream, lineterminator="\n").writerows(rows)
    data[record] = stream.getvalue().encode()
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in data.items():
            archive.writestr(name, value)
    temporary.replace(destination)
    if wheel != destination:
        wheel.unlink()


if __name__ == "__main__":
    main()
