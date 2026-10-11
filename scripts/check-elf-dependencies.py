"""Reject Unix runtime packages with missing libraries or excessive libc requirements."""
from pathlib import Path
import re
import subprocess
import sys


def main():
    directory = Path(sys.argv[1])
    system = {"libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "libgcc_s.so.1",
              "ld-linux-x86-64.so.2", "ld-linux.so.2", "ld-linux-aarch64.so.1"}
    names = {path.name for path in directory.iterdir()}
    for path in directory.iterdir():
        if ".so" not in path.name:
            continue
        output = subprocess.check_output(["readelf", "-d", "-V", str(path)], text=True)
        for dependency in re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', output):
            if dependency not in system and dependency not in names:
                raise SystemExit(f"{path.name}: unbundled dependency {dependency}")
        for major, minor in re.findall(r'\bGLIBC_(\d+)\.(\d+)', output):
            if (int(major), int(minor)) > (2, 28):
                raise SystemExit(f"{path.name}: GLIBC_{major}.{minor} exceeds the glibc 2.28 floor")
        if "$ORIGIN" not in output:
            raise SystemExit(f"{path.name}: missing relocatable runtime search path")
    print("ELF dependency closure and glibc 2.28 floor verified")


if __name__ == "__main__":
    main()
