"""Run packaged Unix inference checks on a matching host or emulator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

from native_assets import architecture, targets, verify_machine

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run(list(map(str, arguments)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rid")
    args = parser.parse_args()
    target = next(item for item in targets() if item["rid"] == args.rid)
    expected_os = "Darwin" if target["os"] == "darwin" else "Linux"
    actual_arch = "x86" if struct.calcsize("P") == 4 else "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x64"
    if platform.system() != expected_os or architecture(target) != actual_arch:
        raise SystemExit(f"Run {args.rid} validation on its matching OS and interpreter architecture")
    output = ROOT / "artifacts/unix" / args.rid
    evidence_path = output / "build-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    for name, expected_hash in evidence["runtime_sha256"].items():
        library = output / "native" / name
        data = library.read_bytes()
        verify_machine(data, actual_arch, name)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise SystemExit(f"Runtime hash changed before execution: {name}")
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one wheel")
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        os.environ.pop(name, None)
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheels[0])
    run(sys.executable, "scripts/smoke-python-package.py")
    plan = json.loads((ROOT / "tests/e2e/final-plan.json").read_text())
    model = ROOT / ".cache/real-validation" / plan["model"]["filename"]
    if not model.is_file():
        run(sys.executable, "scripts/download_model.py", "--plan", "tests/e2e/final-plan.json")
    run(sys.executable, "tests/e2e/python/real_smoke.py", model, "--report", output / "python-real-smoke.json")
    if shutil.which("node"):
        run("node", "tests/e2e/node/real_smoke.mjs", model, output / "node/turbojev.node")
    evidence["python_real_smoke_passed"] = True
    evidence["node_real_smoke_passed"] = bool(shutil.which("node"))
    evidence["execution_environment"] = os.environ.get("TURBOJEV_EXECUTION_ENVIRONMENT", "matching host")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Packaged real-model execution passed: {args.rid}")


if __name__ == "__main__":
    main()
