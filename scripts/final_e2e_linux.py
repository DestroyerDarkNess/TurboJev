#!/usr/bin/env python3
"""One-command Linux release qualification with a real small open-weight GGUF.

This script is intentionally host-only. It downloads/pins the external artifacts,
builds the llama.cpp shim and TurboJev native bindings, produces fused-runtime
evidence, and executes the same request through Node, Python, C and .NET.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def capture(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def require_toolchain_versions(plan: dict[str, Any]) -> None:
    rustc = capture(["rustc", "--version"])
    if not rustc.startswith(f"rustc {plan['rust']}"):
        raise SystemExit(f"final E2E requires Rust {plan['rust']}; found {rustc}")

    node = capture(["node", "--version"]).lstrip("v")
    if node.split(".", 1)[0] != str(plan["node"]):
        raise SystemExit(f"final E2E requires Node {plan['node']}.x; found {node}")

    python = capture(["python3", "--version"]).removeprefix("Python ")
    if not python.startswith(str(plan["python"]) + "."):
        raise SystemExit(f"final E2E requires Python {plan['python']}.x; found {python}")

    dotnet = capture(["dotnet", "--version"])
    if not dotnet.startswith(str(plan["dotnet"]) + "."):
        raise SystemExit(f"final E2E requires .NET SDK {plan['dotnet']}.x; found {dotnet}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_tools(names: list[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise SystemExit("missing final-E2E tools: " + ", ".join(missing))


def hardware_identity() -> tuple[str, str]:
    """Return a stable-enough CPU device + kernel identity for the pinned CPU qualification host."""
    device = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        fields: dict[str, str] = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            key, sep, value = line.partition(":")
            if sep and value.strip():
                fields.setdefault(key.strip().lower(), value.strip())
        device = fields.get("model name") or fields.get("hardware") or device
        processor = fields.get("processor")
        if not device and processor and not processor.isdigit():
            device = processor
    if not device:
        device = platform.machine() or "unknown-cpu"
    return device, f"kernel:{platform.release()}"


def download(url: str, target: Path, expected_sha: str) -> None:
    if target.exists() and sha256(target) == expected_sha:
        print(f"model cache hit: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "TurboJev-final-e2e/1"})
    print(f"downloading {url}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = sha256(temporary)
    if actual != expected_sha:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"model SHA-256 mismatch: expected {expected_sha}, got {actual}")
    temporary.replace(target)


def copy_shared_libraries(build_dir: Path, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    candidates: set[Path] = set()
    for pattern in ("lib*.so", "lib*.so.*", "*.dylib"):
        candidates.update(path for path in build_dir.rglob(pattern) if path.is_file())
    for source in sorted(candidates):
        shutil.copy2(source, stage / source.name)
    if not any(path.name.startswith("libturbojev_llama_shim") for path in stage.iterdir()):
        raise SystemExit("CMake build did not produce turbojev_llama_shim")
    if not any(path.name.startswith("libllama") for path in stage.iterdir()):
        raise SystemExit("CMake build did not produce libllama")
    if not any(path.name.startswith("libmtmd") for path in stage.iterdir()):
        raise SystemExit("CMake build did not produce libmtmd; multimodal runtime dependency is missing")


def copy_required(path: Path, destination: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing build artifact: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def write_final_report(
    target: Path,
    plan: dict[str, Any],
    plan_path: Path,
    model: Path,
    runtime_revision: str,
    evidence: Path,
    qualification: Path,
    language_report: Path,
    cargo_lock: Path,
    node_lock: Path,
) -> bool:
    evidence_json = json.loads(evidence.read_text())
    qualification_json = json.loads(qualification.read_text())
    language_json = json.loads(language_report.read_text())
    passed = bool(qualification_json.get("summary", {}).get("passed")) and bool(language_json.get("passed"))
    result = {
        "contract": "turbojev.final-validation-report.v2",
        "passed": passed,
        "planSha256": sha256(plan_path),
        "model": {
            "repository": plan["model"]["repository"],
            "filename": model.name,
            "sha256": sha256(model),
            "quantization": plan["model"]["quantization"],
            "sizeBytes": model.stat().st_size,
        },
        "runtime": {
            "llamaCppTag": plan["llamaCpp"]["tag"],
            "llamaCppRevision": runtime_revision,
            "backend": plan["llamaCpp"]["backend"],
            "shimAbi": 5,
        },
        "qualification": {
            "passed": qualification_json.get("summary", {}).get("passed", False),
            "evidenceSha256": sha256(evidence),
            "manifestSha256": sha256(qualification),
            "summary": evidence_json,
        },
        "languageMatrix": {
            "passed": language_json.get("passed", False),
            "semanticParity": language_json.get("semanticParity", False),
            "reportSha256": sha256(language_report),
            "results": language_json.get("results", []),
        },
        "dependencyResolution": {
            "cargoLockSha256": sha256(cargo_lock),
            "nodePackageLockSha256": sha256(node_lock),
        },
        "toolchains": {
            "rustc": capture(["rustc", "--version"]),
            "cargo": capture(["cargo", "--version"]),
            "node": capture(["node", "--version"]),
            "python": capture(["python3", "--version"]),
            "dotnet": capture(["dotnet", "--version"]),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return passed


def main() -> int:
    if sys.platform != "linux":
        raise SystemExit("final_e2e_linux.py intentionally targets Linux; desktop OS matrix remains in CI")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=ROOT / "tests/e2e/final-plan.json")
    ap.add_argument("--work-dir", type=Path, default=ROOT / ".e2e-final")
    ap.add_argument("--model", type=Path, help="Use an already downloaded GGUF instead of fetching the pinned model")
    ap.add_argument("--report", type=Path, default=ROOT / "artifacts/final-validation-report.json")
    ap.add_argument("--skip-workspace-gates", action="store_true")
    args = ap.parse_args()

    require_tools(["git", "cmake", "c++", "cc", "cargo", "rustc", "rustfmt", "node", "npm", "python3", "dotnet"])
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    require_toolchain_versions(plan)

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve() if args.model else work / "models" / plan["model"]["filename"]
    if args.model:
        actual = sha256(model)
        if actual != plan["model"]["sha256"]:
            raise SystemExit(f"provided model SHA-256 mismatch: {actual}")
    else:
        download(plan["model"]["url"], model, plan["model"]["sha256"])
    if model.stat().st_size != int(plan["model"]["sizeBytes"]):
        raise SystemExit(
            f"model size mismatch: expected {plan['model']['sizeBytes']}, got {model.stat().st_size}"
        )

    llama_src = work / "llama.cpp"
    if not llama_src.exists():
        run(["git", "clone", "--depth", "1", "--branch", plan["llamaCpp"]["tag"], plan["llamaCpp"]["repository"], str(llama_src)])
    runtime_revision = capture(["git", "rev-parse", "HEAD"], cwd=llama_src)
    if not runtime_revision.startswith(plan["llamaCpp"]["commitPrefix"]):
        raise SystemExit(f"llama.cpp revision mismatch: expected prefix {plan['llamaCpp']['commitPrefix']}, got {runtime_revision}")

    native_build = work / "native-build"
    run([
        "cmake", "-S", str(ROOT / "crates/turbojev-runtime-llamacpp/native"), "-B", str(native_build),
        f"-DLLAMA_CPP_SOURCE_DIR={llama_src}", "-DCMAKE_BUILD_TYPE=Release",
    ])
    run(["cmake", "--build", str(native_build), "--config", "Release", "--target", "turbojev_llama_shim", "-j2"])
    stage = work / "native"
    if stage.exists():
        shutil.rmtree(stage)
    copy_shared_libraries(native_build, stage)

    env = os.environ.copy()
    env["TURBOJEV_LLAMA_SHIM_LIB_DIR"] = str(stage)
    env["LD_LIBRARY_PATH"] = str(stage) + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    dependency_dir = work / "dependency-locks"
    dependency_dir.mkdir(parents=True, exist_ok=True)
    cargo_lock = ROOT / "Cargo.lock"
    if not cargo_lock.exists():
        run(["cargo", "generate-lockfile"], env=env)
    run(["cargo", "metadata", "--locked", "--format-version", "1", "--no-deps"], env=env)
    frozen_cargo_lock = dependency_dir / "Cargo.lock"
    shutil.copy2(cargo_lock, frozen_cargo_lock)

    if not args.skip_workspace_gates:
        run(["cargo", "fmt", "--all", "--check"], env=env)
        run(["cargo", "check", "--locked", "--workspace", "--all-targets"], env=env)
        run(["cargo", "clippy", "--locked", "--workspace", "--all-targets", "--", "-D", "warnings"], env=env)
        run(["cargo", "test", "--locked", "--workspace"], env=env)

    for package in ("turbojev-capi", "turbojev-node", "turbojev-python"):
        run(["cargo", "build", "--locked", "--release", "-p", package, "--features", "llamacpp"], env=env)
    release = ROOT / "target/release"
    copy_required(release / "libturbojev_capi.so", stage / "libturbojev_capi.so")
    copy_required(release / "libturbojev_node.so", stage / "turbojev.node")

    python_root = work / "python"
    package_dir = python_root / "turbojev"
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "bindings/python/turbojev/__init__.py", package_dir / "__init__.py")
    copy_required(release / "lib_native.so", package_dir / "_native.abi3.so")

    node_root = ROOT / "bindings/node"
    node_lock = node_root / "package-lock.json"
    if not node_lock.exists():
        run(["npm", "install", "--package-lock-only", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=node_root)
    run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=node_root)
    frozen_node_lock = dependency_dir / "node-package-lock.json"
    shutil.copy2(node_lock, frozen_node_lock)
    run(["npm", "run", "build"], cwd=node_root)
    run(["dotnet", "build", str(ROOT / "tests/e2e/dotnet/TurboJev.E2E.csproj"), "-c", "Release"])

    evidence = work / "qualification-evidence.json"
    run([
        "cargo", "run", "--locked", "--release", "-p", "turbojev-runtime-llamacpp", "--features", "ffi",
        "--example", "real_fused_qualification", "--", str(model), str(evidence), str(plan["qualification"]["rounds"]),
    ], env=env)
    fingerprint = work / "runtime-fingerprint.json"
    device, driver = hardware_identity()
    run([
        "python3", str(ROOT / "scripts/runtime_qualification.py"), "fingerprint",
        "--model", str(model), "--runtime-revision", runtime_revision,
        "--backend", plan["llamaCpp"]["backend"], "--device", device, "--driver", driver,
        "--quantization", plan["model"]["quantization"], "--output", str(fingerprint),
    ], env=env)
    qualification = work / "runtime-qualification.json"
    run([
        "python3", str(ROOT / "scripts/runtime_qualification.py"), "manifest",
        "--fingerprint", str(fingerprint), "--evidence", str(evidence), "--id", "smollm2-135m-cpu-final",
        "--absolute-tolerance", str(plan["qualification"]["absoluteTolerance"]),
        "--relative-tolerance", str(plan["qualification"]["relativeTolerance"]),
        "--max-fused-items", str(plan["qualification"]["maxFusedItems"]), "--output", str(qualification),
    ], env=env)

    language_report = work / "e2e-language-report.json"
    run([
        "python3", str(ROOT / "scripts/e2e_language_matrix.py"), "--model", str(model),
        "--native-lib-dir", str(stage), "--node-addon", str(stage / "turbojev.node"),
        "--python-root", str(python_root), "--request", str(ROOT / plan["request"]),
        "--report", str(language_report),
    ], env=env)
    passed = write_final_report(
        args.report.resolve(), plan, plan_path, model, runtime_revision, evidence, qualification, language_report,
        frozen_cargo_lock, frozen_node_lock,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
