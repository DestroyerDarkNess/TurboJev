"""Build and validate complete Unix native release artifacts on a matching host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from native_assets import architecture, node_package, required_runtime, targets, verify_machine

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rid", choices=[target["rid"] for target in targets() if target["os"] != "win32"])
    parser.add_argument("--cross-build", action="store_true", help="Package with a supplied cross toolchain; requires a separate real execution check")
    args = parser.parse_args()
    target = next(target for target in targets() if target["rid"] == args.rid)
    arch = architecture(target)
    expected_os = "Darwin" if target["os"] == "darwin" else "Linux"
    if platform.system() != expected_os:
        raise SystemExit(f"{args.rid} must be built on {expected_os}, not {platform.system()}")

    version = json.loads((ROOT / "bindings/node/package.json").read_text())["version"]
    output = ROOT / "artifacts/unix" / args.rid
    native = output / "native"
    native.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "tests/e2e/final-plan.json").read_text())
    source = Path(os.environ.get("TURBOJEV_LLAMA_SOURCE_DIR", ROOT / ".cache/real-validation/llama.cpp"))
    if not source.exists():
        run("git", "clone", "--depth", "1", "--branch", plan["llamaCpp"]["tag"], plan["llamaCpp"]["repository"], source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if not revision.startswith(plan["llamaCpp"]["commitPrefix"]):
        raise SystemExit("llama.cpp checkout does not match the release plan")

    build = Path(os.environ.get("TURBOJEV_NATIVE_BUILD_DIR", output / "cmake"))
    cmake = ["cmake", "-S", ROOT / "crates/turbojev-runtime-llamacpp/native", "-B", build,
             "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_CPP_SOURCE_DIR={source}", "-DGGML_OPENMP=OFF",
             "-DGGML_VULKAN=OFF", "-DGGML_NATIVE=OFF", "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF",
             "-DGGML_FMA=OFF", "-DGGML_F16C=OFF", "-DGGML_BMI2=OFF", "-DGGML_AVX512=OFF"]
    if os.environ.get("TURBOJEV_CMAKE_TOOLCHAIN"):
        cmake.append(f"-DCMAKE_TOOLCHAIN_FILE={os.environ['TURBOJEV_CMAKE_TOOLCHAIN']}")
    if target["os"] == "darwin":
        cmake += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DGGML_BLAS=OFF"]
    else:
        # Keep distribution-specific libstdc++ and libgcc out of the dependency closure.
        cmake += ["-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc"]
    run(*cmake)
    run("cmake", "--build", build, "--target", "turbojev_llama_shim", "--parallel", "4")
    for path in build.rglob("*"):
        if path.is_file() and (path.name.endswith(".dylib") or ".so" in path.name):
            shutil.copyfile(path, native / path.name)
    missing = required_runtime(target["os"], arch) - {path.name for path in native.iterdir()}
    if missing:
        raise SystemExit(f"Missing runtime dependencies: {sorted(missing)}")

    env = dict(os.environ)
    env["TURBOJEV_LLAMA_SHIM_LIB_DIR"] = str(native)
    env["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    rpath = "@loader_path" if target["os"] == "darwin" else "$ORIGIN"
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -C link-arg=-Wl,-rpath,{rpath}".strip()
    env["LD_LIBRARY_PATH"] = str(native)
    run("rustup", "target", "add", target["rust"])
    run("cargo", "build", "--locked", "--release", "--target", target["rust"], "-p", "turbojev-capi",
        "-p", "turbojev-node", "--features", "turbojev-capi/llamacpp,turbojev-node/llamacpp", env=env)
    cargo = Path(env.get("CARGO_TARGET_DIR", ROOT / "target")) / target["rust"] / "release"
    suffix = ".dylib" if target["os"] == "darwin" else ".so"
    shutil.copyfile(cargo / ("libturbojev_capi" + suffix), native / ("libturbojev_capi" + suffix))

    package = output / "node"
    package.mkdir(exist_ok=True)
    for path in native.iterdir():
        shutil.copyfile(path, package / path.name)
    shutil.copyfile(cargo / ("libturbojev_node" + suffix), package / "turbojev.node")
    manifest = {"name": node_package(target), "version": version, "main": "turbojev.node",
                "os": [target["os"]], "cpu": [target["arch"]], "license": "Apache-2.0",
                "engines": {"node": ">=18"}, "repository": "https://github.com/DestroyerDarkNess/TurboJev"}
    (package / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(sys.executable, "scripts/check-native-node-package.py", package)

    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    run(sys.executable, "-m", "maturin", "build", "--release", "--locked", "--target", target["rust"],
        "--features", "llamacpp", "--out", wheels, "--auditwheel", "skip", env=env, cwd=ROOT / "bindings/python")
    for wheel in wheels.glob("*.whl"):
        if target["os"] == "linux":
            # Check the libc symbol floor below before assigning the manylinux tag.
            run(sys.executable, "scripts/retag-native-wheel.py", wheel, target["wheel"])
    wheel_files = list(wheels.glob("*.whl"))
    if len(wheel_files) != 1:
        raise SystemExit("Expected exactly one native Python wheel")
    wheel = wheel_files[0]
    run(sys.executable, "scripts/bundle-native-wheel.py", wheel, native)
    run(sys.executable, "scripts/check-native-python-wheel.py", wheel)
    for path in native.iterdir():
        verify_machine(path.read_bytes(), arch, path.name)
        if target["os"] == "darwin":
            linked = subprocess.check_output(["otool", "-L", str(path)], text=True)
            for line in linked.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                if dependency.startswith(("/System/Library/", "/usr/lib/")):
                    continue
                if not dependency.startswith(("@rpath/", "@loader_path/")):
                    raise SystemExit(f"{path.name}: non-relocatable dependency {dependency}")
                if Path(dependency).name not in {entry.name for entry in native.iterdir()}:
                    raise SystemExit(f"{path.name}: unbundled dependency {dependency}")
    if target["os"] == "linux":
        # auditwheel refuses a false manylinux tag when symbol versions exceed its floor.
        run(sys.executable, "-m", "auditwheel", "show", wheel)
        run(sys.executable, "scripts/check-elf-dependencies.py", native)
    if args.cross_build:
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in native.iterdir()}
        evidence = {"version": version, "rid": args.rid, "llama_cpp_revision": revision,
                    "runtime_sha256": hashes, "cross_compiled": True, "link_checks_passed": True,
                    "python_real_smoke_passed": False, "node_real_smoke_passed": False}
        (output / "build-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print("Cross-build complete; a matching host/emulator must run the package checks before publication.")
        return
    # Hide every original CMake dependency and remove loader environment overrides.
    # The installed wheel and Node package must find only their own sibling libraries.
    shutil.move(str(build), str(output / "cmake-hidden"))
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("DYLD_LIBRARY_PATH", None)
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel)
    run(sys.executable, "scripts/smoke-python-package.py")
    run(sys.executable, "scripts/download_model.py", "--plan", "tests/e2e/final-plan.json")
    model = ROOT / ".cache/real-validation" / plan["model"]["filename"]
    run(sys.executable, "tests/e2e/python/real_smoke.py", model, "--report", output / "python-real-smoke.json")
    if shutil.which("node"):
        run("npm", "install", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", cwd=ROOT / "bindings/node")
        run("npm", "run", "build", cwd=ROOT / "bindings/node")
        run("node", "tests/e2e/node/real_smoke.mjs", model, package / "turbojev.node")

    artifacts = {}
    for path in native.iterdir():
        verify_machine(path.read_bytes(), arch, path.name)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {"version": version, "rid": args.rid, "llama_cpp_revision": revision,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "runtime_sha256": artifacts, "python_real_smoke_passed": True,
                "node_real_smoke_passed": bool(shutil.which("node"))}
    (output / "build-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
