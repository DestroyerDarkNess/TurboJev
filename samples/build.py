"""Build real TurboJev sample dependencies. No prebuilt mock engines."""

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
sys.path.insert(0, str(ROOT / "scripts"))


def run(*args, cwd=ROOT, env=None):
    command = [str(x) for x in args]
    if os.name == "nt" and command[0] == "npm":
        command[0] = "npm.cmd"
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def host():
    system = {"Windows": "win", "Linux": "linux", "Darwin": "osx"}[platform.system()]
    arch = (
        "x86"
        if struct.calcsize("P") == 4
        else ("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64")
    )
    return system, arch


def native(args):
    system, current = host()
    arch = args.arch or current
    if arch != current:
        raise SystemExit(
            "Build on a matching architecture: Node and Python must match the native libraries. Cross-compilation is not silently assumed."
        )
    targets = {
        ("win", "x64"): "x86_64-pc-windows-msvc",
        ("win", "x86"): "i686-pc-windows-msvc",
        ("win", "arm64"): "aarch64-pc-windows-msvc",
        ("linux", "x64"): "x86_64-unknown-linux-gnu",
        ("linux", "x86"): "i686-unknown-linux-gnu",
        ("linux", "arm64"): "aarch64-unknown-linux-gnu",
        ("osx", "x64"): "x86_64-apple-darwin",
        ("osx", "arm64"): "aarch64-apple-darwin",
    }
    target = targets.get((system, arch))
    if target is None:
        raise SystemExit(f"Unsupported native target: {system}-{arch}")
    node_arch = subprocess.check_output(["node", "-p", "process.arch"], text=True).strip()
    if node_arch != {"x86": "ia32", "x64": "x64", "arm64": "arm64"}[arch]:
        raise SystemExit(
            "Install/run Node with the same architecture as Python before building all samples."
        )
    rid = f"{system}-{arch}"
    stage = SAMPLES / ".artifacts" / rid
    stage.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "tests/e2e/final-plan.json").read_text())
    source = ROOT / ".cache/real-validation/llama.cpp"
    if not source.exists():
        run(
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            plan["llamaCpp"]["tag"],
            plan["llamaCpp"]["repository"],
            source,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if not revision.startswith(plan["llamaCpp"]["commitPrefix"]):
        raise SystemExit("Cached llama.cpp revision differs from the pinned plan.")
    build = (
        Path(args.native_build).resolve()
        if args.native_build
        else ROOT / ".cache/samples" / rid / "native"
    )
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_OPENMP=OFF", f"-DLLAMA_CPP_SOURCE_DIR={source}"]
    if system == "win" and not args.native_build:
        flags += ["-A", {"x86": "Win32", "x64": "x64", "arm64": "ARM64"}[arch]]
    elif system != "win" and arch == "x86":
        flags += ["-DCMAKE_C_FLAGS=-m32", "-DCMAKE_CXX_FLAGS=-m32"]
    run("cmake", "-S", ROOT / "crates/turbojev-runtime-llamacpp/native", "-B", build, *flags)
    run(
        "cmake",
        "--build",
        build,
        "--config",
        "Release",
        "--target",
        "turbojev_llama_shim",
        "-j",
        "4",
    )
    for pattern in ("*.dll", "*.lib", "*.so", "*.so.*", "*.dylib"):
        for file in build.rglob(pattern):
            if file.is_file():
                shutil.copy2(file, stage / file.name)
    env = os.environ.copy()
    env["TURBOJEV_LLAMA_SHIM_LIB_DIR"] = str(stage)
    env["PYO3_PYTHON"] = sys.executable
    if system != "win":
        loader = "@loader_path" if system == "osx" else "$ORIGIN"
        env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -C link-arg=-Wl,-rpath,{loader}".strip()
    run("rustup", "target", "add", target)
    profile = [] if args.debug else ["--release"]
    rust_host = (
        subprocess.check_output(["rustc", "-vV"], text=True).split("host: ")[1].splitlines()[0]
    )
    target_flags = [] if target == rust_host else ["--target", target]
    run(
        "cargo",
        "build",
        "--locked",
        *target_flags,
        *profile,
        "-p",
        "turbojev-node",
        "-p",
        "turbojev-python",
        "-p",
        "turbojev-capi",
        "-p",
        "turbojev-rust-sample",
        "--features",
        "turbojev-node/llamacpp,turbojev-python/llamacpp,turbojev-capi/llamacpp,turbojev-rust-sample/native",
        env=env,
    )
    output = ROOT / "target"
    if target_flags:
        output /= target
    output /= "debug" if args.debug else "release"
    prefix, suffix = (
        ("", ".dll") if system == "win" else ("lib", ".dylib" if system == "osx" else ".so")
    )
    shutil.copy2(output / f"{prefix}turbojev_capi{suffix}", stage)
    shutil.copy2(output / f"{prefix}turbojev_node{suffix}", stage / "turbojev.node")
    py = stage / "python/turbojev"
    py.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "bindings/python/turbojev/__init__.py", py)
    shutil.copy2(
        output / f"{prefix}_native{suffix}",
        py / ("_native.pyd" if system == "win" else "_native.abi3.so"),
    )
    for pattern in ("*.dll", "*.so", "*.so.*", "*.dylib"):
        for library in stage.glob(pattern):
            shutil.copy2(library, py)
    shutil.copy2(
        output / ("turbojev-rust-sample.exe" if system == "win" else "turbojev-rust-sample"), stage
    )
    run("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund", cwd=ROOT / "bindings/node")
    run("npm", "run", "build", cwd=ROOT / "bindings/node")
    node = stage / "node"
    shutil.copytree(ROOT / "bindings/node/dist", node / "dist", dirs_exist_ok=True)
    for name in ("worker.mjs", "package.json"):
        shutil.copy2(ROOT / "bindings/node" / name, node)
    run(
        "dotnet",
        "build",
        ROOT / "bindings/dotnet/TurboJev/TurboJev.csproj",
        "-c",
        "Release",
        "-o",
        stage / "dotnet",
    )
    run(
        "dotnet",
        "build",
        SAMPLES / "TurboJev-csharp-NET/Sample.csproj",
        "-c",
        "Release",
        f"-p:TurboJevRid={rid}",
    )
    run(
        "npm",
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        cwd=SAMPLES / "TurboJev-TS-node",
    )
    run("npm", "run", "build", cwd=SAMPLES / "TurboJev-TS-node")
    (stage / "build-info.json").write_text(
        json.dumps(
            {
                "rid": rid,
                "target": target,
                "llamaRevision": revision,
                "profile": "debug" if args.debug else "release",
                "python": sys.version,
                "nodeArch": node_arch,
            },
            indent=2,
        )
    )
    print(f"Ready: {stage}")
    print("Run from repository root: python samples/run.py python --model C:/models/model.gguf")
    print("Run from samples/:        py .\\run.py python")


def web():
    sample = SAMPLES / "TurboJev-js-native"
    llama_manifest = json.loads(
        (ROOT / "bindings/web-runtimes/llamacpp-wasm/package.json").read_text()
    )
    run("npm", "view", f"{llama_manifest['name']}@{llama_manifest['version']}", "version")
    cli = (
        ROOT / ".cache/wasm-tools/bin" / ("wasm-bindgen.exe" if os.name == "nt" else "wasm-bindgen")
    )
    if not cli.exists():
        run(
            "cargo",
            "install",
            "wasm-bindgen-cli",
            "--version",
            "0.2.100",
            "--locked",
            "--root",
            ROOT / ".cache/wasm-tools",
        )
    run("rustup", "target", "add", "wasm32-unknown-unknown")
    run(
        "cargo",
        "build",
        "--locked",
        "--release",
        "-p",
        "turbojev-wasm",
        "--target",
        "wasm32-unknown-unknown",
    )
    run(
        cli,
        ROOT / "target/wasm32-unknown-unknown/release/turbojev_wasm.wasm",
        "--target",
        "web",
        "--out-dir",
        sample / "public/vendor/wasm",
    )
    run("npm", "install", "--ignore-scripts", "--no-audit", "--no-fund", cwd=sample)
    run("npm", "run", "prepare:runtime", cwd=sample)
    run("npm", "run", "build", cwd=sample)
    print("Ready")
    print("Run from repository root: python samples/run.py web")
    print("Run from samples/:        py .\\run.py web")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["native", "web"])
    parser.add_argument("--arch", choices=["x86", "x64", "arm64"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--native-build", help="Reuse a CMake build for this exact architecture")
    options = parser.parse_args()
    try:
        native(options) if options.kind == "native" else web()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
