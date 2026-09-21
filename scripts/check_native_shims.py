#!/usr/bin/env python3
"""Compile and execute native adapter shim smoke tests against ABI stubs."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "crates/turbojev-runtime-llamacpp/native"
SHIM = NATIVE / "turbojev_llama_shim.cpp"
STUB_DIR = NATIVE / "tests/llama_stub"
STUB_IMPL = STUB_DIR / "llama_stub.cpp"
SMOKE = NATIVE / "tests/shim_smoke.cpp"


def main() -> None:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        print("SKIP: no C++ compiler available for native shim smoke test")
        return

    with tempfile.TemporaryDirectory(prefix="turbojev-shim-") as temp:
        executable = Path(temp) / "shim-smoke"
        command = [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(STUB_DIR),
            "-I",
            str(NATIVE),
            str(SHIM),
            str(STUB_IMPL),
            str(SMOKE),
            "-o",
            str(executable),
        ]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], check=True)
    print(f"OK: llama.cpp native shim compiles, links, and passes ABI smoke test with {Path(compiler).name}")


if __name__ == "__main__":
    main()
