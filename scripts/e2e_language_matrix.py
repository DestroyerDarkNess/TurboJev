#!/usr/bin/env python3
"""Run one real TurboJev request through every native binding and prove parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_runner_stdout(stdout: str, identifier: str, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = stdout.strip()
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{identifier} stdout is not one JSON document: {error}") from error
    if not isinstance(envelope, dict) or envelope.get("language") != identifier:
        raise ValueError(f"{identifier} runner emitted the wrong language envelope")
    response = envelope.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{identifier} runner did not emit a response object")
    validate_typed_response(response, identifier, request)
    return envelope, response


def validate_typed_response(response: dict[str, Any], identifier: str, request: dict[str, Any]) -> None:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(f"{identifier}: missing answers object")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("E2E request has no typed questions")
    for answer_id, question in questions.items():
        answer = answers.get(answer_id)
        expected = question.get("type") if isinstance(question, dict) else None
        expected = "noul" if expected == "boolean" else expected
        if not isinstance(answer, dict) or answer.get("type") != expected:
            raise ValueError(f"{identifier}: invalid typed answer for {answer_id}")
        field = {"choice": "choice", "noul": "noul", "score": "score"}.get(expected)
        value = answer.get(field) if field else None
        if expected == "choice" and not isinstance(value, str):
            raise ValueError(f"{identifier}: {answer_id} has no choice value")
        if expected in {"noul", "score"} and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{identifier}: {answer_id} has no numeric {expected} value")
    for answer_id, answer in answers.items():
        if not isinstance(answer, dict):
            raise ValueError(f"{identifier}: {answer_id} is not an object")
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict) or not probabilities:
                raise ValueError(f"{identifier}: {answer_id} has invalid probabilities")
            values = list(probabilities.values())
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0 for v in values):
                raise ValueError(f"{identifier}: {answer_id} has non-finite/out-of-range probability")
            if not math.isclose(sum(float(v) for v in values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"{identifier}: {answer_id} probabilities do not sum to 1")
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("generated_tokens") != 0:
        raise ValueError(f"{identifier}: zero-generation invariant failed")
    metadata = response.get("_turbojev")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("contract"), str):
        raise ValueError(f"{identifier}: missing TurboJev metadata")


def equivalent(expected: Any, actual: Any, path: str, mismatches: list[str], abs_tol: float, rel_tol: float) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            mismatches.append(f"{path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        a, b = float(expected), float(actual)
        if not (math.isfinite(a) and math.isfinite(b)) or not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            mismatches.append(f"{path}: {a!r} != {b!r}")
        return
    if type(expected) is not type(actual):
        mismatches.append(f"{path}: type {type(expected).__name__} != {type(actual).__name__}")
        return
    if isinstance(expected, dict):
        ek, ak = set(expected), set(actual)
        if ek != ak:
            mismatches.append(f"{path}: keys {sorted(ek)} != {sorted(ak)}")
            return
        for key in sorted(ek):
            equivalent(expected[key], actual[key], f"{path}.{key}", mismatches, abs_tol, rel_tol)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            equivalent(left, right, f"{path}[{index}]", mismatches, abs_tol, rel_tol)
        return
    if expected != actual:
        mismatches.append(f"{path}: {expected!r} != {actual!r}")


def run_case(identifier: str, command: list[str], env: dict[str, str], request: dict[str, Any], required: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=300)
        duration = (time.perf_counter() - started) * 1000
        stdout = process.stdout.strip()
        if process.returncode != 0:
            return {
                "id": identifier, "status": "fail", "durationMs": round(duration, 3),
                "exitCode": process.returncode, "stdoutSha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "responseSha256": None, "message": process.stderr.strip() or stdout or f"exit {process.returncode}",
                "_response": None,
            }
        try:
            _, response = parse_runner_stdout(stdout, identifier, request)
        except ValueError as error:
            return {
                "id": identifier, "status": "fail", "durationMs": round(duration, 3),
                "exitCode": process.returncode, "stdoutSha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "responseSha256": None, "message": str(error), "_response": None,
            }
        return {
            "id": identifier, "status": "pass", "durationMs": round(duration, 3),
            "exitCode": process.returncode, "stdoutSha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "responseSha256": canonical_sha(response), "message": None, "_response": response,
        }
    except FileNotFoundError as error:
        return {
            "id": identifier, "status": "fail" if required else "skip",
            "durationMs": round((time.perf_counter() - started) * 1000, 3), "exitCode": None,
            "stdoutSha256": None, "responseSha256": None, "message": str(error), "_response": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "id": identifier, "status": "fail", "durationMs": 300000.0, "exitCode": None,
            "stdoutSha256": None, "responseSha256": None, "message": "timeout", "_response": None,
        }



def run_process(command: list[str], env: dict[str, str], timeout: int = 300) -> dict[str, Any]:
    """Run a non-runner host command such as a compiler without parsing TurboJev JSON."""
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
        duration = (time.perf_counter() - started) * 1000
        return {
            "exitCode": process.returncode,
            "durationMs": round(duration, 3),
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
    except FileNotFoundError as error:
        return {"exitCode": None, "durationMs": round((time.perf_counter() - started) * 1000, 3), "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired:
        return {"exitCode": None, "durationMs": float(timeout * 1000), "stdout": "", "stderr": "timeout"}


def windows_compiler_environment(env: dict[str, str]) -> dict[str, str]:
    if sys.platform != "win32" or shutil.which("cl", path=env.get("PATH")):
        return env
    script = r"""$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $vs) { exit 1 }
$vcvars = Join-Path $vs 'VC/Auxiliary/Build/vcvarsall.bat'
& $env:ComSpec /d /s /c ('"{0}" x64 > nul && set' -f $vcvars)
"""
    install = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
    )
    if install.returncode != 0:
        return env
    result = env.copy()
    for line in install.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            if key.casefold() == "path":
                key = "PATH"
            elif key.casefold() == "include":
                key = "INCLUDE"
            elif key.casefold() == "lib":
                key = "LIB"
            elif key.casefold() == "libpath":
                key = "LIBPATH"
            for existing in tuple(result):
                if existing.casefold() == key.casefold():
                    del result[existing]
            result[key] = value
    return result

def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib-dir", type=Path, required=True)
    ap.add_argument("--node-addon", type=Path, required=True)
    ap.add_argument("--python-root", type=Path, default=ROOT / "bindings/python")
    ap.add_argument("--request", type=Path, default=ROOT / "tests/e2e/request.json")
    ap.add_argument("--mmproj", type=Path, help="Optional llama.cpp multimodal projector GGUF")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--include-swift", action="store_true")
    ap.add_argument("--abs-tolerance", type=float, default=1e-7)
    ap.add_argument("--rel-tolerance", type=float, default=1e-7)
    args = ap.parse_args()
    if args.abs_tolerance < 0 or args.rel_tolerance < 0:
        raise SystemExit("E2E tolerances must be non-negative")
    model = args.model.resolve(); request_path = args.request.resolve(); libdir = args.native_lib_dir.resolve(); addon = args.node_addon.resolve()
    mmproj = args.mmproj.resolve() if args.mmproj else None
    if addon.suffix.lower() != ".node":
        raise SystemExit("--node-addon must name the packaged .node addon; a raw DLL can silently fall back to a different installed build")
    for path in (model, request_path, libdir, addon, *((mmproj,) if mmproj else ())):
        if not path.exists():
            raise SystemExit(f"missing E2E input: {path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    mmproj_arg = [str(mmproj)] if mmproj else []
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(libdir) + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["DYLD_LIBRARY_PATH"] = str(libdir) + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    env["PATH"] = str(libdir) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(args.python_root.resolve()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    dotnet_output = ROOT / "tests/e2e/dotnet/bin/Release/net8.0"
    dotnet_output.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.dll", "*.so", "*.so.*", "*.dylib"):
        for library in libdir.glob(pattern):
            shutil.copy2(library, dotnet_output / library.name)
    results: list[dict[str, Any]] = []
    results.append(run_case("node-ts", ["node", str(ROOT / "tests/e2e/node/run.mjs"), str(model), str(request_path), str(addon), *mmproj_arg], env, request))
    results.append(run_case("python", [sys.executable, str(ROOT / "tests/e2e/python/run.py"), str(model), str(request_path), *mmproj_arg], env, request))
    with tempfile.TemporaryDirectory(prefix="turbojev-e2e-") as td:
        exe = Path(td) / "turbojev-c-e2e"
        cc = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(ROOT / "tests/e2e/c/main.c"), "-I", str(ROOT / "bindings/c/include"), "-L", str(libdir), "-lturbojev_capi", "-o", str(exe)]
        if sys.platform == "win32":
            exe = exe.with_suffix(".exe")
            cc = ["cl", "/nologo", "/std:c11", "/W4", "/WX", str(ROOT / "tests/e2e/c/main.c"),
                  "/I" + str(ROOT / "bindings/c/include"), "/Fo" + str(Path(td) / "main.obj"),
                  "/Fe" + str(exe), "/link", "/LIBPATH:" + str(libdir), "turbojev_capi.dll.lib"]
        compiler_env = windows_compiler_environment(env)
        if sys.platform == "win32":
            compiler = shutil.which("cl", path=compiler_env.get("PATH"))
            if compiler is None:
                compile_result = {
                    "exitCode": None, "durationMs": 0.0, "stdout": "",
                    "stderr": "MSVC compiler cl.exe was not found after initializing Visual Studio",
                }
            else:
                # Use the resolved executable path. CreateProcess does not always
                # resolve a bare `cl` consistently from Python's modified env PATH.
                cc[0] = compiler
                compile_result = run_process(cc, compiler_env)
        else:
            compile_result = run_process(cc, compiler_env)
        if compile_result["exitCode"] == 0:
            results.append(run_case("c", [str(exe), str(model), str(request_path), *mmproj_arg], env, request))
        else:
            diagnostic = compile_result["stderr"] or compile_result["stdout"] or "C runner compilation failed"
            results.append({
                "id": "c", "status": "fail", "durationMs": compile_result["durationMs"],
                "exitCode": compile_result["exitCode"], "stdoutSha256": None,
                "responseSha256": None, "message": diagnostic, "_response": None,
            })
    results.append(run_case("dotnet", ["dotnet", "run", "--project", str(ROOT / "tests/e2e/dotnet/TurboJev.E2E.csproj"), "-c", "Release", "--no-build", "--", str(model), str(request_path), *mmproj_arg], env, request))
    if args.include_swift:
        results.append(run_case("swift", ["swift", "build", "--package-path", str(ROOT / "bindings/swift"), "-Xswiftc", "-warnings-as-errors"], env, request, required=False))

    required = {"node-ts", "python", "c", "dotnet"}
    passed_rows = {r["id"]: r for r in results if r["id"] in required and r["status"] == "pass"}
    reference_language = "node-ts" if "node-ts" in passed_rows else (next(iter(passed_rows)) if passed_rows else None)
    parity_mismatches: list[str] = []
    if reference_language is not None:
        reference = passed_rows[reference_language]["_response"]
        for identifier in sorted(required):
            row = passed_rows.get(identifier)
            if row is None or identifier == reference_language:
                continue
            local: list[str] = []
            equivalent(reference, row["_response"], "$", local, args.abs_tolerance, args.rel_tolerance)
            parity_mismatches.extend(f"{reference_language} vs {identifier}: {item}" for item in local[:32])
    semantic_parity = len(passed_rows) == len(required) and not parity_mismatches
    passed = semantic_parity and all(r["status"] == "pass" for r in results if r["id"] in required)
    report = {
        "contract": "turbojev.e2e-language-report.v2",
        "modelSha256": sha(model),
        "mmprojSha256": sha(mmproj) if mmproj else None,
        "requestSha256": sha(request_path),
        "referenceLanguage": reference_language,
        "tolerance": {"absolute": args.abs_tolerance, "relative": args.rel_tolerance},
        "semanticParity": semantic_parity,
        "parityMismatches": parity_mismatches,
        "results": [public_result(r) for r in results],
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
