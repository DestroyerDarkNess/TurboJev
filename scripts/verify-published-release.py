"""Verify exact public registry versions and native assets; never uses credentials."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib
from urllib.request import Request, urlopen
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, raw=False):
    request = Request(url, headers={"User-Agent": "TurboJev release verifier", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=30) as response:
        body = response.read()
    return body if raw else json.loads(body)


def verify_npm(name, version):
    data = fetch(f"https://registry.npmjs.org/{quote(name, safe='')}/{version}")
    assert data["name"] == name and data["version"] == version
    assert data["dist"]["integrity"].startswith("sha512-")
    if name == "@turbojev/node":
        for arch in ("ia32", "x64", "arm64"):
            assert data["optionalDependencies"][f"@turbojev/node-win32-{arch}"] == version
    return {"name": name, "version": version, "integrity": data["dist"]["integrity"]}


def verify_cargo(name, version):
    data = fetch(f"https://crates.io/api/v1/crates/{name}/{version}")["version"]
    assert data["num"] == version and not data["yanked"]
    return {"name": name, "version": version, "checksum": data["checksum"]}


def verify_python(version):
    data = fetch(f"https://pypi.org/pypi/turbojev/{version}/json")
    assert data["info"]["version"] == version
    wheels = [entry for entry in data["urls"] if entry["packagetype"] == "bdist_wheel" and not entry["yanked"]]
    for platform in ("win32", "win_amd64", "win_arm64"):
        entries = [entry for entry in wheels if entry["filename"].endswith(f"-{platform}.whl")]
        assert len(entries) == 1, (platform, entries)
        entry = entries[0]
        local = list((ROOT/"artifacts/pypi").rglob(entry["filename"]))
        if local:
            assert hashlib.sha256(local[0].read_bytes()).hexdigest() == entry["digests"]["sha256"]
    return {"name": "turbojev", "version": version, "wheels": [{"filename": entry["filename"], "sha256": entry["digests"]["sha256"]} for entry in wheels]}


def verify_nuget(version):
    base = f"https://api.nuget.org/v3-flatcontainer/turbojev/{version}/turbojev.{version}.nupkg"
    # NuGet's flat container serves the package, not a .nupkg.sha512 blob.
    # Hash the actual public bytes and inspect their runtime assets directly.
    published = fetch(base, raw=True)
    expected = base64.b64encode(hashlib.sha512(published).digest()).decode()
    local = ROOT / f"artifacts/nuget/TurboJev.{version}.nupkg"
    if local.exists():
        actual = base64.b64encode(hashlib.sha512(local.read_bytes()).digest()).decode()
        assert actual == expected, "Local and published NuGet packages differ"
    with zipfile.ZipFile(io.BytesIO(published)) as archive:
        names = set(archive.namelist())
    assets = ["win-x86", "win-x64", "win-arm64"]
    for rid in assets:
        for library in ("turbojev_capi.dll", "turbojev_llama_shim.dll", "mtmd.dll", "llama.dll"):
            assert f"runtimes/{rid}/native/{library}" in names, (rid, library)
    return {"name": "TurboJev", "version": version, "sha512": expected, "verified_native_assets": assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=tomllib.loads((ROOT/"Cargo.toml").read_text())["workspace"]["package"]["version"])
    parser.add_argument("--registry", choices=["all", "npm", "python", "nuget", "cargo"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/published-release-verification.json")
    args = parser.parse_args()
    jobs = []
    if args.registry in ("all", "npm"):
        names = ["@turbojev/wasm", "@turbojev/web", "@turbojev/runtime-transformersjs", "@turbojev/runtime-llamacpp-wasm", "@turbojev/node"]
        names += [f"@turbojev/node-win32-{arch}" for arch in ("ia32", "x64", "arm64")]
        jobs += [("npm", name, lambda name=name: verify_npm(name, args.version)) for name in names]
    if args.registry in ("all", "python"):
        jobs.append(("python", "turbojev", lambda: verify_python(args.version)))
    if args.registry in ("all", "nuget"):
        jobs.append(("nuget", "TurboJev", lambda: verify_nuget(args.version)))
    if args.registry in ("all", "cargo"):
        metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--offline", "--locked", "--no-deps", "--format-version", "1"], cwd=ROOT,
            env={**os.environ, "CARGO_CACHE_AUTO_CLEAN_FREQUENCY": "never"}))
        packages = {package["name"]: package for package in metadata["packages"]}
        names = set()
        def visit(name):
            if name in names: return
            names.add(name)
            for dependency in packages[name]["dependencies"]:
                if dependency.get("path") and dependency["name"] in packages: visit(dependency["name"])
        for name in ("turbojev", "turbojev-runtime-llamacpp", "turbojev-host-native"): visit(name)
        jobs += [("cargo", name, lambda name=name: verify_cargo(name, args.version)) for name in sorted(names)]
    def execute(job):
        registry, name, action = job
        try: return {"registry": registry, "name": name, "ok": True, "evidence": action()}
        except Exception as error: return {"registry": registry, "name": name, "ok": False, "error": str(error)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(execute, jobs))
    report = {"version": args.version, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for record in records:
        print(f"{'OK' if record['ok'] else 'FAIL'} {record['registry']} {record['name']} {record.get('error','')}")
    return int(any(not record["ok"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
