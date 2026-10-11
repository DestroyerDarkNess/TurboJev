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

from native_assets import node_package, required_runtime, architecture, targets

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
        for target in targets():
            assert data["optionalDependencies"][node_package(target)] == version
    return {"name": name, "version": version, "integrity": data["dist"]["integrity"]}


def verify_cargo(name, version):
    data = fetch(f"https://crates.io/api/v1/crates/{name}/{version}")["version"]
    assert data["num"] == version and not data["yanked"]
    return {"name": name, "version": version, "checksum": data["checksum"]}


def verify_python(version):
    data = fetch(f"https://pypi.org/pypi/turbojev/{version}/json")
    assert data["info"]["version"] == version
    wheels = [entry for entry in data["urls"] if entry["packagetype"] == "bdist_wheel" and not entry["yanked"]]
    for platform in (target["wheel"] for target in targets()):
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
    with zipfile.ZipFile(io.BytesIO(published)) as archive:
        names = set(archive.namelist())
        assert len(names) == len(archive.namelist()), "Duplicate public NuGet archive entries"
        if local.exists():
            # nuget.org adds its repository signature to the ZIP. Compare the
            # complete unsigned payload, not container bytes changed by signing.
            with zipfile.ZipFile(local) as original:
                original_names = set(original.namelist())
                payload = names - {".signature.p7s"}
                assert payload == original_names - {".signature.p7s"}, "NuGet payload entries differ"
                for name in payload:
                    assert archive.read(name) == original.read(name), f"NuGet payload differs: {name}"
        native_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names if name.startswith("runtimes/") and "/native/" in name
        }
    assets = [target["rid"] for target in targets()]
    for target in targets():
        suffix = ".dll" if target["os"] == "win32" else ".dylib" if target["os"] == "darwin" else ".so"
        prefix = "" if target["os"] == "win32" else "lib"
        libraries = required_runtime(target["os"], architecture(target)) | {prefix + "turbojev_capi" + suffix}
        for library in libraries:
            assert f"runtimes/{target['rid']}/native/{library}" in names, (target["rid"], library)
    return {"name": "TurboJev", "version": version, "sha512": expected,
            "repository_signature_present": ".signature.p7s" in names,
            "payload_matches_local": True if local.exists() else None,
            "native_sha256": native_hashes, "verified_native_assets": assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=tomllib.loads((ROOT/"Cargo.toml").read_text())["workspace"]["package"]["version"])
    parser.add_argument("--registry", choices=["all", "npm", "python", "nuget", "cargo"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/published-release-verification.json")
    args = parser.parse_args()
    jobs = []
    if args.registry in ("all", "npm"):
        names = ["@turbojev/wasm", "@turbojev/web", "@turbojev/runtime-transformersjs", "@turbojev/runtime-llamacpp-wasm", "@turbojev/node"]
        names += [node_package(target) for target in targets()]
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
