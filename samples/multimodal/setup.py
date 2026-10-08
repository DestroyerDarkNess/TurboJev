"""Build TurboJev and install/download only the selected optional CPU sample."""

import argparse
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".cache/multimodal-venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vision", "audio"])
    parser.add_argument(
        "--skip-install", action="store_true", help="Reuse an already prepared Python environment"
    )
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Use Python 3.12 or newer")
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=True).create(ENV)
    if not args.skip_install:
        torch_args = ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"]
        if sys.platform != "darwin":
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.run([str(PYTHON), "-m", "pip", "install", *torch_args], check=True)
        dependencies = ["transformers==4.57.6", "av==16.1.0", "num2words==0.5.14"]
        if args.kind == "audio":
            dependencies += ["liquid-audio==1.3.0", "soundfile==0.13.1"]
        subprocess.run([str(PYTHON), "-m", "pip", "install", *dependencies], check=True)
    env = dict(os.environ, PYO3_PYTHON=str(PYTHON))
    subprocess.run(
        ["cargo", "build", "--locked", "-p", "turbojev-python"], cwd=ROOT, env=env, check=True
    )
    stage = ROOT / "samples/.artifacts/multimodal/python/turbojev"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "bindings/python/turbojev/__init__.py", stage)
    source, dest = (
        ("_native.dll", "_native.pyd")
        if os.name == "nt"
        else (
            "lib_native.dylib" if sys.platform == "darwin" else "lib_native.so",
            "_native.abi3.so",
        )
    )
    binary = ROOT / "target/debug" / source
    if not (stage / dest).exists() or not filecmp.cmp(binary, stage / dest, shallow=False):
        try:
            shutil.copy2(binary, stage / dest)
        except PermissionError as error:
            raise SystemExit(
                "Close running multimodal samples before replacing the native library"
            ) from error
    spec = json.loads(Path(__file__).with_name("models.json").read_text())[args.kind]
    destination = ROOT / ".cache/multimodal-models" / args.kind
    # Download just the input-scoring model: no ONNX duplicates or audio generators.
    script = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
        "allow_patterns=['*.json','*.txt','*.jinja','LICENSE'], "
        "ignore_patterns=['onnx/*','audio_detokenizer/*'])"
    )
    download_env = dict(os.environ)
    download_env.setdefault("HF_HUB_DISABLE_XET", "1")
    subprocess.run(
        [str(PYTHON), "-c", script, spec["id"], spec["revision"], str(destination)],
        env=download_env,
        check=True,
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    from final_e2e_linux import download, sha256

    weights = destination / "model.safetensors"
    url = f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/model.safetensors?download=true"
    if weights.exists() and sha256(weights) == spec["sha256"]:
        print(f"Verified cached weights: {weights}")
    elif shutil.which("curl"):
        # Curl can resume interrupted multi-GB downloads without discarding bytes.
        partial = weights.with_suffix(".safetensors.part")
        subprocess.run(
            [
                shutil.which("curl"),
                "--location",
                "--fail",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        if sha256(partial) != spec["sha256"]:
            raise SystemExit(f"Model checksum mismatch. Remove {partial} and run setup again.")
        partial.replace(weights)
    else:
        download(url, weights, spec["sha256"])
    destination.joinpath("turbojev-source.json").write_text(json.dumps(spec, indent=2))
    example = "TurboJev-SmolVLM2" if args.kind == "vision" else "TurboJev-LFM-Audio"
    print(f'Ready. Run: "{PYTHON}" "{ROOT / "samples" / example / "main.py"}" --help')


if __name__ == "__main__":
    main()
