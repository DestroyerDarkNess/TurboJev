"""Launch a sample using the compiled libraries for this host architecture."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from build import ROOT, SAMPLES, host

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sample", choices=["web", "ts", "python", "csharp", "rust", "vision", "audio"])
parser.add_argument("--model", type=Path, default=ROOT / ".cache/real-validation/SmolLM2-135M-Instruct-Q2_K.gguf")
parser.add_argument("--text", default="My package arrived broken and I need a replacement.")
parser.add_argument("--file", type=Path, help="Local media for vision/audio samples")
parser.add_argument("--modality", choices=["image", "video", "audio"])
parser.add_argument("--question", default="What is happening?")
parser.add_argument("--options", nargs="+", help="Candidate descriptions for vision/audio")
args = parser.parse_args()
if args.sample in ("vision", "audio"):
    python = ROOT / ".cache/multimodal-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        parser.error(f"First run: python {SAMPLES / 'multimodal/setup.py'} {args.sample}")
    if args.file is None:
        parser.error("vision/audio requires --file /path/to/media")
    sample = "TurboJev-SmolVLM2" if args.sample == "vision" else "TurboJev-LFM-Audio"
    command = [str(python), str(SAMPLES / sample / "main.py"), str(args.file.resolve()),
               "--question", args.question]
    if args.modality:
        command += ["--modality", args.modality]
    if args.options:
        command += ["--options", *args.options]
    raise SystemExit(subprocess.call(command))
rid = "-".join(host())
stage = SAMPLES / ".artifacts" / rid
env = os.environ.copy()
env["TURBOJEV_ARTIFACTS"] = str(stage)
env["PYTHONPATH"] = str(stage / "python")
for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
    env[key] = str(stage) + os.pathsep + env.get(key, "")
if args.sample == "web":
    command = ["npm.cmd" if os.name == "nt" else "npm", "run", "dev"]
    cwd = SAMPLES / "TurboJev-js-native"
else:
    if not (stage / "build-info.json").exists():
        raise SystemExit("First run: python samples/build.py native --download-model")
    if not args.model.is_file():
        raise SystemExit("Model not found. Use --model /path/model.gguf or build.py native --download-model")
    executables = {
        "ts": ["node", str(SAMPLES / "TurboJev-TS-node/dist/main.js")],
        "python": [sys.executable, str(SAMPLES / "TurboJev-Python/main.py")],
        "rust": [str(stage / ("turbojev-rust-sample.exe" if os.name == "nt" else "turbojev-rust-sample"))],
        "csharp": ["dotnet", str(SAMPLES / "TurboJev-csharp-NET/bin/Release/net8.0/Sample.dll")],
    }
    command = executables[args.sample] + [str(args.model.resolve()), args.text]
    cwd = ROOT
raise SystemExit(subprocess.call(command, cwd=cwd, env=env))
