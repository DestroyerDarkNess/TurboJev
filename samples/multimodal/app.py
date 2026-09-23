"""Shared CLI for the two real-model examples; Rust owns typed decisions."""
import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(kind):
    parser = argparse.ArgumentParser(description="Classify real local media with TurboJev")
    parser.add_argument("file", type=Path)
    parser.add_argument("--modality", choices=["image", "video"] if kind == "vision" else ["audio"],
                        default="video" if kind == "vision" else "audio")
    parser.add_argument("--question", default="What is happening?")
    parser.add_argument("--options", nargs="+", default=["A person is walking", "A person is sitting"]
                        if kind == "vision" else ["The speaker asks a question", "The speaker gives an instruction"])
    args = parser.parse_args()
    if not 2 <= len(args.options) <= 26 or len(set(args.options)) != len(args.options):
        parser.error("Provide 2 to 26 distinct options")
    if not args.file.is_file() or not 0 < args.file.stat().st_size <= 24 * 1024 * 1024:
        parser.error("Provide an existing nonempty file of at most 24 MiB")
    sys.path.insert(0, str(ROOT / "samples/.artifacts/multimodal/python"))
    from turbojev import TurboJev
    model = ROOT / ".cache/multimodal-models" / kind
    if not (model / "model.safetensors").exists():
        parser.error(f"First run: python samples/multimodal/setup.py {kind}")
    with TurboJev.load(str(model), worker={
        "program": sys.executable,
        "args": [str(Path(__file__).with_name("worker.py")), kind, str(model)],
        "timeoutSeconds": 600,
    }) as engine:
        result = engine.evaluate({
            "state": {"contract": "turbojev.media.v1", "modality": args.modality,
                      "base64": base64.b64encode(args.file.read_bytes()).decode("ascii")},
            "questions": {"scene" if kind == "vision" else "intent": {
                "type": "choice", "instructions": args.question,
                "criteria": {f"option_{i + 1}": label for i, label in enumerate(args.options)},
            }},
        })
    print(json.dumps(result, indent=2))
