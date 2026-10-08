"""Opt-in real inference checks; writes evidence without claiming model accuracy."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "samples/.artifacts/multimodal/python"))


def fixtures(directory):
    import av
    from PIL import Image

    for color in ["red", "blue"]:
        image = Image.new("RGB", (256, 256), color)
        image.save(directory / f"{color}.png")
        with av.open(str(directory / f"{color}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for _ in range(8):
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return [
        (directory / f"{color}.{ext}", modality)
        for ext, modality in [("png", "image"), ("mp4", "video")]
        for color in ["red", "blue"]
    ]


def main():
    from turbojev import TurboJev, TurboJevError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vision", "audio"])
    parser.add_argument(
        "--files",
        nargs=2,
        type=Path,
        help="Two different English speech WAV files for audio checks",
    )
    args = parser.parse_args()
    if args.kind == "audio" and not args.files:
        parser.error("Audio verification requires --files first.wav second.wav")
    directory = ROOT / ".cache/multimodal-validation"
    directory.mkdir(parents=True, exist_ok=True)
    inputs = (
        fixtures(directory) if args.kind == "vision" else [(path, "audio") for path in args.files]
    )
    model = ROOT / ".cache/multimodal-models" / args.kind
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "What is the dominant color?"
            if args.kind == "vision"
            else "What action does the speaker request?",
            "criteria": {
                "a": "Red" if args.kind == "vision" else "Turn on the light",
                "b": "Blue" if args.kind == "vision" else "Turn off the light",
            },
        },
        "boolean": {
            "type": "noul",
            "instructions": "Is there visual or spoken evidence in the input?",
        },
        "score": {
            "type": "score",
            "instructions": "How clear is the input?",
            "criteria": ["unclear", "clear"],
        },
    }
    report = {
        "kind": args.kind,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ["torch", "transformers", "av"]
        },
        "model": json.loads((model / "turbojev-source.json").read_text()),
        "runs": [],
    }
    with TurboJev.load(
        str(model),
        worker={
            "program": sys.executable,
            "args": [str(Path(__file__).with_name("worker.py")), args.kind, str(model)],
            "timeoutSeconds": 600,
        },
    ) as engine:
        for path, modality in inputs:
            data = path.read_bytes()
            request = {
                "state": {
                    "contract": "turbojev.media.v1",
                    "modality": modality,
                    "base64": base64.b64encode(data).decode("ascii"),
                },
                "questions": questions,
            }
            start = time.monotonic()
            result = engine.evaluate(request)
            assert set(result["answers"]) == set(questions), result
            assert result["usage"]["generated_tokens"] == 0, result
            if args.kind == "vision":
                expected = "a" if path.stem == "red" else "b"
                assert result["answers"]["choice"]["choice"] == expected, result
            report["runs"].append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "seconds": round(time.monotonic() - start, 3),
                    "response": result,
                }
            )
            print(f"PASS {path.name}: {result['answers']['choice']}", flush=True)
        repeat = engine.evaluate(request)
        assert repeat == result, "Repeated request produced a different response"
        reordered = dict(
            questions["choice"],
            criteria=dict(reversed(list(questions["choice"]["criteria"].items()))),
        )
        reversed_result = engine.evaluate(
            {"state": request["state"], "questions": {"choice": reordered}}
        )
        for key, value in result["answers"]["choice"]["probabilities"].items():
            assert math.isclose(
                value, reversed_result["answers"]["choice"]["probabilities"][key], abs_tol=1e-7
            ), "Candidate order changed the probability"
        # Hold the question constant: actual media must influence the probabilities.
        for first, second in zip(report["runs"][::2], report["runs"][1::2]):
            assert (
                first["response"]["answers"]["choice"] != second["response"]["answers"]["choice"]
            ), "Different media produced identical choice output"
        for invalid in ["!not-base64!", base64.b64encode(b"not a media file").decode()]:
            request["state"]["base64"] = invalid
            try:
                engine.evaluate(request)
            except TurboJevError:
                pass
            else:
                raise AssertionError("Invalid media was accepted")
        request["state"]["modality"] = "audio" if args.kind == "vision" else "image"
        request["state"]["base64"] = "AA=="
        try:
            engine.evaluate(request)
        except TurboJevError:
            pass
        else:
            raise AssertionError("Unsupported modality was accepted")
    report["checks"] = [
        "real media inference",
        "choice/noul/score",
        "media sensitivity",
        "repeatability",
        "candidate order invariance",
        "zero generated tokens",
        "invalid media",
        "unsupported modality",
    ]
    if args.kind == "vision":
        report["checks"].append("four correct synthetic color classifications")
    output = directory / f"{args.kind}-report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
