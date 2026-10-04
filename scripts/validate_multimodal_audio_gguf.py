#!/usr/bin/env python3
"""Run real GGUF audio decisions and candidate-order checks through llama.cpp/mtmd."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "multimodal-audio"
OPTIONS = {
    "on": "Turn on the light",
    "off": "Turn off the light",
}


def make_request(wav: Path, reverse: bool) -> dict:
    items = list(OPTIONS.items())
    if reverse:
        items.reverse()
    return {
        "state": {
            "contract": "turbojev.media.v1",
            "modality": "audio",
            "text": "",
            "base64": base64.b64encode(wav.read_bytes()).decode("ascii"),
        },
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Which command does the speaker say?",
                "criteria": dict(items),
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True, help="built multimodal_gguf executable")
    parser.add_argument("--model", type=Path, required=True, help="audio-capable model GGUF")
    parser.add_argument("--mmproj", type=Path, required=True, help="matching multimodal projector GGUF")
    parser.add_argument(
        "--library-dir", type=Path, action="append", default=[],
        help="native llama.cpp/mtmd library directory to prepend to PATH (repeat as needed)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    for path in (args.exe, args.model, args.mmproj, *args.library_dir):
        if not path.exists():
            parser.error(f"path does not exist: {path}")

    env = os.environ.copy()
    if args.library_dir:
        env["PATH"] = os.pathsep.join(str(p.resolve()) for p in args.library_dir) + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory(prefix="turbojev-audio-e2e-") as temporary:
        temp = Path(temporary)
        requests: list[tuple[str, str, Path]] = []
        for clip, expected in (("on", "on"), ("off", "off")):
            wav = FIXTURES / f"{clip}.wav"
            for order in ("normal", "reversed"):
                path = temp / f"{clip}-{order}.json"
                path.write_text(
                    json.dumps(make_request(wav, order == "reversed")), encoding="utf-8"
                )
                requests.append((clip, expected, path))

        command = [
            str(args.exe.resolve()), str(args.model.resolve()), str(args.mmproj.resolve()),
            *(str(path) for _, _, path in requests), "--cyclic",
        ]
        started = time.perf_counter()
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=args.timeout_seconds, check=False,
        )
        wall_seconds = time.perf_counter() - started
        if result.returncode != 0:
            sys.stderr.write(result.stderr[-12000:])
            raise SystemExit(f"multimodal_gguf exited {result.returncode}")
        if "audio=true" not in result.stderr:
            raise SystemExit("loaded runtime did not report audio=true")

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != len(requests):
            raise SystemExit(f"expected {len(requests)} JSON responses, received {len(lines)}")
        times = {
            int(index): int(milliseconds)
            for index, milliseconds in re.findall(r"evaluation=(\d+) elapsed_ms=(\d+)", result.stderr)
        }
        responses = [json.loads(line) for line in lines]
        reference_probabilities: dict[str, dict[str, float]] = {}
        for index, ((clip, expected, _), response) in enumerate(zip(requests, responses, strict=True), 1):
            answer = response["answers"]["decision"]
            metadata = response["_turbojev"]
            if answer.get("type") != "choice":
                raise SystemExit(f"{clip}: response is not a typed choice")
            if response["usage"].get("generated_tokens") != 0:
                raise SystemExit(f"{clip}: candidate scoring unexpectedly generated model tokens")
            if metadata.get("order_robustness") != "cyclic" or metadata.get("order_evaluations") != 2:
                raise SystemExit(f"{clip}: cyclic order scoring metadata is missing or incorrect")
            if answer["choice"] != expected:
                raise SystemExit(f"{clip}: expected {expected!r}, received {answer['choice']!r}")

            probabilities = answer["probabilities"]
            if set(probabilities) != set(OPTIONS):
                raise SystemExit(f"{clip}: unexpected probability keys: {sorted(probabilities)}")
            if clip in reference_probabilities:
                reference = reference_probabilities[clip]
                if any(not math.isclose(probabilities[key], reference[key], rel_tol=0, abs_tol=1e-6) for key in OPTIONS):
                    raise SystemExit(f"{clip}: cyclic probabilities changed when candidate order was reversed")
            else:
                reference_probabilities[clip] = probabilities
            elapsed = f"{times[index]} ms" if index in times else "unreported"
            order = "normal" if index % 2 else "reversed"
            print(f"{clip}-{order}: {answer['choice']} ({answer['confidence']:.3f}), {elapsed}")

        print(f"PASS: 4 real GGUF audio evaluations; candidate order invariant; process {wall_seconds:.1f}s")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
