"""Local Jev-Omni scorer: JSON lines in, raw candidate logits out.

Requires a trusted checkout of akhilaaa3/Jev-Omni and its CUDA dependencies.
Model loading may download both Jev-Omni and the Gemma multimodal components.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

MAX_MEDIA_BYTES = 24 * 1024 * 1024


def decode_evidence(state):
    if isinstance(state, dict) and state.get("contract") == "turbojev.media.v1":
        if set(state) - {"contract", "text", "modality", "base64"}:
            raise ValueError("unknown media fields")
        modality = state["modality"]
        if modality not in {"image", "audio", "video"}:
            raise ValueError("unsupported modality")
        encoded = state["base64"]
        if not isinstance(encoded, str) or not 0 < len(encoded) <= MAX_MEDIA_BYTES * 4 // 3:
            raise ValueError("media exceeds 24 MiB limit or is empty")
        data = base64.b64decode(encoded, validate=True)
        if not data or len(data) > MAX_MEDIA_BYTES:
            raise ValueError("media exceeds 24 MiB limit or is empty")
        text = state.get("text", "")
        if not isinstance(text, str):
            raise ValueError("media text must be a string")
        return text, modality, data
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False), "text", None


def score_batch(classifier, reference, torch, batch):
    from PIL import Image

    text, modality, data = decode_evidence(batch["state"])
    scores = []
    with tempfile.TemporaryDirectory(prefix="turbojev-media-") as directory:
        source = Path(directory) / "input.media"
        content = []
        if data is not None:
            source.write_bytes(data)
        if modality == "image":
            with Image.open(source) as image:
                content.append({"type": "image", "image": image.convert("RGB")})
        elif modality == "video":
            content.extend({"type": "image", "image": frame}
                           for frame in reference._video_frames(source, 16))
        elif modality == "audio":
            wav = Path(directory) / "audio.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
                            "-t", "30", "-ac", "1", "-ar", "16000", str(wav)],
                           check=True, timeout=60, stdout=sys.stderr)
            content.append({"type": "audio", "audio": str(wav)})
        for task in batch["tasks"]:
            options = [c["description"] for c in task["candidates"]]
            if not 2 <= len(options) <= 256:
                raise ValueError("Jev-Omni requires 2..256 candidates")
            question = task.get("instruction") or "Choose the option that best matches the state."
            messages = [{"role": "user", "content": content + [
                {"type": "text", "text": reference._prompt(text, question, options)}]}]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inputs = classifier.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", enable_thinking=False)
                inputs = {k: v.to(classifier.device, dtype=torch.bfloat16) if torch.is_floating_point(v)
                          else v.to(classifier.device) for k, v in inputs.items()}
                classifier._capture.clear()
                classifier.model(**inputs, use_cache=False, **classifier._extra)
                counts = torch.tensor([len(options)], device=classifier.device)
                logits = classifier.head(classifier._capture["hidden"], counts)[0, :len(options)]
                scores.append({"id": task["id"], "logits": logits.float().cpu().tolist()})
    return {"model": reference.MODEL_ID, "scores": scores, "metrics": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_dir.resolve()))
    # Keep model/progress messages away from the JSON protocol stream.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import jev_omni as reference
        classifier = reference.load_jev_omni()
    print(json.dumps({"contract": "turbojev.scorer.v1", "modalities": ["text", "image", "audio", "video"]}), flush=True)
    while True:
        line = sys.stdin.buffer.readline(40 * 1024 * 1024 + 1)
        if not line:
            return
        if len(line) > 40 * 1024 * 1024 or not line.endswith(b"\n"):
            raise ValueError("oversized or unterminated request")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = score_batch(classifier, reference, torch, json.loads(line))
            output = json.dumps(result, allow_nan=False)
        except Exception as error:
            output = json.dumps({"error": str(error)})
        print(output, flush=True)


if __name__ == "__main__":
    main()
