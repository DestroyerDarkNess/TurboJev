"""Optional sample runtimes: real media -> candidate logits, never generated answers."""

import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile


def prompt_for(state, task):
    candidates = task["candidates"]
    if not 2 <= len(candidates) <= 26:
        raise ValueError("These samples support 2 to 26 candidates per question")
    labels = [" " + item["description"] for item in candidates]
    # Each candidate is an independent continuation of the same question.
    # Listing candidates in the prompt introduces order/recency bias in tiny VLMs.
    prompt = f"{state.get('text', '')}\n{task.get('instruction') or 'Describe the input.'}"
    return prompt, labels


def candidate_ids(tokenizer, prefix, labels):
    """Resolve complete candidate sequences and reject unstable boundaries."""
    before = tokenizer.encode(prefix, add_special_tokens=False)
    ids = []
    for label in labels:
        after = tokenizer.encode(prefix + label, add_special_tokens=False)
        if after[: len(before)] != before or not 0 < len(after) - len(before) <= 64:
            raise ValueError("Candidate must have a stable token boundary and 1 to 64 tokens")
        ids.append(after[len(before) :])
    if len(set(map(tuple, ids))) != len(ids):
        raise ValueError("Candidates must have distinct token sequences")
    return ids


def sequence_score(logits, ids):
    """Length-normalized conditional log likelihood; no sampling or generation."""
    import torch

    targets = torch.tensor(ids, device=logits.device).unsqueeze(-1)
    return logits.float().log_softmax(-1).gather(-1, targets).mean().item()


class Vision:
    modalities = ["image", "video"]

    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # The checkpoint defaults to upscaling images to 2048 px with many tiles.
        # One 512 px view keeps this CPU example practical on small machines.
        self.processor.image_processor.size = {"longest_edge": 512}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()

    def score(self, state, data, task):
        import torch
        from PIL import Image

        prompt, labels = prompt_for(state, task)
        frames = 4
        with tempfile.TemporaryDirectory(prefix="turbojev-media-") as directory:
            if state["modality"] == "image":
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    media = {"type": "image", "image": source.convert("RGB")}
            else:
                import av

                with av.open(io.BytesIO(data)) as container:
                    if not container.streams.video:
                        raise ValueError("No video stream found")
                    stream = container.streams.video[0]
                    if container.duration is None or container.duration > 30 * av.time_base:
                        raise ValueError(
                            "Provide a video with known duration of at most 30 seconds"
                        )
                    if stream.frames:
                        frames = min(4, stream.frames)
                path = Path(directory) / "input.mp4"
                path.write_bytes(data)
                media = {"type": "video", "path": str(path)}
            messages = [
                {"role": "user", "content": [media, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": "The answer is"}]},
            ]
            prefix = self.processor.apply_chat_template(
                messages, continue_final_message=True, tokenize=False
            )
            candidates = candidate_ids(self.processor.tokenizer, prefix, labels)
            inputs = self.processor.apply_chat_template(
                messages,
                continue_final_message=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                num_frames=frames,
                fps=None,
                do_sample_frames=True,
            )
            if state["modality"] == "video" and inputs["pixel_values"].shape[1] > 4:
                raise ValueError("Video processor exceeded the four-frame budget")
            scores = []
            for ids in candidates:
                suffix = torch.tensor([ids[:-1]], dtype=torch.long)
                scoring_inputs = dict(inputs)
                scoring_inputs["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
                scoring_inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(suffix)], dim=1
                )
                logits = self.model(
                    **scoring_inputs, use_cache=False, logits_to_keep=len(ids)
                ).logits[0]
                scores.append(sequence_score(logits, ids))
            return scores


class Audio:
    modalities = ["audio"]

    def __init__(self, model_path):
        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        # liquid-audio treats strings as Hub IDs, even absolute Windows paths.
        model_path = Path(model_path)
        self.processor = LFM2AudioProcessor.from_pretrained(model_path, device="cpu").eval()
        self.model = LFM2AudioModel.from_pretrained(
            model_path, device="cpu", dtype=torch.bfloat16
        ).eval()

    def score(self, state, data, task):
        import soundfile as sf
        import torch
        from liquid_audio import ChatState

        with sf.SoundFile(io.BytesIO(data)) as source:
            rate = source.samplerate
            if not 0 < source.frames <= rate * 30:
                raise ValueError("Provide nonempty audio of at most 30 seconds")
            waveform = source.read(dtype="float32", always_2d=True)
        if not torch.isfinite(torch.from_numpy(waveform)).all():
            raise ValueError("Audio contains non-finite samples")
        prompt, labels = prompt_for(state, task)
        chat = ChatState(self.processor, dtype=torch.bfloat16)
        chat.new_turn("system")
        chat.add_text("Respond with text only.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0), rate)
        chat.add_text(prompt)
        chat.end_turn()
        chat.new_turn("assistant")
        chat.add_text("The answer is")
        prefix = self.processor.text.decode(chat.text[0], skip_special_tokens=False)
        candidates = candidate_ids(self.processor.text, prefix, labels)
        # liquid-audio 1.3.0 exposes generation, but no public scoring API.
        # Use its prefill and tied output head without sampling any tokens.
        embeddings = self.model._prefill(**chat)
        scores = []
        for ids in candidates:
            suffix = self.model.lfm.embed_tokens(torch.tensor([ids[:-1]], dtype=torch.long))
            hidden = self.model.lfm(
                inputs_embeds=torch.cat([embeddings, suffix], dim=1), use_cache=False
            ).last_hidden_state[0, -len(ids) :]
            logits = torch.nn.functional.linear(hidden, self.model.lfm.embed_tokens.weight)
            scores.append(sequence_score(logits, ids))
        return scores


def main():
    # Native hosts send UTF-8 JSON even when Windows' console uses another codepage.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["vision", "audio"])
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Third-party diagnostic prints must never corrupt the JSON-lines transport.
    with contextlib.redirect_stdout(sys.stderr):
        runtime = (Vision if args.kind == "vision" else Audio)(str(args.model.resolve()))
    source = args.model / "turbojev-source.json"
    model_id = json.loads(source.read_text())["id"] if source.exists() else str(args.model)
    print(
        json.dumps({"contract": "turbojev.scorer.v1", "modalities": runtime.modalities}), flush=True
    )
    for line in sys.stdin:
        try:
            batch = json.loads(line)
            state = batch["state"]
            if (
                state.get("contract") != "turbojev.media.v1"
                or state.get("modality") not in runtime.modalities
            ):
                raise ValueError("Unsupported media contract or modality")
            if len(state["base64"]) > 32 * 1024 * 1024:
                raise ValueError("Media exceeds 24 MiB")
            data = base64.b64decode(state["base64"], validate=True)
            if not data:
                raise ValueError("Media is empty")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                scores = [
                    {"id": task["id"], "logits": runtime.score(state, data, task)}
                    for task in batch["tasks"]
                ]
            response = {"model": model_id, "scores": scores, "metrics": {}}
            payload = json.dumps(response, allow_nan=False)
        except Exception as error:
            payload = json.dumps({"error": f"{type(error).__name__}: {error}"})
        print(payload, flush=True)


if __name__ == "__main__":
    main()
