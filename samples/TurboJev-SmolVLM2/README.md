# SmolVLM2 image and video decisions

This example uses **SmolVLM2-256M-Video-Instruct** on CPU through an optional
Python worker. Compiled TurboJev Rust validates the request and computes the
typed answer and conditional probabilities. The worker reads real candidate
sequence log likelihoods; it does not generate or parse model answers.

## Install and run

Requirements: Python 3.12, Rust with the repository toolchain, and a native
compiler. Run from the repository root:

```powershell
py samples/multimodal/setup.py vision
.cache/multimodal-venv/Scripts/python.exe samples/TurboJev-SmolVLM2/main.py clip.mp4 --question "What is happening?" --options "A person is walking" "A person is sitting"
.cache/multimodal-venv/Scripts/python.exe samples/TurboJev-SmolVLM2/main.py photo.jpg --modality image --question "What animal is shown?" --options "A cat" "A dog"
```

On Linux/macOS, use `python3` for setup and `.cache/multimodal-venv/bin/python`
for running. From the `samples` directory, use `py multimodal/setup.py vision`
and `../.cache/multimodal-venv/Scripts/python.exe TurboJev-SmolVLM2/main.py ...`.
Paths to input files are relative to your current directory.

Setup downloads approximately 1.03 GB of model weights plus dependencies,
builds the Python binding without llama.cpp, and caches everything under
`.cache` and `samples/.artifacts`. The model revision is pinned in
`../multimodal/models.json`. Subsequent inference is offline. No CUDA is needed.

## Scope

- Images: Pillow-supported formats such as JPEG and PNG, resized to a 512 px view.
- Video: PyAV-supported containers/codecs, using four sampled frames and the
  model's video chat template. At most 30 seconds with a known duration.
  Soundtracks are not processed.
- One file up to 24 MiB per request; 2–26 answer options.
- Scores compare mean token log probabilities of complete candidate descriptions
  (at most 64 tokens each), using the real tokenizer and media input. Probabilities
  are conditional on those options, not calibrated accuracy estimates.
- This is a desktop Python sample, not a browser runtime. The worker protocol
  is also accessible through the other native TurboJev bindings.

Model: [official model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
(Apache-2.0). Model weights are downloaded separately and are not bundled with
TurboJev. Support here does not imply support for arbitrary model architectures.

## Real validation

Run the opt-in suite after setup:

```powershell
.cache/multimodal-venv/Scripts/python.exe samples/multimodal/verify.py vision
```

It creates deterministic PNG/MP4 fixtures locally and checks real model scoring,
choice/noul/score, media sensitivity, repeatability, zero generated tokens and
invalid inputs. Reports go to `.cache/multimodal-validation`. This is an
integration check, not a model-quality benchmark. See
[validation status](../../docs/MULTIMODAL_VALIDATION.md).
