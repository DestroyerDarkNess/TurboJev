# LFM2.5-Audio spoken-intent decisions

This optional example sends actual audio to **LFM2.5-Audio-1.5B** and returns
TurboJev's typed decision. It does not transcribe speech into an intermediate
text-only model. It reads candidate logits without generating speech or text.

## Install and run

Requirements: Python 3.12, Rust with the repository toolchain, a native compiler,
and sufficient memory for the 1.5B model. CPU inference uses bfloat16 and can be
slow. Run from the repository root:

```powershell
py samples/multimodal/setup.py audio
.cache/multimodal-venv/Scripts/python.exe samples/TurboJev-LFM-Audio/main.py speech.wav --question "What does the speaker want?" --options "Turn on the light" "Turn off the light"
```

On Linux/macOS, use `python3` for setup and `.cache/multimodal-venv/bin/python`
for running. From `samples`, use `py multimodal/setup.py audio` and
`../.cache/multimodal-venv/Scripts/python.exe TurboJev-LFM-Audio/main.py ...`.

The installer reuses the vision example's environment and compiled Rust
binding, then downloads the pinned audio model separately. Downloads are
several GB. No audio-output detokenizer is needed or downloaded. Inference is
offline after setup. Files remain outside Git.
If the weight download is interrupted, run setup again: it resumes the partial
file when `curl` is installed (included in current Windows installations).
The model is accepted only after its pinned SHA-256 matches.

## Scope

- English speech, WAV/FLAC and other formats supported by libsndfile.
- At most 30 seconds and 24 MiB; stereo is mixed to mono and resampled by the
  official processor. The model accepts real audio embeddings.
- 2–26 answer options, scored by mean conditional token log probability of each
  description (at most 64 tokens), using the real tokenizer.
- Rust computes typed responses and conditional, uncalibrated probabilities.
- The sample pins `liquid-audio==1.3.0`: candidate scoring uses its internal
  `_prefill` method and tied text output head, so upgrades require revalidation.
- This is a desktop Python sample; it does not implement browser audio inference.

Model and runtime: [official model card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
and [LFM Open License](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/blob/main/LICENSE).
These optional downloads have their own license; TurboJev remains Apache-2.0.

## Validation status

The runtime imports and the shared Rust worker path were checked on Windows.
**Real LFM2.5-Audio inference has not yet been validated:** its approximately
2.94 GB weight download did not complete during this session. Treat this sample
as experimental until the following real-model check passes:

```powershell
.cache/multimodal-venv/Scripts/python.exe samples/multimodal/verify.py audio --files turn-on.wav turn-off.wav
```

Use two short English recordings with different instructions. The suite tests
real audio sensitivity, typed answers, repeatability, and malformed inputs.
It does not assert speech-recognition accuracy. See
[validation status](../../docs/MULTIMODAL_VALIDATION.md).
