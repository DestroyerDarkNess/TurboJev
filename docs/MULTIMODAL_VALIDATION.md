# Real multimodal sample validation

These optional examples exercise the compiled TurboJev Python binding and
native worker protocol with application-selected models. They do not change
the portable Rust core or enable media in the built-in llama.cpp/Transformers.js
adapters. Browser, other language bindings and other operating systems have
not been qualified with these workers in this run.

## SmolVLM2: Windows x64, CPU

Model: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, pinned revision and SHA-256
in `samples/multimodal/models.json`. Runtime: PyTorch 2.8.0 CPU, Transformers
4.57.6, Python 3.12. Images use a 512 px view; videos use up to four frames.

The real-model suite passed:

- Four correct classifications: red/blue PNG images and red/blue MPEG-4 clips.
- `choice`, `noul`, and `score` response construction through Rust.
- Different media changes probabilities with the question held constant.
- Repeated requests produce identical complete responses.
- Reversing candidates preserves the probability assigned to each candidate.
- Zero generated tokens, malformed base64/file rejection and unsupported
  modality rejection.

Each input included three questions. The observed evaluation times were
approximately 15 seconds per image and 30–42 seconds per video on this machine;
these are observations, not performance guarantees. Model loading is excluded.
See the machine-readable [vision report](validation/multimodal-vision.json).

The first approach used option letters and exhibited a preference for the last
listed option. The final worker scores each complete candidate description as
an independent continuation, using mean token log probability. No candidate
list is inserted into the model prompt. Rust applies the conditional softmax.
These probabilities are not calibrated confidence, and four synthetic color
fixtures do not establish general visual understanding or temporal reasoning.

A separate natural-image check used the bee photograph linked by the official
SmolVLM2 model card. With the candidates "A bee on a flower", "A dog running",
and "A car on a road", the model selected the bee (conditional probability
0.8744). This also exercises multi-token descriptions through the user-facing
launcher. The [natural-image report](validation/multimodal-natural-image.json)
records the source URL, image hash, options and complete response.

## LFM2.5-Audio

The separate audio sample uses the same contract, with the official audio
processor and the LFM2.5 language backbone. Audio decoding, resampling, feature
construction and candidate tokenization have been checked with two English
speech recordings, synthesized locally with Windows' Microsoft David voice.
The recordings produced finite 128-channel features with 211 and 222 time
steps and distinct four-token candidate sequences.

**Full LFM2.5-Audio inference remains unvalidated.** The 2.94 GB model download
did not finish at the available transfer rate. No audio inference result is
claimed. The worker is experimental until `verify.py audio` passes with the
real weights. The installer resumes `.safetensors.part` downloads when `curl`
is installed, and verifies SHA-256 before making the model available.

## Reproduce

From the repository root on Windows:

```powershell
py samples/multimodal/setup.py vision
.cache/multimodal-venv/Scripts/python.exe samples/multimodal/verify.py vision

py samples/multimodal/setup.py audio
.cache/multimodal-venv/Scripts/python.exe samples/multimodal/verify.py audio --files turn-on.wav turn-off.wav
```

Use English recordings saying "Please turn on the light" and "Please turn off
the light" for the audio pair. The validation script creates the visual
fixtures itself and writes results under `.cache/multimodal-validation`.
On Linux/macOS replace the venv executable with
`.cache/multimodal-venv/bin/python`; those platforms remain unverified here.
Model downloads and generated media are ignored by Git.

`RELEASE_ENABLED` remains disabled. These sample checks do not qualify the
package publishing matrix or imply universal multimodal model/codec support.
