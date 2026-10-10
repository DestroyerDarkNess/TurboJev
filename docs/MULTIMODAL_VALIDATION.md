# Real multimodal sample validation

This report covers two distinct paths: real GGUF smoke tests through the native
llama.cpp/mtmd runtime, and optional Transformers-based application workers.
They share TurboJev's `evaluate` contract but have separate model runtimes. These
checks do not qualify every binding, architecture, or operating system for
multimodal inference.

## Native GGUF / llama.cpp mtmd: Windows x64 and Linux x64

Built the pinned llama.cpp source and its `mtmd` library into the TurboJev
native shim on both Windows x64 and Linux x64. No FFmpeg/ffprobe executable or
external media process was used. On 2026-09-26, real image, audio, and
timestamped video-frame requests passed through Node/TypeScript, Python, C, and
.NET on each OS using the native language matrix. Within each OS, the four
bindings returned identical complete `EvaluationResponse` values (absolute and
relative tolerance `1e-7`) with zero generated tokens:

The Windows x64 PE import tables were also inspected: `turbojev_llama_shim.dll`
imports `mtmd.dll`, `llama-common.dll`, `llama.dll`, and `ggml.dll`; `mtmd.dll`
imports llama/ggml libraries and the Windows runtime. Neither imports FFmpeg or
ffprobe. Packaging checks require `mtmd.dll` in the NuGet, Node, and Python
Windows native artifacts.

- Image: `LFM2-VL-450M-Q8_0` plus its matching projector and a red-image request.
- Audio: `Qwen3-ASR-0.6B-Q8_0` plus its matching projector and an audio request.
- Video: `LFM2-VL-450M-Q8_0` plus its matching projector and two timestamped
  frames.

These results prove that the native GGUF media path and response contract work
across those four bindings on Windows x64 and Linux x64. They do not establish
model accuracy or qualify other platforms and architectures. Windows and Linux
selected the same answer for the audio-on probe, but its candidate probability
was 0.8743 on Windows and 0.8494 on Linux; exact numerical parity across OS
builds is not claimed. Follow-up fixed-choice probes distinguish media-path
execution from model accuracy:

The direct Rust `multimodal_gguf` example was also run on Windows x64 with the
same three requests. It returned `red` for the image, `on` for the audio, and
`blue` for the video, each with zero generated tokens. That smoke run was not
included in the cross-binding parity comparison. The vision-only LFM2 model and
projector reported `image=true`, `audio=false`, and `video=true`; an audio request
was rejected with `UNSUPPORTED_MODALITY` before scoring.

- Image and two-frame video: `ggml-org/LFM2-VL-450M-GGUF`, Q8_0, with its
  matching `mmproj`. Each video frame was encoded and scored in the same
  context after setting the multimodal session to one sequence. Solid red and
  blue single-image probes were both correct (2/2). For video, the red-first
  sequence was classified correctly when asked about the frame at 0 ms; the
  blue-first sequence was incorrectly classified as red. This validates ordered
  frame ingestion, not temporal understanding or video accuracy.
- Audio: Q8_0 artifacts downloaded from
  `ggml-org/Qwen3-ASR-0.6B-GGUF`, with its matching `mmproj`. llama.cpp's model
  descriptor reported `qwen3vl 1.7B Q8_0`; because that differs from the repo
  label, this result only qualifies the audio transport and mtmd execution
  path, not a particular ASR model or transcription quality. In the two
  turn-the-light-on/off probes it chose `on` correctly, then also chose `on` for
  the off recording (1/2).

On 2026-09-27, the Rust `multimodal_gguf` example also loaded
`ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF` Q4_K_M with its matching F16
projector and evaluated the 2.10-second `audio-on.json` WAV on Windows x64.
`mtmd` reported `audio=true` and `evaluate` returned a typed response with zero
generated tokens. It selected `off` at 0.5029 for the “turn on” recording, so
this confirms native audio transport and typed scoring only; this checkpoint
did not pass the semantic probe. The 1.37 GB projector also makes this a heavy
CPU validation model.

The same day, a real decision probe passed with
[`ggml-org/LFM2-Audio-1.5B-GGUF`](https://huggingface.co/ggml-org/LFM2-Audio-1.5B-GGUF)
Q8_0 and its matching Q8_0 projector on Windows x64 CPU. Two short WAV clips
say “Turn on the light” and “Turn off the light”. Single-pass scoring was
candidate-order sensitive: it predicted `off` for the “on” clip when `on` was
listed first, then predicted `on` for the “off” clip when `off` was listed
first. With `OrderRobustness::Cyclic`, all four combinations (both clips with
normal and reversed candidate order) were correct and produced the same
probabilities after permutation: `on` 0.7190 / `off` 0.2810 for the first clip;
`off` 0.6876 / `on` 0.3124 for the second. Each typed response reported zero
generated tokens and two order evaluations. Per-decision latency ranged from
11.1 to 26.3 s in repeated runs on this CPU; one four-case run took 80.5 s.
This is a small native semantic probe, not a general accuracy or
cross-platform qualification. The WAV fixtures and runner are checked in at
`tests/fixtures/multimodal-audio/` and `scripts/validate_multimodal_audio_gguf.py`.

To reproduce it, build the example and provide the matching model/projector
files plus the llama.cpp/mtmd runtime libraries on `PATH`:

```powershell
$env:TURBOJEV_LLAMA_SHIM_LIB_DIR = 'path/to/llama/lib'
$env:PATH = 'path/to/llama/lib;path/to/llama/bin;path/to/llama/mtmd;' + $env:PATH
cargo build --locked --release -p turbojev-runtime-llamacpp --features ffi --example multimodal_gguf
python scripts/validate_multimodal_audio_gguf.py `
  --exe target/release/examples/multimodal_gguf.exe `
  --model path/to/LFM2-Audio-1.5B-Q8_0.gguf `
  --mmproj path/to/mmproj-LFM2-Audio-1.5B-Q8_0.gguf `
  --library-dir path/to/llama/lib `
  --library-dir path/to/llama/bin `
  --library-dir path/to/llama/mtmd
```

The script sends all four requests through one loaded model session, enables
cyclic order handling, checks typed responses and generated-token counts, and
fails if either answer or its probabilities change when candidate order is
reversed. `orderRobustness` remains opt-in; it increases model evaluations and
latency as described in [the multimodal guide](MULTIMODAL.md).

The model results are deliberately not generalized. Native LFM2 Audio passes
the bounded two-phrase test above with cyclic order handling; Ultravox and the
Qwen3-ASR probe did not pass both semantic cases. Video capability means the
runtime accepts ordered decoded frames; temporal reasoning still depends on the
checkpoint. Browser/WASM real audio inference is still unverified.

Real inference was executed only on Windows x64 and Linux x64. The local Windows
publisher requires `mtmd.dll` alongside llama.cpp for x86, x64, and ARM64
packages, and the Linux E2E staging check requires `libmtmd`; those packaging
checks do not qualify Windows x86, Windows ARM64, Linux ARM64, or other targets.
macOS packaging and native inference remain unverified.

## Browser GGUF / llama.cpp WASM smoke test

On 2026-09-27, the local browser adapter loaded Wllama 3.6.1, the compiled
TurboJev WASM engine, and `LFM2-VL-450M-Q8_0` with its matching projector on
CPU/WASM. A real red image was evaluated through `WasmEngine.evaluate_json`;
the response selected `red` (0.8964) over `blue` (0.1036), with zero generated
tokens. This confirms that a real image payload reaches Wllama's `mtmd` path
and returns a typed TurboJev response. It is one synthetic color probe, not a
general vision-quality qualification.

Ten adapter tests cover image-byte conversion, ordered video-frame construction,
projector loading, capability checks, audio-byte conversion with audio placed
before prompt text, and prompt-format errors using a fake Wllama API. These are
adapter tests, not real inference. A real Qwen3-ASR 0.6B Q8_0 model/projector
loaded in the browser and reported `audio=true`; a TurboJev choice request and a
direct transcription probe both stalled. COOP/COEP headers and four WASM threads
did not resolve this. The audio-first adapter path is now enabled, following
Wllama's own chat client, but no real browser GGUF audio request has yet
completed. Keep browser GGUF audio marked experimental. Transformers.js CLAP
remains available for browser sound classification. Real multi-frame video
inference is also not qualified. Android and Swift share the native JSON
contract but were not included in the real-model matrix above.

The Rust harness is `crates/turbojev-runtime-llamacpp/examples/multimodal_gguf.rs`.
The host must build the pinned native shim and set
`TURBOJEV_LLAMA_SHIM_LIB_DIR` to its library directory before running it.

The browser audio check is prepared at `tests/e2e/browser-gguf-audio.html`. Put
the same LFM2 model and projector under
`.cache/real-validation/multimodal/lfm2-audio/`, then serve the repository root
with the isolation headers required for threaded WebAssembly:

```powershell
.\scripts\prepare-wasm-tests.ps1
py scripts/serve-browser-tests.py --port 4176
```

Open `http://127.0.0.1:4176/tests/e2e/browser-gguf-audio.html`. The page loads
the model locally, evaluates both tracked WAV fixtures in normal and reversed
candidate order, and displays `RESULT`/`PASS` or `ERROR` lines. It enables cyclic
order handling. This browser test has not yet been run; browser audio remains
experimental until its real result is recorded here.

The test core is generated under ignored `.artifacts/wasm-tests/`; it no longer
depends on landing-page assets. For Node, the audio harness uses that same
directory by default. Pass `--core PATH_TO_WASM` to test another generated core
with its matching `turbojev_wasm.js` alongside it. Historical validation reports
retain the artifact paths used at the time they were recorded.

## Real GGUF audio in WASM: Node, Windows x64, CPU

On 2026-10-04, LFM2-Audio-1.5B Q8_0 and its matching Q8_0 projector
completed four decisions through the actual Wllama 3.8.1 WASM binary, the
TurboJev llama.cpp adapter, and the compiled Rust/WASM core. Node 24.18.0
provided the WASM host. No native inference executable, HTTP inference server,
or canned scores were used. A disk-backed virtual filesystem avoided retaining
a second full copy of the GGUF files.

Two execution defects were fixed: mtmd's audio preprocessing needs extra
prestarted pthread workers beyond the inference pools, and the WASM
constructor's decision-scoring branch previously ignored the requested cyclic
order policy. With four inference threads, twelve workers were prestarted.
This preserves the requested `nThreads` instead of increasing inference threads
to hide the worker shortage. Audio now rejects missing cross-origin isolation
before model loading.

| Recording          | Candidate order | Answer | Conditional probability | Evaluation time |
| ------------------ | --------------- | ------ | ----------------------- | --------------- |
| Turn on the light  | Normal          | on     | 0.8250                  | 123.81 s        |
| Turn on the light  | Reversed        | on     | 0.8250                  | 109.82 s        |
| Turn off the light | Normal          | off    | 0.7823                  | 103.89 s        |
| Turn off the light | Reversed        | off    | 0.7823                  | 87.76 s         |

All four typed responses reported two cyclic evaluations and zero generated
tokens in the shared response contract. Candidate probabilities were identical
under reversal. Timing includes two full model evaluations and excludes loading;
these CPU/WASM observations are not browser or GPU performance claims.
The [recorded report](validation/multimodal-gguf-audio-wasm-node.json) contains
complete responses and SHA-256 hashes of the model, projector, WASM binaries,
and audio inputs.

To reproduce, extract `npm pack @wllama/wllama@3.8.1` into
`.cache/wllama-3.8.1/package`, prepare the website WASM, and run:

```powershell
node --experimental-wasm-jspi scripts/validate_multimodal_wasm_audio.mjs `
  --model path/to/LFM2-Audio-1.5B-Q8_0.gguf `
  --mmproj path/to/mmproj-LFM2-Audio-1.5B-Q8_0.gguf
```

The runner writes `.cache/multimodal-validation/audio-wasm-node.json` and exits
with a failure if the semantic or order checks fail. Thirteen adapter tests and
the compiled-WASM protocol regression cover worker reservation, single-thread
settings, missing isolation, media transport, and opt-in cyclic handling.

This verifies real audio computation in WASM. **The corrected browser page
still needs a fresh browser run.** Node validation does not qualify browser
download caching, the browser Worker messaging bridge, WebGPU, other OS/CPU
targets, or arbitrary speech tasks. The page above imports the corrected
adapter and rebuilt core, with cache versions updated for this change.

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
