# Multimodal compatibility and release status

Status: experimental integration, not a validated Jev-Omni release.
Reference reviewed: `akhilaaa3/Jev-Omni`, revision `c127654` (local checkout).

## What Jev-Omni actually accepts

The reference Python `predict` accepts text plus one image, audio file, or video.
Pillow decodes images to RGB; FFmpeg converts audio to mono 16 kHz, capped at
30 seconds; OpenCV samples 16 video frames. Codec availability depends on those
libraries. It does not promise every file format (PDF, office documents, archives,
arbitrary media codecs). It supports 2..256 options; quality above 20 is unproven.
Its trained 256-output head consumes hidden states, not vocabulary-token logits.
Loading its safetensors as an ordinary language model cannot reproduce that head.

Model storage formats are separate from media formats: GGUF is handled by the
native llama.cpp adapter, ONNX checkpoints by the Transformers.js adapter, and
Jev-Omni's safetensors plus `head.pt`/`runtime_buffers.pt` by its reference loader.
There is no universal loader for arbitrary `.pt`, safetensors or ONNX models.

## TurboJev support matrix

| Path | Text | Image / audio / video | Execution |
|---|---|---|---|
| llama.cpp / GGUF adapter | Yes | Explicitly rejected | Local native |
| Transformers.js / WASM token adapter | Yes | Explicitly rejected | Local browser |
| Jev-Omni worker | Implemented, unvalidated | Implemented, unvalidated | Reference Transformers + CUDA |
| Web `TurboJevRemote` | Server runtime | Server runtime | HTTP; requires a configured server |
| Rust, Node/TS, Python, C, .NET | Shared native session | Shared worker protocol | Desktop/server host |

The original loader requires CUDA. This does not make Jev-Omni inference work
locally on macOS, iOS, Android, WebGPU or WASM. Those clients can use a suitable
server. The worker transport uses portable Rust process APIs; model portability
is a separate requirement. No mobile subprocess execution is advertised.

## Shared media request

All bindings pass the same `evaluate` JSON, preserving bytes as base64:

```json
{
  "state": {
    "contract": "turbojev.media.v1",
    "text": "Optional context",
    "modality": "image",
    "base64": "<base64-encoded-file-bytes>"
  },
  "questions": {
    "scene": {
      "type": "choice",
      "instructions": "What is shown?",
      "criteria": {"cat": "A cat", "dog": "A dog"}
    }
  }
}
```

Use `audio` or `video` for other modalities. Limit: 24 MiB decoded media.
Only inline bytes are accepted; evaluation requests cannot open server files or
fetch arbitrary URLs. Ordinary structured JSON evidence remains supported.
Use `evaluate` for media; the string-only `classify` convenience method is unchanged.
Rust owns softmax, calibration metadata, choice/noul/score semantics and responses.
The worker returns raw trained-head logits and never generates an answer.

## Native setup

On a CUDA host, install the reference checkout's requirements and FFmpeg.
Use absolute paths for the interpreter, script and trusted reference checkout:

```json
{
  "model": "akhilaaa3/Jev-Omni",
  "worker": {
    "program": "C:/Python312/python.exe",
    "args": ["D:/Projects/TurboJev/scripts/jev_omni_worker.py",
             "--reference-dir", "D:/Projects/TurboJev/other/Jev-Omni"],
    "timeoutSeconds": 900
  }
}
```

The reference loader downloads the checkpoint and Gemma multimodal components
on first load, subject to model access permissions. The original weights require
about 50 GB before runtime overhead. Model files are not bundled with TurboJev.

Pass this load object to Node `TurboJev.load`, .NET `LoadAsync`, C
`tj_engine_open_json`, or Rust `NativeSession::open_json`. Python accepts
`TurboJev.load("akhilaaa3/Jev-Omni", worker={...})`. Node's normal GGUF path
resolver is bypassed when an explicit worker is configured.

## Browser / other operating systems

Run `turbojev-server --config server.json` on the CUDA host, with the load object
above under `load`. Configure `limits.maxBodyBytes` to accommodate base64 (for
example 40000000). For browsers, use a same-origin reverse proxy; the loopback
server does not add permissive CORS headers. Set transport timeouts appropriately.

```javascript
import { TurboJevRemote, mediaEvidence } from '@turbojev/web';
const engine = new TurboJevRemote('/turbojev');
const state = await mediaEvidence(fileInput.files[0], 'image', 'Optional context');
const result = await engine.evaluate({state, questions});
```

Python, C, .NET and Rust clients can also call `/v1/evaluate` using their ordinary
HTTP libraries. A remote connection is explicit; local WASM never silently sends
media elsewhere.

## Evidence and remaining release gates

Tests cover contract validation, byte preservation, subprocess protocol, typed
response construction, runtime rejection, timeout and worker error propagation.
Protocol fixtures are not evidence of actual image/audio/video inference.
The development machine reports Intel UHD Graphics; Jev-Omni weights remain LFS
pointers. Real CUDA inference, numerical parity against the reference, codecs,
and cross-platform deployment must pass before claiming full compatibility.
Publication jobs require repository variable `RELEASE_ENABLED=true`; keep it
unset until those gates and registry packaging gates are complete.
