# TurboJev website

A static landing page, local inference playground, and 13 documentation pages. Serve `website/` with any static HTTP server. No application server or API key is needed.

```powershell
python -m http.server 4173 --directory website
```

Open http://localhost:4173. Use localhost or HTTPS for browser workers, WebGPU, and clipboard access.

## Build

Prerequisites: Rust with the `wasm32-unknown-unknown` target, wasm-bindgen CLI 0.2.100, Node.js, and the browser sample's pinned dependencies.

```powershell
# Recompile Rust/WASM, bundle the runtime, and generate documentation.
pwsh scripts/prepare-website-demo.ps1

# Rebuild JS and documentation only (requires the prepared WASM files).
node website/build.mjs
```

The PowerShell script uses the cached wasm-bindgen installed by `python samples/build.py web`, or a CLI on PATH. It uses Cargo metadata to locate the compiled WASM. It does not copy an older sample binary over the current runtime.

## Edit

- `index.html`, `styles.css`: landing, playground, and documentation styling.
- `app.js`: navigation, search dialog, keyboard controls, and code copying.
- `src/docs-content.html`: documentation source; `build.mjs` generates `docs/*.html` and the search index. Each topic has its own URL. Legacy `docs.html#python` links redirect to the corresponding page.
- `demo.js`: model catalog, presets, download progress, media decoding, and UI state.
- `demo-worker.js`, `src/runtime.ts`: inference worker and runtime adapters. Rust/WASM constructs the final typed responses.
- `vendor/`: deployable runtime, WASM, Prism syntax highlighting, licenses, and content hashes. Rebuild after changing sources. Asset versions prevent stale cached WASM from being paired with new JavaScript.

Deploy the entire `website/` directory as the hosting output directory. The site uses relative links and works under a subdirectory. If using GitHub Pages, upload this directory as the Pages artifact; the Pages branch UI does not offer an arbitrary `website/` folder.

## Playground behavior

Text: Qwen2.5 0.5B and Qwen3 0.6B use the existing TurboJev tokenizer/runtime adapter.

Images: CLIP B/32 and B/16 score image–text similarity. SmolVLM 256M and SmolVLM2 256M score answer codes with model-derived token IDs and the model's chat template.

Audio: CLAP classifies sounds from a mono 48 kHz clip, limited to 10 seconds. It does not transcribe speech or replace the native LFM2.5-Audio example.

Video: the host samples four frames from clips up to 60 seconds. CLIP or SmolVLM2 scores each frame independently and the adapter averages raw logits. This is scene classification, not temporal reasoning; the audio track is not analyzed.

All media files are limited to 24 MB. Uploaded media stays in the browser. Model files download from Hugging Face only after Load; sample image/audio downloads require a separate click. ONNX Runtime is fetched from a pinned jsDelivr URL. Model performance and WebGPU availability depend on the device. The interface reports real elapsed time, errors, and relative probabilities; it does not substitute canned results.

## Dependencies

Transformers.js 4.3.0 and ONNX Runtime Web use the versions locked in the browser sample. Prism 1.30.0 is bundled locally with its MIT license. Third-party notices are in `vendor/licenses/`.
