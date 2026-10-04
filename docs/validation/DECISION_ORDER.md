# Candidate-order regression audit

Date: 2026-10-04. Model: Qwen3-0.6B Q4_K_M, SHA-256
`9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14`.

## Defect and change

The previous native adapter requested a bare option letter and read its logit
at the start of the assistant message. In the failing Snake state, rotating
the same four options changed the selected action to whichever option came
first, with probabilities from 90.0% to 99.3%.

The built-in renderers now request `{"answer": "<option letter>"}` and insert
the literal `{"answer": "` before reading candidate logits. This is answer
prefill, as used by TypeLLM's `Choice.label_prefill`; it does not generate JSON,
choose an action, reorder candidates, or average additional evaluations.
The tokenizer still verifies each candidate at this exact continuation point.
Custom renderers can leave `assistant_prefix` empty.

Native text, native mtmd, Transformers.js text, and llama.cpp WASM use the same
prefill boundary. Native shim ABI 8 adds the media prefill parameter; mixing
old and new native libraries is unsupported and rejected by the ABI check.

## Real measurements

The source-built Windows CPU C ABI used the actual Rust engine and llama.cpp.
`orderRobustness` was `none`; every decision reported zero generated tokens
and zero additional order evaluations. No host chose or replaced an answer.

| Cases | Rotations | Correct |
| --- | ---: | ---: |
| Replacement routing, billing, software crash, traffic color, explicit food direction | 4 per case | 20/20 |
| Original Snake wall state, with raw coordinates | 4 | 0/4 |

Repeating the complete native audit with the cached Qwen3-0.6B Q8_0 also gave
20/24, including 0/4 on the wall case. Changing quantization did not repair that
failure. The Q8 requests, responses, model checksum, and timings are retained
in [the Q8 report](decision-order-qwen-q8.json).

The wall case now selects `up` regardless of where `up` appears. It remains a
wrong answer, but is distinct from selecting the first option. This audit does
**not** qualify Qwen3-0.6B Q4_K_M for autonomous Snake or claim that spatial
reasoning is solved. Its 0/4 result is retained in the report.

An exploratory native reasoning run was stopped after 1,537 generated tokens
because it repeatedly reconsidered and confused the coordinate axes. It did
not complete its answer and therefore is neither a passing decision test nor
evidence that completed reasoning cannot solve the state. No reasoning or
host movement heuristic was added to production as a substitute for a fix.

The actual Wllama 3.8.1 WASM binary, JavaScript adapter, and Rust WASM core also
returned `shipping` in all four positions of the replacement example, with
`dog` and unrelated departments as distractors. This ran on CPU in Node 24.18.0;
it is not a browser or WebGPU qualification. Latencies are recorded for
reproducibility, not presented as a controlled performance comparison.

- [Full native requests, responses, and timings](decision-order-qwen.json)
- [Full Node WASM requests, responses, and timings](decision-order-qwen-wasm.json)

## Reproduce native inference

Build the current shim and `turbojev-capi` with the `llamacpp` feature. Supply
the C ABI library and every required DLL search directory:

```powershell
python scripts/validate_decision_order.py `
  --library target/release/turbojev_capi.dll `
  --library-dir .cache/real-validation/multimodal-native-x64-fresh/lib/Release `
  --library-dir .cache/real-validation/multimodal-native-x64-fresh/bin/Release `
  --library-dir .cache/real-validation/multimodal-native-x64-fresh/llama.cpp/tools/mtmd/Release `
  --model C:/models/Qwen3-0.6B-Q4_K_M.gguf `
  --output .cache/decision-order.json
```

This script records incorrect decisions instead of hiding them. Unit tests
cover exact prefill placement, unchanged candidate IDs, legacy empty-prefill
messages, and the WASM request configuration; they are not model-quality tests.

For the real Node WASM audit, extract Wllama 3.8.1 to
`.cache/wllama-3.8.1/package`, build the Rust WASM assets, then run:

```powershell
node --experimental-wasm-jspi scripts/validate_multimodal_wasm_audio.mjs `
  --text --model C:/models/Qwen3-0.6B-Q4_K_M.gguf `
  --output .cache/decision-order-wasm.json
```
