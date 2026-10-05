# Direct model comparison

This Windows x64 diagnostic compares the published Python package against
direct llama.cpp inference. It is not a demo dependency or a new public API.
The direct executable does not call TurboJev's engine, scorer or native shim.
It loads the same GGUF and the same llama.cpp runtime DLLs from the installed
Python wheel. Its headers/import libraries must match the pinned native build.

Three controls use identical system/user messages, option order, model Jinja
template, `enable_thinking=false`, CPU load request, and context/thread settings:

1. TurboJev `evaluate`, with single-pass candidate scoring.
2. Direct llama.cpp with the same answer prefill. Record candidate-token logits,
   conditional softmax and unrestricted greedy generation from that boundary.
3. Direct greedy generation without the assistant prefill. This deliberately
   changes one prompt boundary, while preserving the messages and template.

Candidate IDs are obtained from full prompt tokenization, checking that each
letter extends the exact prefix by one token. They are never hard-coded. All
vocabulary tokens are available during generation; there is no grammar, action
mask, model-choice correction or host-selected movement. Generation stops at a
complete requested JSON object/EOG or 32 tokens. Invalid and unfinished answers
are counted as failures. Timings include different work, so they are not a
like-for-like performance comparison or a measure of thinking mode.

## Reproduce from the repository root

With the source and native build produced by the local 0.29.2 release:

```powershell
python -m pip install --index-url https://pypi.org/simple turbojev==0.29.2
cmake -S scripts/native-decision-audit -B .cache/direct-decision-build -A x64 `
  -DLLAMA_SOURCE=D:/Projects/TurboJev/.cache/real-validation/llama.cpp `
  -DLLAMA_BUILD=D:/Projects/TurboJev/artifacts/llama-build/win-x64
cmake --build .cache/direct-decision-build --config Release
python scripts/compare_direct_decisions.py `
  --model C:/Users/Administrator/.cache/turbojev/Qwen3-0.6B-Q4_K_M.gguf `
  --direct .cache/direct-decision-build/Release/direct_decision.exe `
  --output docs/validation/direct-vs-turbojev-qwen-0292.json
```

Adjust paths to your checkout/model. The runner imports the Snake serializer
from `other/TurboJev.Demos/snake-python`, which must be present, and records
requests, prompt tokens, verified candidate IDs, logits, complete generated
answers, probabilities, model hash and native DLL hashes. The pinned llama.cpp
revision is `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`.

Agreement with direct inference establishes a limited scoring/mapping check.
It does not establish an optimal prompt or prove that the model cannot solve a
task under other instructions, a reasoning mode or a specialized checkpoint.
