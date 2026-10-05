# Real-model evaluations

Install the published package and supply a compatible text GGUF:

```powershell
python -m pip install -r evals/requirements.txt
python evals/run.py --model C:/models/Qwen3-0.6B-Q4_K_M.gguf --output evals/results/qwen --suite all
```

The runner includes TypeLLM's 128 numeric cases and the 231 pinned JevBench public tasks. Numeric cases use `generate` for integers, decimals, extraction, arithmetic, negative values and dependent decisions. JevBench uses `evaluate` for choice, noul and ordered scores. For noul, upstream's explicit true/false rubric is appended to the instruction because the existing noul contract does not accept named criteria. Expected values are checked after inference and never inserted into prompts. Outputs include every raw request, response or error, dataset/model hashes, package version, hardware, accuracy, Brier score, ECE, ordinal MAE and latency mean/p50/p95. Failed attempts stay in the accuracy denominator. ECE describes this dataset; it does not turn conditional probabilities into calibrated confidence.

Use `--suite numeric --thinking true --thinking-budget 512` to compare bounded reasoning with `--thinking false`. `--thinking auto` selects effort with an extra model decision and is supported by the numeric `generate` suite. `--device gpu` explicitly requests GPU. CPU is the default. `--limit 8` runs a smoke subset and marks both manifest and summary accordingly. `--offset` supports explicit subsets; no subset is presented as the full benchmark.

For pre-release development only, `--native-library PATH --library-dir DIR` selects a built C ABI library. The manifest labels this source as unpublished and records its hash. Normal runs import the registry wheel; no source checkout is used as the Python package.

Dataset provenance and licenses are recorded in `data/provenance.json` and adjacent LICENSE files. TypeLLM's published results, remote GPU environments and private model-server configuration are not TurboJev results and are not copied. Receipt extraction and permutation examples live in the separate TurboJev.Demos repository. The unit-test coverage map is in `docs/TYPED_FIELDS.md`.
