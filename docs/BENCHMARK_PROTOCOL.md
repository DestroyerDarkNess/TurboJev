# Benchmark protocol

`turbojev-benchmark` defines a runtime-independent, machine-readable benchmark
contract. It deliberately does not load models or inspect the operating system.

## Contract

Current suite identifier:

```text
turbojev.benchmark.v1
```

A suite contains canonical `EvaluationRequest` values. The committed reference
fixture covers `choice`, `noul`/`boolean`, `score`, structured state,
multi-question requests, and multiple human languages.

A run records:

- exact suite id;
- model/revision/quantization metadata supplied by the host;
- backend, prompt profile and tokenizer profile;
- exact `ExecutionPlan`;
- warmups/repetitions;
- optional opaque host description;
- per-sample total/prefill/scoring time;
- optional input-token count;
- an answer fingerprint for parity checks.

`generated_tokens` must be zero for the direct TurboJev path.

## Why timing is outside the core

Different hosts expose clocks and accelerator synchronization differently.
TurboJev therefore standardizes benchmark records, not the system clock. A
runtime runner measures according to its platform/runtime and writes the common
report format.

## Reference fixture

`benchmarks/fixtures/reference-semantic-v1.json` is intentionally model-neutral.
It contains English, Spanish, French and Japanese examples so adapters cannot
silently assume ASCII or one human language.
