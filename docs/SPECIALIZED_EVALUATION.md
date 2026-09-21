# Specialized evaluation and external-run interchange

TurboJev v0.20 adds a portable evaluation layer for comparing a generic open-weight checkpoint with a decision-specialized checkpoint on the **same semantic benchmark**. It does not load models, train weights, or choose a training framework.

## Contracts

Three versioned artifacts are introduced:

- `turbojev.specialization-run.v1`: one externally measured model run;
- `turbojev.specialization-report.v1`: paired generic-vs-specialized metrics and deltas;
- `turbojev.specialization-bundle.v1`: SHA-256 references tying plan, benchmark, runs, report, recipes, manifests and calibration profiles together.

The portable implementation lives in `turbojev-specialized-eval`. Filesystem/CLI concerns live separately in `turbojev-specialized-report`.

## External run

Any harness may produce a run: Rust, Python, llama.cpp tooling, a browser runtime, or another environment. The JSON must contain raw candidate logits and semantic candidate keys, not a framework-specific tensor format.

Each observation pins:

- stable observation/family/case/question ids;
- variant kind: `baseline`, `candidate-order`, or `paraphrase`;
- semantic candidate keys in exact logit order;
- raw candidate logits and reference target key;
- positive sample weight;
- end-to-end decision latency;
- peak-memory measurement.

The run also records model prompt/tokenizer/calibration provenance and the measurement environment: backend, host, quantization, execution path and optional runner version.

## Fair-pair gate

A generic/specialized comparison is rejected unless both runs use:

- the same evaluation-plan id;
- the same benchmark-suite id;
- the same observation ids and semantic metadata;
- the same prompt IR, renderer profile and tokenizer revision;
- the same backend, host, quantization and execution path.

The models and revisions themselves are expected to differ.

This gate makes a latency or robustness delta auditable instead of silently comparing unrelated runs.

## Headline quality metrics

Accuracy, NLL, multiclass Brier and top-label ECE are computed **only from baseline observations**. Candidate-order and paraphrase variants are not mixed into the headline sample distribution.

Probability metrics reuse `turbojev-calibration::evaluate_calibration` with the temperature declared by each run. A fitted temperature must carry a profile id; an unfitted `T=1` run remains explicitly unfitted.

## Robustness metrics

Controlled variants are reported separately.

For each variant family TurboJev computes:

- `invariance`: fraction of variant predictions equal to the corresponding baseline semantic prediction;
- `accuracy`: fraction of variants still equal to the reference semantic target.

Candidate order is interpreted semantically through candidate keys, so a correct permutation such as `[shipping, billing]` is not confused with raw logit position.

## Performance metrics

The report includes deterministic latency summary statistics, decisions/second, and maximum reported peak memory. Deltas are always encoded as:

```text
specialized - generic
```

No implicit winner label is stored. For example, lower NLL/latency and higher accuracy/throughput have different preferred directions, so the report preserves the measurements rather than collapsing them into one score.

## Reproducible experiment bundle

`turbojev.specialization-bundle.v1` stores SHA-256 references to all evidence required to reconstruct an experiment:

```text
evaluation plan
benchmark suite
external generic/specialized runs
canonical report
training recipe(s)
specialized model manifest(s)
calibration profile(s)
```

The bundle is metadata only. It does not download artifacts and does not introduce storage/network behavior into portable crates.

## Host CLI

The optional host tool reads external JSON runs and writes the canonical report:

```text
turbojev-specialized-report evaluate \
  evaluation-plan.json \
  report.json \
  generic-run.json \
  specialized-run.json
```

For plans containing multiple model pairs, pass all required run files. The CLI is deliberately outside the portable core.

## Training boundary

Nothing in v0.20 changes the Phase 10 boundary:

```text
TurboJev model-bound export
        │
        ▼
external trainer selected by user
        │
        ▼
open-weight specialized checkpoint
        │
        ▼
external evaluation harness
        │
        ▼
turbojev.specialization-run.v1
        │
        ▼
turbojev-specialized-eval
```

PyTorch, TRL, Axolotl, MLX, CUDA and similar trainer stacks remain external and optional.
