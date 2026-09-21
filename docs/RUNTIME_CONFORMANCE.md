# Runtime Conformance Kit

`turbojev-runtime-conformance` is the acceptance gate for concrete model
runtimes. A runtime is not considered TurboJev-conforming merely because it can
return numbers.

The kit verifies four classes of invariants.

## 1. Exact answer-boundary tokenization

Candidate symbols are checked through the checkpoint's actual prompt/chat
template, not by concatenating raw strings outside the template.

A candidate must:

- extend the exact encoded answer boundary;
- decode back to the same symbol;
- map to a unique token sequence;
- remain within the configured sequence width;
- form a prefix-free codebook when multi-token candidates are used.

## 2. Capability truthfulness

The declared runtime capabilities must support the requested execution plan.
Multi-token candidate codes are rejected unless the runtime advertises exact
sequence scoring.

## 3. Output shape and numeric validity

Every task must produce exactly one score row. IDs must be unique, each row must
contain exactly one value per candidate, and all values must be finite.

## 4. Reference-vs-optimized parity

Optimized paths such as shared prefix, parallel suffixes, quantization, and
candidate-only projection are compared against a reference path using explicit
absolute logit tolerances.

```text
reference path --------------------┐
                                   ├-> runtime parity report
optimized runtime path ------------┘
```

A runtime adapter's own crate is responsible for supplying the actual model
fixture and invoking these checks. TurboJev supplies the invariant machinery;
it does not bake a particular model or file format into the conformance crate.

## Required adapter test matrix

Before a first-party adapter is marked stable it should cover:

- at least two model/tokenizer families;
- Unicode and non-English text;
- single-token candidate codes;
- a multi-token candidate case when supported;
- one and many questions over a shared state;
- reference/optimized parity;
- malformed/unsupported-plan rejection;
- finite score validation;
- synchronous or asynchronous execution as appropriate.


## Selective output-head gate

Candidate-only projection has an extra acceptance gate in v0.9.
`verify_selective_projection` compares optimized logits against the frozen
full-vocabulary reference and verifies that the optimized output reports exactly
the number of candidate rows present in the prepared batch. Missing accounting
is a conformance failure even when the logits happen to match. This prevents
full-vocabulary + gather from being mislabeled as a selective optimization.

The tolerance is explicit. Deterministic dense fixtures use zero tolerance;
quantized/mixed-precision runtimes may use a measured non-zero tolerance in
their own conformance suite.
