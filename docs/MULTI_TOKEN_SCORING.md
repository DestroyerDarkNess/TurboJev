# Exact multi-token candidate scoring

TurboJev v0.15 keeps one-token candidate codes as the preferred fast path and
adds an exact fallback for tokenizers where enough stable one-token codes cannot
be found at the real model answer boundary.

## Mathematical contract

For candidate tokens `c = [c1, ..., cn]`, TurboJev scores

```text
score(c) = sum_i(logit_i(ci) - logsumexp(full_vocab_logits_i))
```

The vocabulary normalizer is evaluated at every continuation state. Summing raw
logits is not a sequence probability. Normalizing only over the TurboJev
candidate tokens is also not equivalent.

## Prefix trie

`turbojev-sequence::CandidateTrie` requires non-empty, unique, prefix-free token
sequences. It evaluates each unique candidate prefix once. For `AA`, `AB`, and
`BA`, the state after the first `A` is shared by the first two candidates.

The crate reports:

```text
naive_token_steps
unique_prefix_evaluations
shared_token_steps_saved
```

These are execution-accounting metrics; they are not model confidence.

## Planning interaction

A discovered multi-token codebook changes which optimizations are mathematically
safe. In v0.15 TurboJev retains a valid shared prompt prefix, but uses serial task
scheduling and full-vocabulary projection for sequence traversal. The runtime
reports this as `BackendMetrics.effective_plan`, and the core exposes that actual
plan in response metadata.

Candidate-only output projection is intentionally not used for exact multi-token
scoring unless a future runtime can supply the exact full-vocabulary normalizer
(or an equivalent exact `logZ`) at every candidate continuation state.

## llama.cpp adapter

The exact sequence-scoring entry points were introduced in shim ABI v4 and remain part of the current ABI v5:

```text
tj_llama_sequence_scores_supported
tj_llama_score_sequences
tj_llama_score_suffix_sequences
```

The shim builds a candidate trie, walks isolated llama.cpp sequence state, and
backtracks candidate-prefix tokens after each branch. Native smoke coverage
checks exact manual log-likelihood, independent/shared-prefix parity, and the
single-token cancellation property.

## Universality

Production code contains no model-specific candidate token IDs. Candidate
symbols are discovered dynamically using the selected checkpoint's actual chat
template and tokenizer. The trie operates on opaque token IDs and therefore does
not know model family, weight format, operating system, accelerator, or human
language.
