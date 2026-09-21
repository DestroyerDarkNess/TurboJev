# ADR 0003: Candidate codes are discovered, never hard-coded

Status: Accepted

## Context

A token such as `A` does not have a universal vocabulary ID. Token boundaries can also change after the preceding prompt. Hard-coded IDs are fragile across model families, tokenizer revisions and chat templates.

## Decision

A single-token candidate code is valid only when the tokenizer proves:

```text
encode(symbol) = [token]
decode([token]) = symbol
encode(prefix + symbol) = encode(prefix) + [token]
```

The model backend discovers a codebook per tokenizer/prompt boundary. Future multi-token scoring is a separate exact fallback path.

## Consequences

- generic support for multiple model families
- no model-name conditionals for answer token IDs
- prompt/tokenizer changes fail loudly instead of silently changing probabilities
