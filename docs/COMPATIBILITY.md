# TurboJev compatibility profiles

TurboJev separates semantic compatibility from runtime implementation. All profiles map to the same canonical Rust types before inference.

## Profiles

### `turbojev`
Canonical TurboJev request/response. Boolean decisions serialize as `noul`; `boolean` is accepted as an input alias.

### `typesafe-systemone-v1`
Compatibility profile for the public TypeSafe System One vocabulary: shared `state`, named `questions`, and `choice` / `noul` / `score`. Choice and score responses include distributions; Noul represents P(true).

### `ai-sdk-evaluation-v4`
Compatibility profile for AI SDK evaluation vocabulary. Noul is exposed as `boolean`; its answer is `{ probability }`. Choice and score retain their typed values and probability distributions. Choice/Score confidence is mirrored under `providerMetadata.typesafe.confidence` for source compatibility, while `providerMetadata.turbojev` identifies the value as TurboJev-derived and records calibration provenance.

## Important confidence distinction

A generic open-weight model's normalized candidate distribution is not automatically equivalent to TypeSafe/Jev's separately trained confidence/calibration. TurboJev never hides this distinction: compatibility fields can be mirrored, but provenance remains present.

## No transport dependency

These profiles do not perform HTTP or require Node, a browser, sockets, environment variables, or an operating system. They are pure JSON transformations and compile with the portable core.
