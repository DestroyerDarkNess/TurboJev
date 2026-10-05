# Published-package numeric smoke check

This is a real run using **TurboJev 0.30.1 installed from PyPI in a fresh
environment**, Qwen3 0.6B Q4_K_M, explicit Vulkan GPU, and thinking disabled.
The first eight numeric cases completed with eight correct answers and no
execution errors. This is a small installation/API smoke subset, not a new
full-suite accuracy result.

`manifest.json` records the package, model/dataset hashes and configuration;
`results.jsonl` retains every request, answer and gold comparison;
`summary.json` records measured results. `packages.json` contains independent
public-registry verification, including the three PyPI wheel hashes.

Other native builds and sample runs overlapped this check. Its raw times are
retained, but are not an isolated performance benchmark. The separate complete
[development benchmark](../0.30.0-development/README.md) has 128 numeric and
231 JevBench cases and remains explicitly identified as a development build.
