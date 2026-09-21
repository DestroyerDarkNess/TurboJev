# Runtime qualification

TurboJev does not infer production capabilities from a runtime name. `fusedBatch`, cancellation, and hard-deadline claims are deployment evidence, not model-family constants.

## v0.26 contract

`turbojev.runtime-qualification.v1` binds evidence to one exact target:

- model source + SHA-256 + optional revision/quantization;
- llama.cpp revision and TurboJev shim ABI;
- backend and backend revision;
- hardware architecture/device/driver/OS;
- numerical tolerance and conformance summary;
- the capabilities that the evidence actually authorizes.

A host that requests `execution.llamacppMultiRootFused=true` must also provide `execution.llamacppQualification` and `execution.runtimeFingerprint`. The manifest is accepted only when the exact fingerprint matches and the evidence passed. Changing the model file, llama.cpp revision, backend, driver, or hardware invalidates the qualification instead of silently carrying the capability forward.

`runtime_qualification.py fingerprint` computes the model SHA-256 and creates a target fingerprint. `runtime_qualification.py manifest` combines that fingerprint with raw independent-vs-fused observations into the versioned manifest.

## Cancellation boundary

Current llama.cpp exposes an abort callback in its context parameters, but upstream documents that callback as CPU-only. TurboJev therefore keeps `cooperativeCancellation=false` and `hardDeadlines=false` for the normal llama.cpp host path unless a future target-specific implementation carries control state into in-flight inference and proves cancellation semantics on that exact backend. Server soft cancellation/deadlines still stop queued or not-yet-started work.
