# Async execution model

TurboJev separates synchronous and asynchronous execution instead of making the
portable core depend on one async runtime.

## Why two backend traits exist

`ScoringBackend` is the zero-overhead synchronous contract. It is appropriate
for native runtimes that can score a batch directly on the calling thread.

`AsyncScoringBackend` is the non-blocking contract used by browser/WebGPU,
JavaScript-hosted runtimes, mobile APIs, remote/custom hosts, and any executor
that naturally returns a future.

Neither trait depends on Tokio, async-std, smol, JavaScript APIs, JNI, Obj-C, or
an operating system.

```text
TurboJevEngine       -> ScoringBackend      -> sync runtime
TurboJevAsyncEngine  -> AsyncScoringBackend -> async runtime
```

Both engines share the same canonical schema, execution planner, calibration,
answer construction, compatibility profiles, and conformance rules.

## Futures are deliberately not `Send`

The public future aliases do not impose `Send`. A browser WebGPU device or host
object may be thread-affine. Requiring thread mobility in the portable API would
exclude legitimate runtimes.

Server adapters that need cross-thread execution may add stronger bounds in
their own outer crate.

## Runtime bridge

At the token/runtime layer the same split exists:

```text
RuntimeExecutor      -> RuntimeBackend -> ScoringBackend
AsyncRuntimeExecutor -> RuntimeBackend -> AsyncScoringBackend
```

Prompt rendering, exact answer-boundary tokenization, codebook discovery, and
shared-prefix factoring remain identical in both paths.

## Host bindings

Bindings should map their native async mechanism to `TurboJevAsyncEngine`:

- JavaScript/TypeScript: `Promise`
- Python: awaitable/coroutine
- C#: `Task`
- Swift: `async`
- Kotlin: coroutine/suspend wrapper

No binding is allowed to reimplement decision semantics.
