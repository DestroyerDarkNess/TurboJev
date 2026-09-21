# TurboJev server

TurboJev v0.21 introduced the loopback HTTP host, v0.22 added truthful cross-request scheduling, v0.23 added the native runtime-capability bridge plus deployment/lifecycle tooling, v0.24 added an explicit fused backend/runtime path plus memory-aware batch planning, and v0.25 adds an opt-in llama.cpp multi-root implementation for eligible single-token request groups. HTTP/networking remains host-only; semantic routing, typing, scoring, calibration, and runtime selection stay in their existing Rust layers.

## Architecture

```text
HTTP clients / local reverse proxy
   │
   ▼
turbojev-server                 host-only
   │                            TCP / threads / clocks / queue / lifecycle
   │                            request IDs / deployment validation
   ├──────────────► metrics
   │
   ├──────────────► turbojev-server-protocol
   │                   portable routes / profiles / batches / errors
   │
   └──────────────► turbojev-server-execution
                       portable cross-request capabilities / planning / parity
   │
   ▼
turbojev-host-native
   │   NativeSession::execution_capabilities()
   │   NativeSession::evaluate_fused(...)
   ▼
TurboJev core/runtime
```

The stock native server owns exactly one thread-affine `NativeSession`. Concurrent HTTP connections never clone the model session.

## Endpoints

- `GET /healthz` — process liveness; remains 200 while draining.
- `GET /readyz` — admission readiness; changes to 503 before graceful drain.
- `GET /v1/models`
- `GET /v1/server/metrics`
- `POST /v1/evaluate` — canonical TurboJev wire contract.
- `POST /v1/systemone` — TypeSafe/System One compatibility profile.
- `POST /v1/ai-sdk/evaluate` — AI SDK evaluation compatibility profile.
- `POST /v1/vercel/evaluate` — AI SDK alias.
- `POST /v1/openrouter/evaluate` — canonical TurboJev decision contract at a gateway-friendly alias.
- batch forms append `/batch` as recorded by `tests/server/routes.json`.

The OpenRouter alias is **not** an OpenAI `/chat/completions` emulator. TurboJev does not convert candidate-logit decisions into generated chat text.

## Runtime capability bridge

v0.22 deliberately kept cross-request capability truth inside the server adapter. v0.23 moves that truth to the loaded native session:

```text
NativeSession
   │
   ├── execution_capabilities()
   │      ├── fusedBatch
   │      ├── cooperativeCancellation
   │      ├── hardDeadlines
   │      └── maxFusedItems
   │
   └── evaluate_fused(...)
```

The server maps this host-native contract into `turbojev-server-execution`. It no longer infers capability from a model file name, architecture string, or runtime label.

The deterministic fixture in v0.24 reports `fusedBatch=true` for conformance, while stock llama.cpp still reports:

```json
{
  "fusedBatch": false,
  "cooperativeCancellation": false,
  "hardDeadlines": false,
  "maxFusedItems": null
}
```

For the fixture, `NativeSession::evaluate_fused` delegates to `TurboJevEngine::evaluate_fused` and the fixture's explicit `FusedScoringBackend` implementation. For stock llama.cpp it returns `UNSUPPORTED_CAPABILITY`. llama.cpp's existing parallel suffix execution remains an optimization *inside one EvaluationRequest* and is not cross-request batching.

A production runtime may set `fusedBatch=true` only when one real runtime operation accepts multiple independent requests and its output passes `verify_fused_runtime_executor` plus the server-level `verify_fused_parity` gate.

## Fair scheduling

A client batch is a scheduler job, not an indivisible worker command. The dedicated session thread processes at most `fairnessQuantum` items from one job before rotating unfinished work behind other pending jobs.

```text
fairnessQuantum = 1

batch A: A0 A1 A2 A3
batch B: B0 B1

worker:
A0 -> B0 -> A1 -> B1 -> A2 -> A3
```

Response order inside each batch remains identical to request order. If a future runtime truthfully supports fused execution, each turn is bounded by the smallest of remaining items, fairness quantum, host `maxFusedItems`, and backend `maxFusedItems`.

## Deadlines and cancellation

`transport.sessionTimeoutMs` bounds queue plus execution wait. A client may request a shorter deadline with:

```text
X-TurboJev-Timeout-Ms: 2500
```

The value is clamped to the server maximum. Work that has not started is skipped after caller timeout. The stock native runtimes still do not advertise cooperative in-flight cancellation or hard deadlines, so an already-started model call is not claimed to be interruptible.

## Liveness, readiness, and graceful drain

The lifecycle is explicit:

```text
ready -> draining -> stopped
```

While ready:

```text
GET /healthz -> 200
GET /readyz  -> 200
new inference -> admitted subject to normal limits
```

After graceful shutdown begins:

```text
GET /healthz -> 200
GET /readyz  -> 503
new inference -> 503 SERVER_DRAINING
already admitted work -> allowed to finish
```

Embedders use `ShutdownToken`:

```rust
let token = ShutdownToken::new();
let server_token = token.clone();

// run HttpServer::serve_until(server_token) on the server thread
// ... later ...
token.request();
```

The listener stops accepting before the drain. Existing connections/jobs receive up to `transport.drainTimeoutMs`. If they do not finish within that grace period, `serve_until` returns `DRAIN_TIMEOUT` rather than claiming a clean drain.

The standalone CLI intentionally does not add a platform-specific signal dependency to the Rust core. Process supervisors may terminate the CLI normally; applications that require controlled draining should embed the host API or add signal wiring in their platform layer.

## Structured request IDs

Every HTTP response carries:

```text
X-TurboJev-Request-Id: ...
```

Clients or a trusted local proxy may provide `X-TurboJev-Request-Id` or `X-Request-Id`. Accepted IDs are restricted to 1–128 safe ASCII token characters to prevent response-header injection. When no ID is supplied, the host generates a local monotonic ID such as:

```text
tjreq-0000000000000001
```

The ID is correlation metadata only; it does not affect semantic inference or model input.

## Reverse-proxy deployment contract

TurboJev remains loopback-bound even in reverse-proxy mode. Remote/public exposure is intentionally delegated to a colocated proxy that owns TLS, authentication, rate limiting, and internet-facing policy.

```json
{
  "bind": "127.0.0.1:3217",
  "deployment": {
    "mode": "reverse-proxy",
    "requireForwardedHttps": true
  }
}
```

For non-probe routes, `requireForwardedHttps=true` requires:

```text
X-Forwarded-Proto: https
```

`/healthz` and `/readyz` remain directly probeable by a local supervisor. TurboJev does not trust `X-Forwarded-For` for authorization, does not terminate TLS, and does not interpret proxy headers as proof of client identity.

This keeps the safe default intact: binding `0.0.0.0`, a LAN address, or a public address is still rejected.

## Overload metrics

`GET /v1/server/metrics` reports local counters and the currently loaded executor capabilities:

```text
lifecycle
submittedJobs
submittedRequests
completedJobs
completedRequests
rejectedJobs
callerDeadlines
cancelledBeforeExecution
serialTurns
fusedTurns
inflightJobs
inflightHighWater
```

These counters never leave the server automatically; TurboJev has no mandatory telemetry.

## Admission and backpressure

Transport concurrency is bounded by `maxConnections`, admission to the session worker is bounded by `queueCapacity`, and the scheduler's pending deque is bounded by the same admission capacity. A full queue fails immediately with:

```text
503 SERVER_BUSY
```

No unbounded internal request queue is introduced.

## Reproducible load harness

v0.23 adds a stdlib-only external harness:

```text
python scripts/server_load_harness.py \
  --plan tests/server/load-plan.json \
  --out load-report.json
```

The plan and report use:

```text
turbojev.server-load-plan.v1
turbojev.server-load-report.v1
```

Reports include status counts, min/p50/p95/p99/max/mean latency, wall time, and requests/sec. The harness refuses non-loopback targets unless `--allow-remote` is provided explicitly. It is a host benchmark tool, not part of the inference core.

## Configuration

```json
{
  "bind": "127.0.0.1:3217",
  "load": {"model": "/models/model.gguf"},
  "limits": {
    "maxBodyBytes": 1048576,
    "maxBatchItems": 64
  },
  "transport": {
    "maxConnections": 64,
    "maxHeaderBytes": 16384,
    "queueCapacity": 128,
    "ioTimeoutMs": 30000,
    "sessionTimeoutMs": 120000,
    "drainTimeoutMs": 30000,
    "fairnessQuantum": 1,
    "maxFusedItems": 8
  },
  "deployment": {
    "mode": "loopback",
    "requireForwardedHttps": true
  }
}
```

Run with `turbojev-server --config server.json`. Concrete llama.cpp loading remains opt-in through the crate feature; `fixture://` remains available for deterministic conformance and binding tests.


## v0.24 fused backend/runtime contract

Cross-request fusion is now a first-class operation rather than a label applied to repeated single-request calls.

```text
EvaluationRequest[]
      │
      ▼
TurboJevEngine::evaluate_fused
      │
      ▼
FusedScoringBackend::score_fused
      │
      ▼
(optional RuntimeBackend bridge)
FusedRuntimeExecutor::score_fused
```

`turbojev-runtime-conformance::verify_fused_runtime_executor` compares the fused runtime output against independent reference scoring. Production adapters must pass that gate before advertising `fusedBatch=true`. The deterministic fixture implements the fused backend SPI only so the server/host/core plumbing can be exercised end-to-end; it is not performance evidence for a model runtime.

v0.25 adds a separate llama.cpp multi-root operation for arbitrary independent prompt roots. Each decision root has an isolated sequence ID and position space; no shared-prefix branch state is reused. This path is intentionally distinct from the older intra-request shared-prefix parallel-suffix optimization.

The native operation is **not enabled by default**. A deployment must opt in only after the concrete llama.cpp revision, model, backend and hardware pass the independent-vs-fused conformance suite. Multi-token candidate codebooks or groups exceeding the runtime's root/token limits automatically fall back to serial cross-request execution.

### Resource-aware fused planning

Capabilities may provide:

```text
estimatedFusedItemBytes
```

and the server may configure:

```json
{
  "transport": {
    "maxFusedItems": 8,
    "fusedMemoryBudgetBytes": 67108864
  }
}
```

When both memory values are known, the fused turn is additionally bounded by `floor(fusedMemoryBudgetBytes / estimatedFusedItemBytes)`. Unknown estimates never invent a memory claim.

### Benchmark provenance bundle

After running the load harness, create a deterministic evidence manifest:

```text
python scripts/server_benchmark_bundle.py \
  server.json load-plan.json load-report.json benchmark-bundle.json
```

The `turbojev.server-benchmark-bundle.v1` document records SHA-256 references to the exact server config, load plan and load report used for a measurement.

## v0.25 llama.cpp multi-root activation

Native load options accept:

```json
{
  "execution": {
    "llamacppMultiRootFused": true
  }
}
```

The default is `false`. This flag does not bypass shape checks: `NativeSession::can_evaluate_fused()` still verifies candidate width, projection mode, root capacity and total prompt-token capacity for each proposed fused turn. Hosts should treat the flag as evidence that the target deployment has already passed TurboJev conformance, not as a generic performance toggle.

Hard/cooperative cancellation remains false for the generic llama.cpp path through v0.28. TurboJev does not generalize CPU abort callback behavior into an accelerator-independent guarantee.


## v0.26 evidence-bound activation

A server/native host may request llama.cpp multi-root only when `llamacppMultiRootFused=true` is accompanied by `llamacppQualification` and `runtimeFingerprint`. TurboJev validates the manifest and requires an exact fingerprint match before enabling the underlying FFI capability. See `RUNTIME_QUALIFICATION.md`.
