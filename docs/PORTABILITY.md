# TurboJev portability contract

TurboJev separates **portable semantic logic** from **host/runtime integration**.
The portable workspace must compile without depending on a specific desktop,
mobile, server, or browser operating environment.

## Portable layer

The following crates are host-agnostic:

- `turbojev-schema`
- `turbojev-calibration`
- `turbojev-tokenizer`
- `turbojev-prompt`
- `turbojev-model`
- `turbojev-specialized`
- `turbojev-training-data`
- `turbojev-specialized-artifact`
- `turbojev-specialized-eval`
- `turbojev-server-protocol`
- `turbojev-server-execution`
- `turbojev-backend`
- `turbojev-engine`
- `turbojev-core`
- `turbojev-wire`
- `turbojev`
- `turbojev-backend-mock`
- `turbojev-backend-custom`
- `turbojev-compat`
- `turbojev-conformance`
- `turbojev-runtime`
- `turbojev-sequence`
- `turbojev-runtime-conformance`
- `turbojev-projection`
- `turbojev-runtime-projection`
- `turbojev-benchmark`

They may operate on values, byte/token identifiers, strings, JSON-compatible
state, candidate logits, and abstract capabilities. They must not assume how a
model is stored, downloaded, mapped into memory, accelerated, or scheduled by an
OS.

## Forbidden in portable crates

Portable crates do not directly use:

- filesystem APIs;
- sockets or HTTP clients;
- environment variables;
- subprocesses;
- `target_os` or `target_arch` branches;
- Windows, POSIX, Apple, Android, JNI, browser, or WASM APIs;
- CUDA/Metal/Vulkan/WebGPU runtime APIs;
- a concrete model runtime such as llama.cpp or Candle.

`scripts/audit_portability.py` enforces the easy-to-detect parts of this rule.

## Adapter layer

Concrete execution will be implemented in optional adapter crates. Examples:

```text
turbojev-runtime-llamacpp
turbojev-runtime-candle
turbojev-runtime-webgpu
turbojev-runtime-user
```

An adapter can be platform-specific internally, but it is **never required by
turbojev-core**. Applications choose the adapter that exists on their target.
A new model architecture or file format therefore does not require changing the
decision engine.

## Model universality

"Universal model support" means the TurboJev semantic engine has no allowlist of
model families. Any open-weight model can participate when a runtime adapter can
provide the minimum scoring contract required for the selected path:

1. render the TurboJev prompt IR into the model's expected template;
2. tokenize it with that model's tokenizer;
3. score candidate outputs exactly;
4. return ordered candidate logits/scores.

The fast path uses next-token logits. When stable single-token codes are not
available, the tokenizer module can discover fixed-width, prefix-free multi-token
codes and a capable runtime can score their sequences exactly.

## Human language universality

The semantic core never translates or rewrites user text. `state`,
`instructions`, and candidate descriptions are preserved verbatim. Prompt IR is
language-neutral; the selected model adapter owns any model-facing instruction
phrasing needed by that checkpoint. This keeps TurboJev usable with multilingual
or language-specialized open-weight models.

## Host language universality

`turbojev-wire` is the canonical JSON boundary. Future Node/TypeScript, browser,
Python, .NET, Swift, Kotlin, Dart, JVM, and C-compatible bindings must delegate
to the same Rust contracts rather than reimplementing decision semantics.


## Runtime bridge rule

`turbojev-runtime` is itself portable and is included in the OS/browser-coupling audit. Concrete adapters such as future `turbojev-runtime-llamacpp`, `turbojev-runtime-webgpu`, `turbojev-runtime-candle`, or host-owned executors may have platform/runtime dependencies, but they must remain outside the portable crate set.

## Concrete adapters are not portable-core dependencies

A concrete adapter may need a native compiler, accelerator runtime, browser API,
or host packaging logic. That does not weaken the portable-core rule: adapters
are leaves of the dependency graph and never become dependencies of the core.

`turbojev-runtime-llamacpp` demonstrates this policy. Its Rust-facing logic is
cross-platform and its optional native shim is C++17 with no `target_os`
branches in TurboJev Rust code. Applications build/package the shim and
llama.cpp for their target platform; the canonical core remains unchanged.
