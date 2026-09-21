# TurboJev — Plan maestro v0.28.2

## 1. Visión

TurboJev será una librería multiplataforma escrita en Rust para ejecutar decisiones semánticas tipadas al estilo Jev usando modelos de pesos abiertos. Debe ofrecer una API sencilla de clasificación y una API completa compatible por adaptadores con el patrón `state + questions -> typed probabilistic answers`.

TurboJev no será un fork de SemIf ni de Nimble. Toma sus mejores ideas y las convierte en una arquitectura más general:

- de SemIf: usar modelos congelados y leer directamente candidate logits sin generar texto;
- de Nimble: candidate-only projection, shared prefix, parallel suffixes y entrenamiento especializado sobre candidate logits;
- de TurboJev: backend SPI estable, calibración separada, candidate-code discovery dinámico, contratos versionados, FFI multi-idioma, ejecución multiplataforma y compatibilidad de wire protocol desacoplada del core.

Compatibilidad con Jev significa compatibilidad de contrato y semántica de tipos. No se afirmará equivalencia con los pesos, entrenamiento o calibración propietarios de Jev.

---

## 2. Principios no negociables

1. **Rust-first.** La lógica de dominio vive una sola vez en Rust.
2. **Nada de spaghetti.** Crates pequeños, responsabilidades únicas y dependencias dirigidas.
3. **Backend agnostic.** El core no conoce llama.cpp, wllama, Candle, WebGPU, CUDA, Metal ni ningún runtime concreto.
4. **Model agnostic.** No existe una allowlist de Qwen/Llama/Gemma/etc.; cualquier modelo de pesos abiertos puede integrarse cuando un runtime adapter pueda puntuar exactamente los candidatos.
4.1 **OS agnostic.** Ningún crate portable puede depender de Windows, Linux, macOS, Android, iOS, navegador, filesystem, sockets, procesos o variables de entorno.
5. **Zero-generation fast path.** La decisión optimizada no genera explicación ni JSON con el modelo.
6. **Typed by construction.** Nunca se puede devolver una opción fuera del schema.
7. **Shared-state first.** Una request ya representa varias preguntas sobre un solo estado.
8. **Calibration first-class.** Softmax condicional y confianza calibrada no se confunden.
9. **Prompt contracts versionados.** Un modelo especializado conoce exactamente el prompt con el que fue entrenado.
10. **Tokenizer-safe.** Nunca hardcodear IDs de tokens A/B/C por modelo.
11. **Reference path.** Siempre habrá un modo independiente simple para comparar corrección con paths optimizados.
12. **Privacy by default.** Inferencia local y cero telemetría obligatoria.
13. **Bindings finos.** Node/Python/.NET/etc. no reimplementan scoring ni schema.
14. **Unsafe aislado.** El workspace prohíbe unsafe; solo el futuro crate FFI podrá encapsularlo de forma auditada.
15. **Prompt IR neutral.** El core no impone inglés ni una plantilla de chat; cada adapter renderiza el IR semántico para su checkpoint.
16. **Wire contract único.** JS/TS/Python/C#/Swift/Kotlin/etc. delegan al mismo JSON/schema Rust, sin reimplementar reglas.

---

## 3. Producto final

### API sencilla

```ts
import { TurboJev } from "@turbojev/node";

const jev = await TurboJev.load({
  model: "hf://Qwen/Qwen3-0.6B-GGUF",
  device: "auto",
});

const result = await jev.classify(
  "My package never arrived",
  ["billing", "shipping", "technical", "other"]
);

result.choice;
result.probabilities;
```

### API tipada completa

```ts
const result = await jev.evaluate({
  state: {
    message: "My package never arrived",
    daysSinceShipment: 12
  },
  questions: {
    route: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: {
        billing: "Billing or payment problems",
        shipping: "Delivery and logistics problems",
        technical: "Technical problems",
        other: "None of the above"
      }
    },
    urgent: {
      type: "boolean",
      instructions: "Does this require urgent human intervention?"
    },
    severity: {
      type: "score",
      instructions: "How severe is this issue?",
      criteria: [
        "No meaningful impact",
        "Minor impact",
        "Significant impact",
        "Critical impact"
      ]
    }
  }
});
```

---

## 4. Arquitectura de alto nivel

```text
Host languages / applications
        │
        ├──────────────► turbojev-wire
        │                     │
        │                     ▼
        │               turbojev-schema
        │                     │
        │                     ▼
        │               turbojev-core
        │                │         │
        │                │         ├──► turbojev-calibration
        │                │         ├──► turbojev-engine
        │                │         ├──► turbojev-prompt (semantic IR)
        │                │         ├──► turbojev-tokenizer
        │                │         └──► turbojev-model
        │                │
        │                ▼
        └──────────► ScoringBackend SPI ◄──────── runtime adapters
                                             │
                                             ├── custom host runtime
                                             ├── optional GGUF adapter
                                             ├── optional Rust-native adapter
                                             ├── optional WebGPU/WASM adapter
                                             └── future runtimes/formats
```

El backend solo produce logits/scores de candidatos. El core aplica normalización/calibración y construye `choice`, `noul` y `score`.

**Regla nueva v0.2:** ningún runtime concreto forma parte de la dependencia obligatoria de `turbojev-core`. Incluso la carga/descarga del modelo se resuelve fuera del core mediante `ModelSource` opaco.

---

## 5. Workspace objetivo

```text
turbojev/
├── Cargo.toml
├── PLAN.md
├── SPEC.md
├── docs/
│   └── ARCHITECTURE.md
├── crates/
│   ├── turbojev/                    # facade
│   ├── turbojev-schema/             # request/response
│   ├── turbojev-core/               # orchestration
│   ├── turbojev-backend/            # SPI
│   ├── turbojev-calibration/        # probability/calibration
│   ├── turbojev-tokenizer/          # codebook/token boundaries
│   ├── turbojev-prompt/             # semantic prompt IR
│   ├── turbojev-model/              # opaque model descriptors
│   ├── turbojev-specialized/        # training/checkpoint contracts
│   ├── turbojev-training-data/      # semantic datasets/materialization
│   ├── turbojev-engine/             # execution planner
│   ├── turbojev-wire/               # canonical cross-language JSON boundary
│   ├── turbojev-backend-custom/     # arbitrary application-owned runtimes
│   ├── turbojev-runtime-llamacpp/   # optional GGUF adapter
│   ├── turbojev-runtime-webgpu/     # optional WASM/WebGPU adapter
│   ├── turbojev-runtime-candle/     # optional Rust-native adapter
│   ├── turbojev-server-protocol/    # portable server routes/batch protocol
│   ├── turbojev-server-execution/   # portable cross-request execution SPI
│   ├── turbojev-server/             # host-only HTTP compatibility
│   ├── turbojev-cli/
│   └── turbojev-ffi/
├── bindings/
│   ├── node/
│   ├── web/
│   ├── python/
│   ├── dotnet/
│   ├── swift/
│   └── kotlin/
├── tests/
│   ├── conformance/
│   ├── golden/
│   ├── multilingual/
│   ├── compatibility/
│   └── fuzz/
└── benches/
```

---

## 5.1 Universalidad real: modelo, idioma, host y plataforma

TurboJev no promete que un archivo arbitrario pueda ejecutarse sin un runtime. Promete algo más sostenible: **el motor semántico nunca necesita saber qué arquitectura de modelo hay debajo**.

Un modelo de pesos abiertos queda integrado cuando un adapter puede:

1. renderizar `PromptDocument` al formato esperado por el checkpoint;
2. tokenizar con el tokenizer real de ese modelo;
3. descubrir códigos de salida estables;
4. producir scores exactos para esos candidatos;
5. devolverlos por `ScoringBackend`.

No habrá `match architecture { qwen => ..., llama => ... }` en el core. Arquitectura y formato son strings/metadata libres.

La misma regla aplica al idioma humano: el core preserva texto literalmente y no traduce. El `PromptRenderer` del modelo decide cómo expresar instrucciones auxiliares.

La misma regla aplica al lenguaje de programación: `turbojev-wire` será la frontera canónica y los bindings finos no duplicarán semántica.

La misma regla aplica a plataforma: los crates portables no pueden usar filesystem, networking, procesos, env vars, APIs browser, OS o GPU. `scripts/audit_portability.py` lo vigila y CI compilará el workspace portable también para `wasm32-unknown-unknown`.

---

## 6. Mejoras directas sobre SemIf

### 6.1 Candidate IDs dinámicos

No habrá tablas por modelo como `A = token 32`.

TurboJev valida:

```text
encode(symbol) = [token]
decode([token]) = symbol
encode(prompt + symbol) = encode(prompt) + [token]
```

Esto ya está iniciado en `turbojev-tokenizer`.

### 6.2 Softmax/calibration fuera del backend

SemIf puede implementar scoring de formas diferentes por backend. TurboJev fuerza un contrato único:

```text
backend -> raw logits -> calibrator -> typed answer
```

Así CUDA, Metal, browser y Candle tienen exactamente la misma semántica pública.

### 6.3 Capability negotiation

Cada backend declara si soporta:

- shared prefix
- parallel suffixes
- selective output head
- multi-token candidates
- local inference

El execution planner seleccionará el path más rápido que preserve la semántica.

### 6.4 Multi-token fallback

A/B/C será solo el fast path. Si el tokenizer no expone suficientes códigos estables, TurboJev podrá puntuar códigos multi-token mediante un trie exacto sin cambiar el schema público.

### 6.5 Prompt contract formal

Prompting deja de ser una cadena dispersa por el código. Existe un ID de contrato estable que puede quedar registrado en manifests y checkpoints.

---

## 7. Mejoras directas sobre Nimble

### 7.1 Candidate-only projection como capacidad, no como implementación fija

Nimble optimiza el head en MLX. TurboJev define `selective_output_head` en el backend SPI. Un backend que pueda obtener hidden states calculará solo las filas del output head requeridas; uno que no pueda seguirá usando full-vocab + gather.

### 7.2 Specialized model no rompe Generic mode

TurboJev tendrá dos modos bajo la misma API:

```text
Generic Model
  frozen Qwen/Gemma/etc.

Decision-specialized Model
  base + LoRA / merged checkpoint
```

Esto evita crear dos productos diferentes.

### 7.3 Training contract portable

Los modelos especializados publicarán:

- base model y revision
- tokenizer revision
- prompt contract
- candidate encoding policy
- max context
- training objective
- calibration profile

El runtime rechazará combinaciones incompatibles en vez de degradarse silenciosamente.

### 7.4 Calibration split independiente

Entrenar candidate logits y calibrar probabilidades son problemas distintos. TurboJev los tratará como dos etapas explícitas.

### 7.5 Menor tamaño como objetivo de investigación

Nimble demuestra el valor del specialization training con 9B. TurboJev investigará 0.6B–4B como decisores dedicados para routing, guardrails, tool choice y agent loops.

---

## 8. Pipeline de inferencia objetivo

```text
EvaluationRequest
      │
      ▼
validate schema
      │
      ▼
compile DecisionBatch
      │
      ▼
execution planner
      │
      ├─ independent
      ├─ cached serial
      └─ shared-prefix parallel
      │
      ▼
prompt contract + tokenizer codebook
      │
      ▼
model forward
      │
      ├─ selective candidate head, if supported
      └─ full vocab + gather, otherwise
      │
      ▼
raw candidate logits
      │
      ▼
calibration
      │
      ▼
typed response
```

---

## 9. Runtime adapter strategy

TurboJev 1.0 no tendrá un runtime obligatorio. Los adapters se construyen como paquetes separados y opcionales.

### Adapter custom — ya iniciado en v0.2

`turbojev-backend-custom` acepta una implementación de scoring suministrada por la aplicación. Esto permite integrar hoy cualquier runtime Rust/host que ya tenga el modelo cargado sin modificar TurboJev core.

### Adapter GGUF opcional

Un adapter para llama.cpp/GGUF sigue siendo prioritario por cobertura de modelos, pero será un módulo opt-in. Sus tipos jamás cruzarán al core. Puede usar aceleración distinta según cómo la aplicación lo compile, sin introducir `cfg(target_os)` en los crates portables.

Milestones:

1. runtime handle abstracto;
2. tokenizer adapter;
3. `PromptRenderer` por perfil/checkpoint;
4. exact single-token candidate scoring;
5. multi-token sequence scoring;
6. context limit validation;
7. shared-prefix reuse;
8. parallel suffix execution donde el runtime lo permita;
9. candidate-only output projection si el runtime expone hidden/head rows;
10. benchmark contra JSON generado.

### Adapter Rust-native opcional

Un runtime Rust-native puede usar Candle u otra implementación futura. El objetivo no es depender de Candle, sino demostrar que el mismo `ScoringBackend` puede funcionar sin C/C++.

### Adapter Web/WASM opcional

```text
@turbojev/web
    │
    ▼
Rust/WASM binding
    │
    ▼
optional browser runtime adapter
    │
    ├─ WebGPU engine
    ├─ wllama bridge if useful
    └─ future browser runtimes
```

`wasm-bindgen`, `web-sys`, IndexedDB y WebGPU vivirán exclusivamente en el binding/adapter web. El core portable no los importará.

---

---

## 11. Bindings

### Node / TypeScript
- napi-rs para native Node.
- API Promise-friendly aunque el scoring core sea síncrono.
- tipos TypeScript generados/centralizados.

### Browser
- wasm-bindgen.
- JS objects -> canonical JSON/schema -> Rust.

### Python
- PyO3/maturin.
- mismos nombres conceptuales: `classify`, `evaluate`, `load`.

### .NET
- C ABI estable + wrapper NuGet seguro.
- `SafeHandle` para ownership.

### Swift/Kotlin
- C ABI / UniFFI evaluado según constraints móviles.
- ningún business rule duplicado en wrappers.

---

## 12. C ABI

Handles opacos:

```c
typedef struct tj_engine tj_engine;
typedef struct tj_model tj_model;
typedef struct tj_result tj_result;
```

Entrada universal adicional:

```c
int tj_evaluate_json(
    tj_model* model,
    const char* request_json,
    char** response_json
);
```

Todo allocation/free crossing the boundary tendrá funciones explícitas y tests de ownership.

---

## 13. Training especializado

### Objetivo

No entrenar generación de JSON. Optimizar directamente:

```text
candidate_logits = logits[candidate_token_ids]
loss = cross_entropy(candidate_logits, target_index)
```

### Dataset

- choice
- boolean/noul
- ordered score
- policy application
- routing
- evidence judgment
- tool selection
- agent-control decisions

### Mejoras de datos

- contrastive pairs donde un solo hecho cambia el label
- candidate order permutation
- candidate code permutation
- multilingual variants
- negative/no-match options
- adversarially similar criteria
- long shared-state tasks

### Evaluation

Separar:

- accuracy/agreement
- NLL
- Brier score
- ECE/calibration
- latency
- peak memory
- decisions/sec
- robustness to option permutation
- robustness to paraphrase

---

## 14. Seguridad y robustez

- límites de tamaño antes de allocation grande
- sin truncado silencioso por defecto
- rechazo de NaN/Inf
- model manifests con hashes/revisions
- paths y downloads validados por backend
- no ejecutar código de repositorios de modelos (`trust_remote_code=false` conceptualmente)
- fuzzing de JSON/schema/tokenizer boundaries
- cancellation y resource caps en servidores
- no telemetry by default

---

## 15. CI/CD objetivo

Matrices:

- Windows x86_64
- Linux x86_64
- macOS arm64
- WASM/browser
- Android compile smoke
- iOS compile smoke

Gates:

```text
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
cargo audit
schema golden tests
ABI compatibility tests
```

Backends GPU tendrán jobs separados y no bloquearán tests puros del core.

---

## 16. Benchmark suite

Todo benchmark debe registrar:

- model + exact revision/hash
- quantization
- backend
- hardware
- prompt contract
- tokenizer
- decision count
- input tokens
- execution path
- candidate count
- raw latency distribution

Comparaciones mínimas:

1. candidate logits vs generated JSON
2. independent vs cached serial vs parallel
3. full vocabulary head vs selective head
4. generic base model vs specialized adapter
5. calibration before/after
6. browser vs native where model artifacts are equivalent

---

## 17. Roadmap

### Fase 0 — contrato y arquitectura
**Estado: COMPLETADA**

- workspace Rust
- schema canónico
- backend SPI
- calibration interface
- deterministic core
- fixture backend
- SPEC/architecture docs

### Fase 1 — universal portable core
**Estado: COMPLETADA EN v0.3**

Implementado:

- `turbojev-engine`: planner por capabilities
- `ExecutionPlan`: reuse/scheduling/projection
- `turbojev-model`: `ModelSource` opaco + `ModelSpec` sin allowlist de arquitecturas
- `turbojev-wire`: frontera JSON única para bindings
- `turbojev-prompt` v2: IR semántico sin system prompt inglés obligatorio
- `turbojev-tokenizer`: single-token + fixed-width multi-token fallback
- `turbojev-backend-custom`: integración de runtimes arbitrarios
- auditoría automática anti dependencias OS/browser
- CI portable + wasm32 check

Añadido en v0.3:

- `turbojev-compat`: perfiles `turbojev`, `typesafe-systemone-v1` y `ai-sdk-evaluation-v4`;
- `turbojev-conformance`: comparación reference-vs-optimized con tolerancias explícitas;
- golden fixtures para request/response;
- schemas JSON independientes para TypeSafe System One y AI SDK;
- `input_tokens` opcional propagado desde backend metrics;
- provenance explícita de `confidence`;
- ADRs que prohíben mezclar compatibilidad wire con inferencia.

Pendiente transversal para fases siguientes:

- property tests extensivos;
- fuzz targets;
- perfil OpenRouter Decisions cuando estabilice su contrato;
- model/profile manifest validation.

### Fase 2 — runtime bridge universal + primer runtime real opt-in
**Estado: COMPLETADA EN v0.6**

Implementado:

- bridge token-level universal (`PromptRenderer`, `PromptCodec`, `RuntimeExecutor`);
- adapter GGUF/llama.cpp **sin dependencia desde el core**;
- tokenizer boundary/codebook usando el tokenizer y chat template reales;
- prompt renderer profiles sin imponer idioma desde el core;
- reference path independent/serial/full-vocabulary;
- runtime conformance y benchmark protocol;
- shim C ABI propio para aislar el ABI de llama.cpp.

### Fase 3 — shared prefix + bounded parallel suffixes
**Estado: COMPLETADA EN v0.8**

Implementado:

- lifecycle explícito de prefill shared-prefix en `LlamaCppApi`;
- secuencia base inmutable + ramas efímeras aisladas por task;
- cached serial manteniendo full-vocabulary projection;
- bounded parallel suffix batches con orden de salida determinista;
- fallback correcto al reference path si el LCP tokenizado es vacío;
- suffix vacío mediante retención de logits del prefijo;
- parity exacta reference vs cached serial vs parallel en fixtures;
- stress nativo de aislamiento de cache con 2..7 branches;
- `ExecutionBudget` + `BackendResourceHints` para límites portables de memoria/cantidad;
- `PlanningPolicy::Reference` fuerza None + Serial + FullVocabulary.

Pendiente transversal de benchmark: ejecutar mediciones reales sobre GGUF cuando el entorno de release disponga de modelos/toolchain; esto no bloquea el contrato de Fase 3.

### Fase 3B — candidate-only projection (v0.9)
**Estado: COMPLETADA EN v0.9**

Implementado:

- `turbojev-projection` con candidate-row projection portable y full-vocabulary oracle;
- `turbojev-runtime-projection` como executor selectivo real, independiente de formato/modelo/SO;
- `selective_output_head` detrás de capability explícita;
- full-vocabulary v0.6–v0.8 preservado como baseline de corrección;
- `candidate_rows_projected` como contador auditable del trabajo selectivo;
- `verify_selective_projection` exige parity numérica + row accounting exacto;
- `LlamaCppApi` soporta rutas selected independent/shared/parallel sin obligar al FFI stock a fingir soporte;
- bounded-parallel + candidate-only cubierto por fixture parity del adapter llama.cpp;
- stock llama.cpp FFI mantiene `selective_output_head = false` mientras no exista una ruta estable que salte realmente el full vocabulary head.

Gate cerrado: los bindings de host pueden comenzar en Fase 4 sin modificar el contrato de inferencia.

### Fase 4 — Node/TypeScript
**Estado: COMPLETADA EN v0.10**

Implementado:

- `turbojev-node` con Node-API/napi-rs y frontera JSON fina;
- package `@turbojev/node`;
- `TurboJev.load/classify/evaluate/evaluateCompatible/close`;
- worker dedicado por sesión para no bloquear el event loop ni forzar `Send + Sync`;
- errores tipados `TurboJevError`;
- model resolver local / `file://` / HTTP(S) / `hf://`;
- cache/download manager exclusivamente en el binding Node;
- discovery de native addon por override o package de plataforma;
- tipos TS generados desde los JSON Schemas canónicos;
- tests de worker protocol, lifecycle, local model y download/cache.

### Fase 5 — Browser/WASM/WebGPU
**Estado: COMPLETADA EN v0.11/v0.12**

v0.11 — binding browser universal:

- `turbojev-wasm` sobre `TurboJevAsync` y `AsyncRuntimeBackend`;
- package `@turbojev/web` con `TurboJevWeb.load/classify/evaluate/evaluateCompatible/close`;
- worker dedicado para mantener runtime/modelo fuera del main thread;
- contrato browser low-level `encodePrompt/encodePromptWithAnswer/decodeTokens/scoreRuntimeBatch`;
- candidate-code discovery, boundary verification, planner, calibration y typed answers permanecen en Rust/WASM;
- model resolution `browser-cache` o `runtime` según quién sea dueño del cache/repositorio;
- CacheStorage consume un clone concurrente sin bloquear el stream entregado al runtime;
- tipos TS generados desde los schemas canónicos;
- browser/WASM CI separado de los crates portables.

v0.12 — primer runtime browser real:

- `@turbojev/runtime-transformersjs` como adapter opcional;
- checkpoint/chat-template/tokenizer reales mediante Transformers.js;
- inferencia causal local con device/dtype delegados al runtime (por ejemplo WebGPU);
- exact assistant-boundary candidate discovery, sin IDs hardcodeados por familia de modelo;
- ruta reference serial full-vocabulary como baseline de corrección;
- `num_logits_to_keep=1` usado solo como optimización de posición cuando el modelo lo soporta;
- capacidades no implementadas permanecen explícitamente falsas: shared-prefix, parallel suffixes, sequence scoring y candidate-only LM head;
- tests unitarios con runtime Transformers.js inyectado sin requerir descarga de modelos en CI.

Gate cerrado: TurboJev ya dispone de un host browser/WASM y de un runtime browser concreto sin contaminar el core con WebGPU, ONNX, Transformers.js ni APIs del navegador.

### Fase 6 — Python / .NET
**Estado: COMPLETADA EN v0.13**

Implementado:

- `turbojev-host-native` como única capa compartida de sesión/modelo nativo para bindings;
- refactor de `turbojev-node` para reutilizar esa capa en vez de duplicar configuración;
- `turbojev-capi` con handle opaco, UTF-8 JSON, ownership explícito de strings, panic containment y errores thread-local;
- header C estable `bindings/c/include/turbojev.h`;
- `turbojev-python` con PyO3/ABI3 y package Python fino;
- wrapper Python `TurboJev.load/evaluate/classify/evaluate_compatible/close` sin semántica duplicada;
- wrapper .NET sobre P/Invoke del C ABI;
- dispatcher .NET dedicado para mantener thread affinity aunque los `Task` cambien de thread;
- gates Python/C/.NET y jobs CI separados;
- ADR 0015: toda integración native-host comparte `turbojev-host-native`.

Gate cerrado: Node, Python, C y .NET convergen en la misma implementación Rust de sesión nativa y el mismo wire contract.

### Fase 7 — Mobile
**Estado: COMPLETADA EN v0.14**

- `turbojev-jni` + Kotlin/Android wrapper con `HandlerThread`;
- Swift package sobre el C ABI con `Thread` dedicado (no solo serial GCD);
- runtime/model selection sigue en la capa host;
- cero branches `target_os` en core;
- mismo JSON/wire contract que desktop;
- gates estructurales de thread affinity y headers/bindings.

### Fase 8 — multi-token candidate scoring avanzado
**Estado: COMPLETADA EN v0.15**

Implementado:

- `turbojev-sequence` con trie prefix-free portable y reutilización de candidate prefixes;
- log-likelihood condicional exacta por token usando normalización de vocabulario completo;
- single-token conservado como fast path preferido;
- fallback dinámico a codebooks multi-token sin IDs hardcodeados por modelo;
- métricas de candidate-token work, prefix evaluations y token steps ahorrados;
- `BackendMetrics.effective_plan` para reportar el plan realmente ejecutado;
- downgrade correcto de multi-token a serial + full-vocabulary cuando parallel/candidate-only no son matemáticamente seguros;
- llama.cpp shim ABI v4 con scoring exacto independiente y shared-prefix;
- parity nativa contra cálculo manual de secuencia y propiedad de cancelación del normalizador para single-token;
- oracle reusable en `turbojev-runtime-conformance` para validar adapters externos contra la misma matemática exacta;
- candidate-only multi-token bloqueado hasta disponer de `logZ` exacto por continuación.

Gate cerrado: el fallback multi-token es exacto, prefix-sharing y model/tokenizer agnostic; no convierte una aproximación de logits en falsa probabilidad.

### Fase 9 — calibration tooling
**Estado: COMPLETADA EN v0.16**

Implementado:

- `CalibrationDataset` versionado para logits + referencia + sample weights sobre holdout independiente;
- fitting determinista de temperatura positiva minimizando NLL en espacio `log(T)`;
- NLL, multiclass Brier, top-label ECE, accuracy y bins de reliability diagram;
- `CalibrationProfile` versionado con provenance de model/tokenizer/prompt/renderer;
- `TemperatureCalibrator::from_profile` para distinguir profiles fitted de softmax condicional sin calibrar;
- CLI `turbojev-calibrate fit/report` fuera del portable core;
- JSON schemas para dataset/profile, fixture numérico reproducible, ADR 0018 y release gate offline.

Gate cerrado: cambiar la temperatura no puede cambiar el argmax; calibration queda auditable, reproducible y separada de inference.

### Fase 10 — TurboJev specialized models
**Estado: EN PROGRESO — v0.20 evaluation/interchange tooling completado**

Implementado en v0.17:

- crate portable `turbojev-specialized`;
- contratos versionados de training recipe, specialized-model manifest y evaluation plan;
- objective explícito `candidate-logit-cross-entropy`;
- recetas framework-neutral LoRA / QLoRA;
- matriz inicial 0.6B / 0.8B / 2B / 4B;
- candidate-order + candidate-code debiasing con seed reproducible;
- política multilingual + contrastive + negative/no-match + adversarial + long shared-state;
- split ids train/validation/calibration/benchmark obligatoriamente distintos;
- `ModelSpec.decision_contract` opcional: Generic Mode permanece sin cambios;
- manifest compatibility entre recipe, tokenizer, prompt IR, renderer, candidate encoding, context y calibration;
- generic-vs-specialized same-benchmark evaluation contract con accuracy/NLL/Brier/ECE/latency/memory/throughput/robustness;
- JSON schemas, fixtures, ADR 0019 y gate offline de Fase 10.

Implementado en v0.18:

- crate portable `turbojev-training-data`;
- contrato `turbojev.semantic-training-dataset.v1` para requests tipadas + ground truth;
- contrato `turbojev.materialized-training-dataset.v1` para registros candidate-logit;
- reutilización del compilador canónico `EvaluationRequest -> DecisionBatch` entre inference y training;
- targets tipados `choice` / `noul` / `score` con validación contra el schema real;
- materializador determinista de candidate-order + candidate-code assignments;
- permutations únicas con identidad incluida y cap automático por `n!`;
- `CandidateCodeProvider` model/tokenizer-aware: el core portable no inventa token ids;
- conservación exacta del peso de cada decisión a través de augmentation;
- validación de multilingual coverage, contrastive pairs, no-match, adversarial criteria y long shared-state;
- schemas, fixtures multilingües, docs y ADR 0020.

Implementado en v0.19:

- crate portable `turbojev-specialized-artifact`;
- `BoundaryVerifiedCodeProvider` sobre el mismo `PromptRenderer` + `PromptCodec` usados por inference;
- discovery determinista de códigos en el assistant boundary real del checkpoint;
- candidate-logit CE v1 exige codebooks single-token exactos y falla si el modelo no los expone;
- contrato `turbojev.model-bound-training-dataset.v1` con rendered messages, prompt token ids, candidate token ids y target token id;
- context-limit + recipe/materialized/model provenance validation;
- helper de emisión de `SpecializedModelManifest` después de entrenamiento externo;
- calibration handoff estricto por holdout/model/tokenizer/prompt/renderer provenance;
- runtime compatibility gate para specialized checkpoints antes de inference;
- schema, fixtures, docs y ADR 0021.

Implementado en v0.20:

- crate portable `turbojev-specialized-eval`;
- contrato `turbojev.specialization-run.v1` para resultados externos framework/runtime-neutral;
- provenance obligatoria de backend, host, quantization, execution path, prompt/tokenizer y calibration;
- same-benchmark/same-observation gate entre generic y specialized;
- quality metrics baseline-only reutilizando `turbojev-calibration`;
- candidate-order y paraphrase robustness como invariance + accuracy semánticas;
- latency summary, decisions/sec, peak memory y deltas explícitos `specialized - generic`;
- contrato `turbojev.specialization-report.v1`;
- contrato `turbojev.specialization-bundle.v1` con SHA-256 de evidencia reproducible;
- host CLI separado `turbojev-specialized-report`;
- schemas, fixtures, docs, ADR 0022 y gate offline dedicado.

Límite arquitectónico confirmado en v0.19 y preservado en v0.20:

- TurboJev **no** incorpora un trainer LoRA/QLoRA, backpropagation, optimizer ni framework de entrenamiento;
- PyTorch/TRL/Axolotl/MLX/CUDA/etc. pueden consumir el export model-bound externamente, pero no son dependencias de TurboJev;
- generic mode permanece completamente independiente de los contratos specialized.

Pendiente para cerrar Fase 10 empíricamente:

- checkpoints experimentales externos 0.6B/0.8B/2B/4B producidos fuera de TurboJev;
- calibration profiles obtenidos de holdout real;
- runs reales emitidos con `turbojev.specialization-run.v1`;
- publicación del bundle/report generic-vs-specialized reproducible.

El tooling requerido por TurboJev para esa evidencia queda completado en v0.20; la optimización de pesos permanece externa.

### Fase 11 — server compatibility
**Estado: EN PROGRESO — v0.25 llama.cpp multi-root runtime path implementado; activación productiva gated por conformance real**

Implementado en v0.21:

- `turbojev-server-protocol` portable para routing, perfiles, batches y errores;
- `turbojev-server` host-only con TCP/HTTP local, config file, timeouts y threads;
- endpoints canonical, TypeSafe/System One y AI SDK/Vercel;
- alias OpenRouter gateway-friendly sin afirmar `/chat/completions`;
- una sola `NativeSession` thread-affine detrás de cola bounded;
- concurrencia HTTP bounded con backpressure 503;
- bind loopback-only;
- schemas, fixtures, docs, ADR 0023 y gate offline.

Añadido en v0.22:

- `turbojev-server-execution` portable con capability negotiation cross-request;
- `CrossRequestExecutor` SPI para serial reference y futuros fused adapters;
- `fusedBatch`, `cooperativeCancellation`, `hardDeadlines` y `maxFusedItems` como capacidades explícitas;
- planner de turnos bounded por fairness/backend/host limits;
- helper de conformance serial-vs-fused con parity exacta de `EvaluationResponse`;
- scheduler host round-robin para que un batch grande no monopolice la sesión;
- `X-TurboJev-Timeout-Ms` con clamp al timeout máximo del servidor;
- cancelación soft de trabajo todavía no iniciado después de timeout;
- métricas locales de overload/scheduler en `GET /v1/server/metrics`;
- metadata batch `serial-round-robin` o `fused-batch` según ejecución real;
- stock `NativeSession` continúa anunciando `fusedBatch=false`, `cooperativeCancellation=false` y `hardDeadlines=false`;
- schemas/fixtures nuevos, ADR 0024 y gates de server execution.

Añadido en v0.23:

- `NativeExecutionCapabilities` en `turbojev-host-native` como fuente de verdad de capacidades cross-request del runtime cargado;
- `NativeSession::evaluate_fused` como entry point real para futuros runtimes fused;
- el server deja de hardcodear capacidades y las mapea desde la sesión nativa;
- `GET /readyz` separado de `GET /healthz`; lifecycle `ready -> draining -> stopped`;
- `ShutdownToken` para graceful drain programático con `drainTimeoutMs`;
- durante draining, health permanece 200, readiness e inference nueva pasan a 503 mientras el trabajo ya admitido puede terminar;
- request IDs estructurados y sanitizados vía `X-TurboJev-Request-Id` / `X-Request-Id`;
- deployment mode `reverse-proxy` manteniendo bind loopback-only y enforcement opcional de `X-Forwarded-Proto: https`;
- harness externo stdlib-only con contratos `turbojev.server-load-plan.v1` / `turbojev.server-load-report.v1`;
- schemas/fixtures de native capabilities, lifecycle, load reports, reverse-proxy config y ADR 0025.

Añadido en v0.24:

- `FusedScoringBackend` + `FusedBackendCapabilities` como SPI portable separado de `ScoringBackend`;
- `TurboJevEngine::evaluate_fused` para múltiples requests independientes sin duplicar validación, planning, calibration ni typed-answer construction;
- `FusedRuntimeExecutor` como hook multi-root explícito para runtimes capaces de una operación batch real;
- `verify_fused_runtime_executor` en runtime conformance para referencia independiente vs fused;
- fixture backend habilitado como ruta fused end-to-end de conformance, con `NativeSession` y server reportando esa capability de forma real;
- llama.cpp stock conserva `fusedBatch=false`: shared-prefix/parallel-suffix dentro de un request no se reetiqueta como cross-request fusion;
- `estimatedFusedItemBytes` y `fusedMemoryBudgetBytes` para limitar batches por memoria además de fairness/host/runtime item caps;
- benchmark provenance bundle `turbojev.server-benchmark-bundle.v1` con SHA-256 de config/load-plan/load-report;
- ADR 0026 y gates/fixtures actualizados.

Añadido en v0.25:

- shim llama.cpp ABI v5 con `tj_llama_score_root_batch` para raíces de prompt independientes en sequence IDs distintos dentro de un único `llama_decode`;
- límites nativos de raíces y tokens totales derivados del contexto cargado;
- `LlamaCppExecutor` implementa `FusedRuntimeExecutor` para shapes single-token elegibles y recompone `RuntimeOutput` por request preservando orden;
- `TurboJevEngine::can_evaluate_fused` / `NativeSession::can_evaluate_fused` hacen eligibility por grupo antes de ejecutar el path fused;
- corrección del acceso batched a `llama_get_logits_ith`: los outputs se consultan usando el índice del token solicitado dentro del batch, tanto para multi-root como para parallel suffixes;
- stub llama.cpp endurecido con la misma semántica de output-index que upstream;
- native smoke de parity independent-vs-multi-root y 48 rondas de aislamiento sobre 2..8 raíces;
- FFI multi-root compilado pero `execution.llamacppMultiRootFused=false` por defecto; activarlo requiere conformance en la combinación exacta modelo/revisión llama.cpp/backend/hardware;
- candidatos multi-token, projection incompatible o límites excedidos degradan a serial cross-request sin degradar la semántica;
- hard/cooperative cancellation de llama.cpp permanece desactivada; no se extrapola el abort callback CPU a backends acelerados;
- contrato/schema/fixture `turbojev.llamacpp-multi-root.v1`, ADR 0027 y gates/docs actualizados.

Pendiente de Fase 11:

- ejecutar y publicar conformance multi-root sobre GGUF real en las combinaciones de backend/hardware que se vayan a declarar soportadas y, solo entonces, decidir qué perfiles pueden habilitar llama.cpp fused por defecto;
- definir, solo si preserva semántica, un adapter OpenAI/OpenRouter wire real en vez de fingir generated chat text;
- hard cancellation/deadline únicamente para runtimes que puedan interrumpir una ejecución ya iniciada de forma segura;
- streaming no aplica al fast path de decisión salvo futura metadata/telemetry explícita.

---

## 18. Definition of Done 1.0

TurboJev 1.0 no se publica hasta que:

- existe al menos un backend GGUF nativo estable;
- existe browser/WASM usable;
- choice/noul/score pasan conformance tests;
- shared state reuse está implementado al menos en un backend;
- token IDs nunca están hardcodeados por modelo en generic mode;
- probabilities tienen metadata de calibration clara;
- Node, Python y .NET comparten exactamente el mismo core;
- Android/iOS tienen smoke tests;
- benchmark bundle reproducible está publicado;
- no hay generated text en direct fast path;
- malformed responses/backends no pueden producir un typed answer inválido;
- API pública tiene semver y documentación de compatibilidad.

---

## 19. Entregas de Fase 9/10

Workspace actual añade:

```text
crates/turbojev-calibration
crates/turbojev-calibrate
schemas/turbojev.calibration-dataset.schema.json
schemas/turbojev.calibration-profile.schema.json
tests/calibration/holdout-overconfident.json
tests/calibration/profile-overconfident.json
docs/CALIBRATION.md
docs/adr/0018-versioned-temperature-calibration.md
```

Invariantes v0.16:

- fitting solo usa holdout explícito; nunca inference/training implícito;
- temperatura siempre es finita y estrictamente positiva;
- objetivo de fitting es weighted multiclass NLL;
- Brier/ECE/accuracy se reportan junto con NLL y sample count;
- `SoftmaxCalibrator` continúa `fitted=false`;
- `TemperatureCalibrator::from_profile` produce un calibrator `fitted=true`;
- model/tokenizer/prompt/renderer provenance viaja en el profile cuando se conoce;
- ningún filesystem, CLI o variable de entorno entra en `turbojev-calibration`;
- `turbojev-calibrate` es una herramienta host separada y no forma parte del WASM portable core;
- calibration no altera el orden de candidatos porque divide todos los scores por el mismo `T > 0`.

Fase 9 queda cerrada en v0.16. Fase 10 comenzó en v0.17 por contracts/recipes reproducibles antes de entrenar checkpoints.


## Milestone v0.4.0 — Universal Runtime Bridge — implemented

- Portable `turbojev-runtime` crate.
- Model-facing `PromptCodec` abstraction for exact chat-template/tokenizer boundaries.
- Runtime-facing `RuntimeExecutor` abstraction for token-level candidate scoring.
- `RuntimeBackend` composes renderer + codec + executor into the stable semantic backend SPI.
- Dynamic candidate symbol sets; no model-family token IDs.
- Single-token fast path; exact multi-token sequence path when advertised.
- Exact common-token-prefix factoring for shared-state reuse.
- No OS, browser, device, model-family, weight-format or runtime dependency in portable crates.
- No mandatory cross-thread model: thread-affine browser/mobile runtimes are first-class; server hosts can opt into `ThreadSafeScoringBackend`.

## Milestone v0.5.0 — Async + Runtime Conformance — implemented

- Added executor-neutral `AsyncScoringBackend` and `AsyncRuntimeExecutor` contracts.
- Added `TurboJevAsyncEngine`/`TurboJevAsync` with the same schema, planner, calibration and typed outputs as the sync engine.
- No Tokio, async-std, browser API or OS scheduler dependency in portable crates.
- Async futures deliberately do not require `Send`; thread-affine WebGPU/mobile/JS runtimes remain valid.
- Added `AsyncCustomBackend` for host-provided futures/promises.
- Added `turbojev-runtime-conformance` as the common acceptance kit for concrete adapters.
- Conformance now covers exact answer-boundary tokenization, candidate round-trip/uniqueness/prefix-free rules, capability-plan truthfulness, finite output shapes, and reference-vs-optimized logit parity.
- Added ADR 0007/0008 and dedicated async/runtime-conformance documentation.

## Milestone v0.6.0 — Concrete reference runtime + benchmark protocol — implemented

- Added optional `turbojev-runtime-llamacpp`; no portable core crate depends on it.
- Added a narrow `LlamaCppApi` boundary and an opt-in native FFI implementation behind a TurboJev-owned shim ABI.
- Reference execution is deliberately independent + serial + full-vocabulary gather.
- v0.6 accepts verified single-token candidate codes only and emits zero generated answer tokens.
- Model chat-template/tokenizer candidate boundaries are checked exactly through the runtime.
- Prompt control language is provided by `ReferencePromptConfig`, not by portable core code.
- Added conformance fixtures representing two different tokenizer/chat-template families.
- Added `turbojev-benchmark` and a multilingual model-neutral reference suite.
- Added native C++ shim syntax checking and benchmark fixture verification.
- Shared-prefix reuse, parallel suffixes, selective output-head projection and multi-token llama.cpp scoring remain intentionally disabled until a later milestone proves parity against this path.

## Milestone v0.7.0 — Deterministic shared-prefix reuse — implemented

- Added explicit reusable-prefix lifecycle methods to `LlamaCppApi`.
- Bumped the TurboJev llama.cpp shim ABI to v2 with prefix-prefill and suffix-scoring calls.
- llama.cpp keeps sequence 0 as the immutable prefix and clones it into sequence 1 for each serial suffix.
- Added retained prefix logits so identical/full-prefix tasks with an empty suffix remain valid.
- `LlamaCppExecutor` dynamically advertises shared-prefix support and preserves the independent reference path.
- Added exact reference-vs-shared logit parity tests and semantic typed-answer parity tests.
- Strengthened the native ABI smoke stub so logits depend on emulated sequence state; cache mistakes can no longer pass through constant logits.
- Native shim builds with `-Wall -Wextra -Wpedantic -Werror` in the offline smoke gate.
- Parallel suffixes are enabled in v0.8; selective-head projection remains disabled until v0.9.

## Milestone v0.8.0 — Bounded parallel suffix execution — implemented

- Added bounded parallel suffix scheduling while preserving the v0.6 reference and v0.7 cached-serial baselines.
- Added portable caller `ExecutionBudget` and backend `BackendResourceHints`.
- Added `parallel_task_limit` to the execution plan and native execution metadata.
- Bumped the llama.cpp shim ABI to v3 with batched branch scoring.
- Sequence 0 stays immutable; branch sequence IDs are copied, scored together, then removed.
- Added mixed empty/non-empty branch handling and deterministic result ordering.
- Added native 48-round cache-isolation stress testing over varying branch counts.
- Added exact reference/shared/parallel logit parity fixtures.
- Candidate-only output projection remains disabled and is the sole optimization focus of v0.9.

## Milestone v0.9.0 — Candidate-only output projection — implemented

- Added portable dense output-head projection math and row gathering/reference utilities.
- Added a runtime-neutral selective executor over final hidden states + requested output rows.
- Added exact candidate-row accounting to backend metrics.
- Added selective projection conformance with numerical parity and truthful-work verification.
- Extended llama.cpp adapter contracts with independent/shared/parallel selected paths.
- Stock llama.cpp FFI remains full-vocabulary and does not advertise a false capability.
- Added ADR 0011 and dedicated candidate-projection documentation.

### Milestone v0.10.0 — Node/TypeScript — implemented

- Added the `turbojev-node` Node-API crate.
- Added `@turbojev/node` with a dedicated worker-thread architecture.
- Added binding-owned local/remote/Hugging Face model resolution and cache.
- Added generated TypeScript schema declarations and typed public errors.
- Added native addon discovery and a stable JSON-based addon protocol.
- Added Node lifecycle/download/cache tests that run without a real model through a fake native addon fixture.

### Milestone v0.11.0 — Browser/WASM host binding — implemented

- Added `turbojev-wasm` on the canonical asynchronous runtime bridge.
- Added `@turbojev/web`, worker lifecycle, browser model cache/resolution and typed host API.
- Added a low-level model-facing browser runtime contract; JavaScript does not own TurboJev semantics.
- Dynamic candidate-code discovery and exact tokenizer boundary checks remain in Rust/WASM.
- Added runtime-owned opaque model references and non-blocking CacheStorage cloning.
- Added Web binding tests/build/package gates and dedicated WASM CI.

### Milestone v0.12.0 — Concrete Transformers.js browser runtime — implemented

- Added `@turbojev/runtime-transformersjs` as an optional browser runtime adapter.
- Uses checkpoint chat templates/tokenization and causal-LM forward logits.
- Supports Transformers.js device/dtype configuration without exposing those concepts to TurboJev Core.
- Keeps conservative reference capabilities truthful: serial, full-vocabulary, single-token direct scoring.
- Uses exact boundary discovery instead of hard-coded Qwen/Llama/Gemma token IDs.
- Added adapter unit tests, TypeScript build/package gates, documentation and ADR 0014.

### Milestone v0.13.0 — C ABI + Python + .NET — implemented

- Added shared `turbojev-host-native` and refactored Node onto it.
- Added stable opaque-handle C ABI and C header.
- Added PyO3 ABI3 native extension plus Pythonic thin wrapper.
- Added .NET P/Invoke package with dedicated thread-affinity dispatcher.
- Added C header, Python wrapper and .NET structural/build release gates.

### Milestone v0.14.0 — Mobile host bindings — implemented

- Added Android JNI + Kotlin wrapper with a dedicated `HandlerThread`.
- Added Swift package over the stable C ABI with a dedicated Foundation `Thread`.
- Reused `turbojev-host-native`; no semantic decision logic is duplicated on mobile.
- Added mobile binding structural gates and ADR 0016.

### Milestone v0.15.0 — Exact multi-token candidate scoring — implemented

- Release artifacts are now generated from a deterministic source manifest, with transient build/cache directories excluded.
- Release closure now also compiles the Swift package surface, runs Python wrapper tests directly, validates the C11 header, and builds/tests/packs all TypeScript host packages.
- Host-binding CI is split into dedicated jobs instead of relying only on Rust workspace checks.
- Added portable `turbojev-sequence` candidate trie and exact sequence log-likelihood math.
- Preserved single-token scoring as the preferred zero-generation fast path.
- Added dynamic prefix-free multi-token fallback with no model-family token IDs.
- Added sequence execution work metrics and effective-plan reporting.
- Multi-token execution conservatively disables unsafe task parallelism and candidate-only projection while retaining valid shared prompt-prefix reuse.
- Bumped the optional llama.cpp shim to ABI v4 with exact independent/shared sequence scoring.
- Added native exact-log-likelihood and single-token parity smoke coverage.
- Added ADR 0017 and `docs/MULTI_TOKEN_SCORING.md`.

### Next milestone — Phase 9 calibration tooling

Implement temperature fitting on an independent holdout set plus ECE, Brier and NLL reporting and versioned calibration-profile manifests.


### Milestone v0.16.0 — Calibration tooling — implemented

- Added versioned calibration datasets and profiles.
- Added deterministic scalar-temperature fitting by weighted NLL.
- Added NLL, multiclass Brier, top-label ECE, reliability bins and weighted accuracy.
- Added profile provenance for model/tokenizer/prompt/renderer compatibility.
- Added `turbojev-calibrate` fit/report CLI without contaminating portable core.
- Added JSON schemas, numeric fixtures, ADR 0018 and CI/offline calibration gates.
- Plain softmax remains explicitly unfitted; profile-backed temperature scaling is explicitly fitted.


### Patch v0.28.2 — Final .NET lifecycle + release-run correctness — implemented
- `.NET DisposeAsync` always tears down the affinity worker in `finally`, including native-close failure paths.

- The cross-language matrix runs the already-built .NET Release artifact (`-c Release --no-build`) instead of rebuilding Debug during the final parity gate.
- The .NET binding now shuts down and joins its dedicated native-affinity thread if `LoadAsync` fails before returning an engine.
- Structural release gates assert the .NET affinity/lifecycle cleanup and Release/no-build E2E invocation.
- No new architecture milestone is introduced; the only remaining 1.0-readiness condition is the real host run with Rust 1.82, .NET 8 and the pinned GGUF.

## Verification policy

Every milestone must preserve two separate truths: offline structural validation is useful, but it is never described as compilation. Release candidates require the pinned Rust toolchain and must pass formatting, workspace/all-target checks, Clippy with warnings denied, tests, WASM compilation, and the three-desktop-OS CI matrix. See `docs/TESTING.md`.


### Milestone v0.17.0 — Specialized model contracts + training recipes — implemented

- Added portable `turbojev-specialized` with no trainer/runtime/OS dependency.
- Added versioned LoRA/QLoRA recipes using direct candidate-logit cross-entropy.
- Added deterministic candidate-order and candidate-code permutation policy.
- Added multilingual/contrastive/no-match/adversarial/long-state data requirements.
- Added pairwise-distinct train/validation/calibration/benchmark split enforcement.
- Added specialized checkpoint manifests and recipe compatibility validation.
- Attached optional `DecisionModelContract` to `ModelSpec`; generic models remain contract-free.
- Added 0.6B/0.8B/2B/4B experiment fixtures and a same-benchmark generic-vs-specialized evaluation plan.
- Added schemas, offline fixture gate, documentation and ADR 0019.

### Milestone v0.18.0 — Semantic training dataset + deterministic materialization — implemented

- Added portable `turbojev-training-data`.
- Added semantic source datasets using canonical `EvaluationRequest` plus typed targets.
- Exposed `turbojev-core::compile_decision_batch` as the single inference/training candidate-normalization path.
- Added deterministic unique candidate-order and code-assignment debiasing with stable seeded permutations.
- Candidate-code supply is an explicit host/runtime boundary; portable code never guesses model token ids.
- Added candidate-logit records containing `PromptDocument`, target key/index/code and augmentation provenance.
- Augmented rows conserve source-decision sample weight.
- Added recipe data-policy validation, schemas, multilingual fixtures, documentation and ADR 0020.

The framework-neutral host/export and evaluation tooling described here was completed across v0.19-v0.20. Actual LoRA/QLoRA optimization remains external to TurboJev.


### Milestone v0.19.0 — Model-bound specialized artifact pipeline — implemented

- Added portable `turbojev-specialized-artifact`; no trainer, optimizer, filesystem, network, GPU or OS dependency.
- Added `BoundaryVerifiedCodeProvider` that uses the exact checkpoint renderer/chat-template/tokenizer boundary.
- Added deterministic single-token candidate-code discovery for candidate-logit CE v1.
- Added `turbojev.model-bound-training-dataset.v1` with rendered messages, exact prompt token ids, candidate token ids and target token id.
- Added context-limit and provenance validation across recipe/materialized/model identities.
- Added specialized manifest construction after external training.
- Added strict calibration-profile handoff and runtime specialized-contract compatibility gates.
- Preserved exact multi-token inference while refusing to reinterpret it as one-step candidate-logit CE training.
- Added schema, fixtures, documentation and ADR 0021.

Next Phase 10 evidence block: run external open-weight specialization experiments and feed their results back through the v0.20 contracts; external trainers remain outside TurboJev.


### Milestone v0.20.0 — Specialized evaluation + external-run interchange — implemented

- Added portable `turbojev-specialized-eval` with no filesystem, model loading, training framework, GPU, browser or OS dependency.
- Added `turbojev.specialization-run.v1` so any external harness can submit raw candidate logits plus semantic/provenance/performance data.
- Added strict fair-pair gates for benchmark ids, observation semantics, prompt/tokenizer contracts, backend, host, quantization and execution path.
- Reused `turbojev-calibration` for baseline accuracy/NLL/Brier/ECE instead of duplicating probability math.
- Added candidate-order and paraphrase robustness as semantic invariance + controlled accuracy.
- Added canonical `turbojev.specialization-report.v1` with timing, throughput, memory and explicit specialized-minus-generic deltas.
- Added `turbojev.specialization-bundle.v1` with SHA-256 artifact references for reproducible experiment evidence.
- Added host-only `turbojev-specialized-report` CLI, schemas, smoke fixtures, documentation, ADR 0022 and CI/offline verification.
- Actual LoRA/QLoRA optimization remains external to TurboJev.

### Milestone v0.21.0 — Local server protocol + bounded HTTP host — implemented

- Added portable `turbojev-server-protocol` for route/profile/batch/error semantics.
- Added host-only `turbojev-server` with loopback TCP/HTTP, config files, timeouts and bounded connection admission.
- Added canonical, TypeSafe/System One and AI SDK/Vercel evaluation endpoints plus documented gateway aliases.
- Added one dedicated native-session worker behind a bounded queue; thread affinity is preserved and the model is not cloned per connection.
- Added bounded ordered batch envelopes while explicitly reporting cross-request execution as serial in v0.21.
- Added typed native-session evaluation, schemas, fixtures, `docs/SERVER.md`, ADR 0023 and CI/offline checks.

### Milestone v0.22.0 — Cross-request execution SPI + fair scheduler — implemented

- Added portable `turbojev-server-execution` with truthful cross-request capability negotiation.
- Added serial/fused planning bounded by fairness quantum, host limits and backend-advertised fused capacity.
- Added `CrossRequestExecutor` plus exact serial-reference-vs-fused response parity conformance helper.
- Reworked the host worker from whole-batch commands into round-robin jobs while preserving per-batch response order.
- Added caller deadlines through `X-TurboJev-Timeout-Ms`, clamped to server limits, plus soft cancellation of not-yet-started work.
- Added scheduler/overload counters at `GET /v1/server/metrics`.
- Batch metadata now reports `serial-round-robin` or `fused-batch` from the execution path actually used.
- Stock `NativeSession` remains conservatively serial and does not claim cooperative in-flight cancellation or hard deadlines.
- Added execution/metrics schemas, fixtures, ADR 0024 and portable/WASM verification wiring.


### Milestone v0.23.0 — Native runtime bridge + deployment lifecycle — implemented

- Moved cross-request capability truth into `turbojev-host-native` through `NativeExecutionCapabilities`.
- Added the concrete `NativeSession::evaluate_fused` bridge while keeping stock fixture/llama.cpp runtimes explicitly unsupported rather than emulating fusion.
- Added `/readyz`, lifecycle state, `ShutdownToken`, bounded graceful drain and `SERVER_DRAINING` admission semantics.
- Added safe request correlation headers with generated fallback IDs and header-injection validation.
- Added loopback-only reverse-proxy deployment mode with optional forwarded-HTTPS enforcement; TLS/auth/public exposure remain external.
- Added versioned server load-plan/report contracts and a stdlib-only load harness that refuses remote targets by default.
- Added schemas, fixtures, docs, ADR 0025 and release verification wiring.
- Cross-request fused execution remains disabled for stock runtimes until a concrete backend demonstrates real fused execution and exact parity.

### Milestone v0.24.0 — Fused backend/runtime SPI + server conformance — implemented

- Added `FusedScoringBackend` / `FusedBackendCapabilities` as a distinct portable cross-request backend operation.
- Added `TurboJevEngine::evaluate_fused` while reusing normal request validation, planning, calibration and typed response construction.
- Added `FusedRuntimeExecutor` so concrete model adapters can expose one true multi-root runtime operation without abusing intra-request suffix batching.
- Added reusable fused-runtime parity conformance against independent reference scoring.
- Enabled the deterministic fixture backend as an end-to-end fused conformance path through host-native and the HTTP scheduler; stock llama.cpp remains serial across independent roots.
- Added memory-aware fused planning with backend `estimatedFusedItemBytes` and host `fusedMemoryBudgetBytes`.
- Added deterministic server benchmark provenance bundles with SHA-256 artifact references.
- Added ADR 0026, schemas/fixtures/docs and release verification updates.

### Milestone v0.25.0 — llama.cpp multi-root cross-request path — implemented, conservative activation

- Bumped the TurboJev-owned llama.cpp shim to ABI v5 and added a true independent-root batch call over distinct sequence IDs.
- Added llama.cpp `FusedRuntimeExecutor` support for eligible single-token decision groups and dynamic root/token capacity checks.
- Wired per-shape fused eligibility through core, native host and server scheduler so unsupported groups fall back to serial execution.
- Corrected llama.cpp batched output lookup to current batch-token-index semantics and made the stateful native stub enforce the same rule.
- Added independent-vs-fused native parity plus 48-round root/cache isolation stress coverage.
- Kept `execution.llamacppMultiRootFused=false` by default; the concrete deployment must supply real model/backend/hardware conformance evidence before advertising the capability.
- Added `turbojev.llamacpp-multi-root.v1`, ADR 0027, documentation and release gates.
- Cooperative/hard cancellation remains unclaimed for llama.cpp.


### Milestone v0.26.0 — Evidence-bound runtime qualification — implemented

- Added portable `turbojev-runtime-qualification` with a versioned exact-target qualification manifest.
- Bound fused capability evidence to model SHA-256, llama.cpp revision, shim ABI, backend, hardware and driver identity.
- `llamacppMultiRootFused=true` now requires both a passing qualification manifest and an exact current fingerprint; boolean opt-in alone is rejected.
- Added deterministic host tooling to create fingerprints/manifests from real conformance observations.
- Kept cooperative/hard cancellation false for the generic llama.cpp path; CPU-only upstream abort callback behavior is not generalized across accelerator backends.
- Added schema/fixture/docs/ADR 0028 and release gates.

### Milestone v0.27.0 — Cross-language real-model release harness — implemented

- Added a single canonical real-model request shared by Node/TypeScript, Python, C and .NET/C# runners.
- Added `scripts/e2e_language_matrix.py` to execute the same GGUF artifact through all required native host bindings.
- Added model/request SHA-256 provenance plus per-language status/duration/stdout hashes in `turbojev.e2e-language-report.v1`.
- Added C11, Python, Node and .NET runners that assert a valid typed `route.choice` response rather than model-specific accuracy.
- Kept Swift as an optional execution row because true Swift runtime testing requires an Apple host; its package remains a release build gate.
- Added schemas, fixtures, `docs/E2E_LANGUAGE_MATRIX.md` and ADR 0029.

### Milestone v0.28.0 — Final real-model validation gate — implementation complete, release evidence pending

Implemented:

- pinned final validation plan for Rust 1.82.0, Node 22, Python 3.12, .NET 8, a fixed llama.cpp revision and a small open-weight instruct GGUF with exact SHA-256;
- real llama.cpp qualification executable using the checkpoint tokenizer/chat-template boundary, independent scoring, multi-root scoring and post-fusion isolation checks;
- cross-language E2E report v2 comparing the complete canonical response across Node/TS, Python, C and .NET/C# with explicit numerical tolerance;
- one canonical request exercising choice, boolean/noul and ordered score in the same evaluation;
- native CMake integration plus one-command Linux orchestration that builds the shim/native bindings, creates qualification evidence and runs the language matrix;
- reproducible `real-e2e.yml` workflow and versioned final-validation report/schema;
- ADR 0030 establishing that fixture/stub/offline checks are not substitutes for real-model release evidence.

Release evidence still required before declaring 1.0 readiness:

- execute the pinned v0.28 workflow on a host with network access and the required native toolchains;
- require the real llama.cpp qualification manifest to pass for the exact model/runtime/backend/hardware fingerprint;
- require Node/TS, Python, C and .NET/C# mandatory rows to pass with full semantic parity;
- publish the qualification, language-matrix and final-validation reports.

No additional architecture milestone is planned after v0.28. Failures in the final gate should produce targeted fixes, not new feature phases.

### Patch v0.28.1 — Final gate reproducibility hardening — implemented

- No new architecture phase; this patch only hardens the already-final v0.28 release gate.
- Corrected the pinned SmolLM2 Q2_K artifact size to 88,202,816 bytes while preserving its SHA-256 identity.
- Enforced exact planned host toolchain families at runtime: Rust 1.82.0, Node 22.x, Python 3.12.x and .NET 8.0.x.
- Added per-run Cargo/npm dependency freezing, `--locked` Cargo execution, `npm ci`, and lockfile SHA-256 provenance in final report v2.
- Pinned TypeScript-side development dependencies to remove avoidable host-build drift.
- Exact-pinned ABI/toolchain-sensitive Rust binding dependencies before generating the full Cargo.lock used by the evidence run.
- real-model CI no longer performs an unlocked workspace build before lock generation; the final orchestrator owns the locked workspace gates and the model run.
- The only remaining work is still the same single real-host evidence run; failures produce targeted bugfixes only.
