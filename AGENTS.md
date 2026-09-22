# Contexto del proyecto: TurboJev

Resumen guardado a petición del usuario el 2026-09-21. Basado en README.md,
SPEC.md, Cargo.toml y CHANGELOG.md; no constituye una auditoría del código
ni evidencia de que las pruebas hayan pasado. Revalidar el estado al hacer cambios.

## Propósito

TurboJev es una biblioteca multiplataforma, escrita principalmente en Rust, para
tomar decisiones semánticas tipadas con modelos de pesos abiertos. Su flujo es:
estado + preguntas tipadas -> puntuaciones de candidatos -> probabilidades ->
respuestas tipadas. Sirve para clasificación, enrutamiento y evaluación de estados.
La vía optimizada lee logits de los candidatos sin generar texto ni JSON con el modelo.

Tipos centrales: `choice` (selección entre opciones), `noul` (decisión booleana;
`boolean` es alias de entrada) y `score` (evaluación con criterios ordenados).
Ofrece una API sencilla `classify` y una API completa `evaluate`.

## Arquitectura y principios

- La lógica semántica, validación, planificación, calibración y construcción de
  respuestas vive una sola vez en crates Rust pequeños y separados.
- El núcleo portable es independiente del modelo, sistema operativo, hardware,
  runtime, idioma, plantilla de prompt y ejecutor asíncrono. No incorpora acceso
  directo a archivos, red, procesos o entorno: eso pertenece a los hosts/adaptadores.
- Los adaptadores usan la plantilla y el tokenizer reales del modelo. Nunca
  adivinar ni fijar IDs de candidatos por familia de modelos.
- Bindings finos para Node/TypeScript, navegador/WASM, Python, C, .NET, Android y
  Swift/iOS comparten contratos Rust y JSON. Hay integración opcional llama.cpp/GGUF,
  adaptador web Transformers.js y conexión de runtimes personalizados.
- Optimizaciones: reutilización de prefijos, sufijos paralelos, proyección selectiva
  de candidatos y fusión de solicitudes cuando el runtime lo permita y lo demuestre.
  Conservar una ruta independiente de referencia y verificar paridad numérica.
- Softmax condicional no equivale a confianza calibrada. La calibración y su
  procedencia son explícitas. No anunciar capacidades que el runtime no implemente.
- Inferencia local y ausencia de telemetría obligatoria son principios del proyecto.
- Inspirado en ideas de SemIf y Nimble; no es un fork. La compatibilidad con Jev es
  de contratos y semántica, sin afirmar equivalencia con pesos o calibración privados.
- Incluye contratos, datasets, artefactos y evaluación reproducible de modelos
  especializados; el entrenamiento LoRA/QLoRA se realiza fuera de TurboJev.

## Estado observado

Cargo.toml y CHANGELOG.md indican v0.28.2. La cabecera v0.20 del README está
desactualizada respecto a sus secciones finales y al manifiesto.
El foco reciente es la validación reproducible con modelos reales y paridad de
respuestas completas entre Node, Python, C y .NET; también la calificación de
llama.cpp multi-root con evidencia ligada a modelo/runtime/hardware.
v0.28.2 corrige la liberación del hilo de afinidad .NET ante fallos y usa el binario
Release precompilado en E2E. No asumir preparación para 1.0 sin el reporte real:
pruebas con fixtures o stubs no sustituyen evidencia con un modelo real.

## Orientación

- README.md: producto, API e integraciones.
- SPEC.md: contrato portable; su versión propia es distinta de la del workspace.
- CHANGELOG.md y Cargo.toml: cambios y versión del workspace.
- crates/: núcleo, adaptadores, hosts y herramientas.
- bindings/: paquetes para otros lenguajes/plataformas.
- docs/, schemas/, tests/, scripts/ y benchmarks/: contratos, documentación y validación.

Para preguntas estructurales del código, respetar las instrucciones CodeGraph del
entorno y preferir sus herramientas cuando estén disponibles.

## Validación real posterior (2026-09-21)

Se ejecutó SmolLM2-135M-Instruct Q2_K en Windows con paridad completa entre Node,
Python, C y .NET; 108 pruebas Rust y comprobaciones reales de robustez pasaron.
Se corrigieron fallos de compilación, contadores de tokens de llama.cpp, rutas
Windows y cierre de workers. Consultar docs/WINDOWS_VALIDATION.md para resultados,
limitaciones y reproducción. Clippy estricto sigue pendiente. La fusión multi-root
no pasó la tolerancia numérica y debe continuar desactivada en esta configuración.
Esto no constituye aprobación del gate Linux ni preparación automática para 1.0.
