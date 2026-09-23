#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! TurboJev facade crate.
//!
//! The facade exposes portable contracts only. Concrete model runtimes live in
//! separate optional adapter crates and are intentionally not required here.

pub use turbojev_core::{
    ClassificationResult, Error, OrderRobustness, TurboJevAsyncEngine, TurboJevEngine,
};

/// Default engine alias using the softmax calibrator.
pub type TurboJev<B> = TurboJevEngine<B, turbojev_calibration::SoftmaxCalibrator>;

/// Default asynchronous engine alias using the softmax calibrator.
pub type TurboJevAsync<B> = TurboJevAsyncEngine<B, turbojev_calibration::SoftmaxCalibrator>;

/// Public schema types.
pub mod schema {
    pub use turbojev_schema::*;
}

/// Backend SPI types.
pub mod backend {
    pub use turbojev_backend::*;
}

/// Portable execution planner.
pub mod engine {
    pub use turbojev_engine::*;
}

/// Runtime-independent model descriptors.
pub mod model {
    pub use turbojev_model::*;
}

/// Portable contracts for decision-specialized checkpoints and training recipes.
pub mod specialized {
    pub use turbojev_specialized::*;
}

/// Deterministic backend intended for tests and SDK development.
pub mod backend_mock {
    pub use turbojev_backend_mock::*;
}

/// Adapter for application-provided/open-model runtimes.
pub mod backend_custom {
    pub use turbojev_backend_custom::*;
}

/// Calibration primitives.
pub mod calibration {
    pub use turbojev_calibration::*;
}

/// Prompt IR primitives for model/runtime adapters.
pub mod prompt {
    pub use turbojev_prompt::*;
}

/// Portable candidate-only output projection math and parity diagnostics.
pub mod projection {
    pub use turbojev_projection::*;
}

/// Tokenizer-safe single- and multi-token candidate-code discovery.
pub mod tokenizer {
    pub use turbojev_tokenizer::*;
}

/// Stable JSON boundary for cross-language bindings.
pub mod wire {
    pub use turbojev_wire::*;
}

/// Jev/TypeSafe/AI SDK wire compatibility profiles.
pub mod compat {
    pub use turbojev_compat::*;
}

/// Portable conformance and numerical parity tooling.
pub mod conformance {
    pub use turbojev_conformance::*;
}

/// Universal token-level bridge for model/runtime adapters.
pub mod runtime {
    pub use turbojev_runtime::*;
}

/// Exact prefix-sharing multi-token candidate scoring.
pub mod sequence {
    pub use turbojev_sequence::*;
}

/// Conformance kit for tokenizer/runtime adapter authors.
pub mod runtime_conformance {
    pub use turbojev_runtime_conformance::*;
}

/// Portable dense candidate-only projection runtime.
pub mod runtime_projection {
    pub use turbojev_runtime_projection::*;
}

/// Semantic training datasets and deterministic specialization materialization.
pub mod training_data {
    pub use turbojev_training_data::*;
}

/// Model-bound specialized artifact generation and compatibility gates.
pub mod specialized_artifact {
    pub use turbojev_specialized_artifact::*;
}

/// Generic-vs-specialized evaluation and external-run interchange.
pub mod specialized_eval {
    pub use turbojev_specialized_eval::*;
}

/// Reproducible evidence contracts for runtime capability qualification.
pub mod runtime_qualification {
    pub use turbojev_runtime_qualification::*;
}
