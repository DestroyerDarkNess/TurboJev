#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Stable JSON boundary for language bindings and compatibility adapters.
//!
//! Bindings should delegate parsing/serialization to this crate instead of
//! recreating the request contract in JavaScript, Python, C#, Swift, Kotlin,
//! or any other host language.

use thiserror::Error;
use turbojev_schema::{EvaluationRequest, EvaluationResponse, SchemaError};

/// Parse and validate a UTF-8 JSON request.
pub fn decode_request(json: &str) -> Result<EvaluationRequest, WireError> {
    let request: EvaluationRequest = serde_json::from_str(json).map_err(WireError::Decode)?;
    request.validate().map_err(WireError::Schema)?;
    Ok(request)
}

/// Serialize a native TurboJev response to compact UTF-8 JSON.
pub fn encode_response(response: &EvaluationResponse) -> Result<String, WireError> {
    serde_json::to_string(response).map_err(WireError::Encode)
}

/// Stable wire-boundary error.
#[derive(Debug, Error)]
pub enum WireError {
    /// JSON could not be decoded.
    #[error("invalid TurboJev JSON request: {0}")]
    Decode(#[source] serde_json::Error),
    /// Decoded JSON violates the TurboJev schema.
    #[error("invalid TurboJev request: {0}")]
    Schema(#[source] SchemaError),
    /// Response serialization failed.
    #[error("failed to encode TurboJev response: {0}")]
    Encode(#[source] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_alias_is_available_through_wire_boundary() {
        let request = decode_request(
            r#"{
                "state":"hola",
                "questions":{
                    "ok":{"type":"boolean","instructions":"¿Es correcto?"}
                }
            }"#,
        )
        .unwrap();
        assert_eq!(request.questions.len(), 1);
    }
}
