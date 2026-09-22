#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable server protocol for TurboJev.
//!
//! This crate owns route selection, request decoding, compatibility-profile
//! translation, batch envelopes, stable errors, and response encoding. It
//! deliberately does **not** open sockets, read files, spawn processes, inspect
//! environment variables, or choose a concrete model runtime.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use turbojev_compat::{self, WireProfile};
use turbojev_schema::{EvaluationRequest, EvaluationResponse};
use turbojev_server_execution::CrossRequestMode;

/// Stable protocol contract exposed by the local server.
pub const SERVER_PROTOCOL_CONTRACT: &str = "turbojev.server-protocol.v1";
/// Stable batch response contract.
pub const SERVER_BATCH_CONTRACT: &str = "turbojev.server-batch.v1";
/// Stable error envelope contract.
pub const SERVER_ERROR_CONTRACT: &str = "turbojev.server-error.v1";

const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_BATCH_ITEMS: usize = 64;

fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

fn default_max_batch_items() -> usize {
    DEFAULT_MAX_BATCH_ITEMS
}

/// Request-admission limits that can be validated without any host APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServerLimits {
    /// Maximum UTF-8 JSON body accepted by the protocol layer.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Maximum independent evaluation requests accepted by one batch envelope.
    #[serde(default = "default_max_batch_items")]
    pub max_batch_items: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_batch_items: DEFAULT_MAX_BATCH_ITEMS,
        }
    }
}

impl ServerLimits {
    /// Validate portable limits before a server starts accepting work.
    pub fn validate(self) -> Result<Self, ProtocolError> {
        if self.max_body_bytes == 0 {
            return Err(ProtocolError::InvalidLimits(
                "maxBodyBytes must be greater than zero".to_owned(),
            ));
        }
        if self.max_batch_items == 0 {
            return Err(ProtocolError::InvalidLimits(
                "maxBatchItems must be greater than zero".to_owned(),
            ));
        }
        Ok(self)
    }
}

/// Canonical response profile selected by a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiProfile {
    /// TurboJev's native request/response JSON.
    TurboJev,
    /// TypeSafe/System One compatibility shape.
    TypeSafeSystemOneV1,
    /// Vercel AI SDK evaluation compatibility shape already supported by
    /// `turbojev-compat`.
    AiSdkEvaluationV4,
}

impl ApiProfile {
    /// Stable profile identifier used in metadata and fixtures.
    pub const fn id(self) -> &'static str {
        match self {
            Self::TurboJev => "turbojev",
            Self::TypeSafeSystemOneV1 => "typesafe-systemone-v1",
            Self::AiSdkEvaluationV4 => "ai-sdk-evaluation-v4",
        }
    }

    fn wire_profile(self) -> WireProfile {
        match self {
            Self::TurboJev => WireProfile::TurboJev,
            Self::TypeSafeSystemOneV1 => WireProfile::TypeSafeSystemOne,
            Self::AiSdkEvaluationV4 => WireProfile::AiSdkEvaluationV4,
        }
    }
}

/// A recognized HTTP-facing route. The transport crate may map these onto a
/// concrete HTTP implementation without owning any semantic translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerRoute {
    Health,
    Ready,
    Models,
    Metrics,
    Evaluate(ApiProfile),
    Batch(ApiProfile),
}

impl ServerRoute {
    /// Profile used by evaluation routes.
    pub const fn profile(self) -> Option<ApiProfile> {
        match self {
            Self::Evaluate(profile) | Self::Batch(profile) => Some(profile),
            Self::Health | Self::Ready | Self::Models | Self::Metrics => None,
        }
    }
}

/// Fully decoded semantic command handed to the host service.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerCommand {
    Health,
    Ready,
    Models,
    Metrics,
    Evaluate {
        profile: ApiProfile,
        request: EvaluationRequest,
    },
    Batch {
        profile: ApiProfile,
        requests: Vec<EvaluationRequest>,
    },
}

/// Resolve an HTTP method/path pair without touching host networking APIs.
///
/// `openrouter` and `vercel` paths are deployment aliases only. In particular,
/// `/v1/openrouter/evaluate` does not claim OpenAI Chat Completions wire
/// compatibility; it exposes TurboJev's canonical decision request at a stable
/// gateway-friendly path.
pub fn resolve_route(method: &str, path: &str) -> Result<ServerRoute, ProtocolError> {
    let normalized_path = path.split('?').next().unwrap_or(path);
    let expected_method = match normalized_path {
        "/healthz" | "/readyz" | "/v1/models" | "/v1/server/metrics" => "GET",
        "/v1/evaluate"
        | "/v1/openrouter/evaluate"
        | "/v1/systemone"
        | "/v1/ai-sdk/evaluate"
        | "/v1/vercel/evaluate"
        | "/v1/batch"
        | "/v1/openrouter/evaluate/batch"
        | "/v1/systemone/batch"
        | "/v1/ai-sdk/evaluate/batch"
        | "/v1/vercel/evaluate/batch" => "POST",
        _ => return Err(ProtocolError::NotFound),
    };

    if method != expected_method {
        return Err(ProtocolError::MethodNotAllowed {
            allow: expected_method,
        });
    }

    let route = match normalized_path {
        "/healthz" => ServerRoute::Health,
        "/readyz" => ServerRoute::Ready,
        "/v1/models" => ServerRoute::Models,
        "/v1/server/metrics" => ServerRoute::Metrics,
        "/v1/evaluate" | "/v1/openrouter/evaluate" => ServerRoute::Evaluate(ApiProfile::TurboJev),
        "/v1/systemone" => ServerRoute::Evaluate(ApiProfile::TypeSafeSystemOneV1),
        "/v1/ai-sdk/evaluate" | "/v1/vercel/evaluate" => {
            ServerRoute::Evaluate(ApiProfile::AiSdkEvaluationV4)
        }
        "/v1/batch" | "/v1/openrouter/evaluate/batch" => ServerRoute::Batch(ApiProfile::TurboJev),
        "/v1/systemone/batch" => ServerRoute::Batch(ApiProfile::TypeSafeSystemOneV1),
        "/v1/ai-sdk/evaluate/batch" | "/v1/vercel/evaluate/batch" => {
            ServerRoute::Batch(ApiProfile::AiSdkEvaluationV4)
        }
        _ => return Err(ProtocolError::NotFound),
    };
    Ok(route)
}

/// Decode a complete request body after transport-level size enforcement.
pub fn decode_command(
    method: &str,
    path: &str,
    body: &str,
    limits: ServerLimits,
) -> Result<ServerCommand, ProtocolError> {
    let limits = limits.validate()?;
    if body.len() > limits.max_body_bytes {
        return Err(ProtocolError::BodyTooLarge {
            actual: body.len(),
            maximum: limits.max_body_bytes,
        });
    }

    match resolve_route(method, path)? {
        ServerRoute::Health => Ok(ServerCommand::Health),
        ServerRoute::Ready => Ok(ServerCommand::Ready),
        ServerRoute::Models => Ok(ServerCommand::Models),
        ServerRoute::Metrics => Ok(ServerCommand::Metrics),
        ServerRoute::Evaluate(profile) => Ok(ServerCommand::Evaluate {
            profile,
            request: decode_one(profile, body)?,
        }),
        ServerRoute::Batch(profile) => {
            let raw: RawBatch = serde_json::from_str(body).map_err(ProtocolError::DecodeBatch)?;
            if raw.requests.is_empty() {
                return Err(ProtocolError::EmptyBatch);
            }
            if raw.requests.len() > limits.max_batch_items {
                return Err(ProtocolError::BatchTooLarge {
                    actual: raw.requests.len(),
                    maximum: limits.max_batch_items,
                });
            }
            let mut requests = Vec::with_capacity(raw.requests.len());
            for (index, value) in raw.requests.into_iter().enumerate() {
                let encoded = serde_json::to_string(&value).map_err(ProtocolError::EncodeJson)?;
                let request = decode_one(profile, &encoded).map_err(|error| {
                    ProtocolError::InvalidBatchItem {
                        index,
                        message: error.to_string(),
                    }
                })?;
                requests.push(request);
            }
            Ok(ServerCommand::Batch { profile, requests })
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBatch {
    requests: Vec<Value>,
}

fn decode_one(profile: ApiProfile, input: &str) -> Result<EvaluationRequest, ProtocolError> {
    match profile {
        ApiProfile::TurboJev => turbojev_wire::decode_request(input)
            .map_err(|error| ProtocolError::InvalidRequest(error.to_string())),
        ApiProfile::TypeSafeSystemOneV1 | ApiProfile::AiSdkEvaluationV4 => {
            turbojev_compat::decode_request(profile.wire_profile(), input)
                .map_err(|error| ProtocolError::InvalidRequest(error.to_string()))
        }
    }
}

/// Encode one canonical response using the route-selected compatibility profile.
pub fn encode_evaluation_response(
    profile: ApiProfile,
    request: &EvaluationRequest,
    response: &EvaluationResponse,
) -> Result<String, ProtocolError> {
    match profile {
        ApiProfile::TurboJev => turbojev_wire::encode_response(response)
            .map_err(|error| ProtocolError::Encode(error.to_string())),
        ApiProfile::TypeSafeSystemOneV1 | ApiProfile::AiSdkEvaluationV4 => {
            turbojev_compat::encode_response(profile.wire_profile(), request, response)
                .map_err(|error| ProtocolError::Encode(error.to_string()))
        }
    }
}

/// Encode an ordered batch. Requests and responses must remain index-aligned.
pub fn encode_batch_response(
    profile: ApiProfile,
    requests: &[EvaluationRequest],
    responses: &[EvaluationResponse],
) -> Result<String, ProtocolError> {
    encode_batch_response_with_execution(profile, requests, responses, CrossRequestMode::Serial)
}

/// Encode an ordered batch while truthfully reporting how independent requests
/// were executed by the server scheduler.
pub fn encode_batch_response_with_execution(
    profile: ApiProfile,
    requests: &[EvaluationRequest],
    responses: &[EvaluationResponse],
    execution: CrossRequestMode,
) -> Result<String, ProtocolError> {
    if requests.len() != responses.len() {
        return Err(ProtocolError::ResponseCountMismatch {
            requests: requests.len(),
            responses: responses.len(),
        });
    }

    let mut values = Vec::with_capacity(responses.len());
    for (request, response) in requests.iter().zip(responses) {
        let encoded = encode_evaluation_response(profile, request, response)?;
        let value: Value = serde_json::from_str(&encoded).map_err(ProtocolError::EncodeJson)?;
        values.push(value);
    }

    serde_json::to_string(&json!({
        "responses": values,
        "_turbojev": {
            "contract": SERVER_BATCH_CONTRACT,
            "profile": profile.id(),
            "count": responses.len(),
            "crossRequestExecution": execution.as_str()
        }
    }))
    .map_err(ProtocolError::EncodeJson)
}

/// JSON body returned for stable protocol failures.
pub fn encode_error(error: &ProtocolError) -> String {
    json!({
        "error": {
            "code": error.code(),
            "message": error.to_string()
        },
        "_turbojev": {
            "contract": SERVER_ERROR_CONTRACT
        }
    })
    .to_string()
}

/// Portable protocol failure with an HTTP status mapping owned by this crate.
#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("server limits are invalid: {0}")]
    InvalidLimits(String),
    #[error("route not found")]
    NotFound,
    #[error("method not allowed; expected {allow}")]
    MethodNotAllowed { allow: &'static str },
    #[error("request body is {actual} bytes; maximum is {maximum}")]
    BodyTooLarge { actual: usize, maximum: usize },
    #[error("invalid evaluation request: {0}")]
    InvalidRequest(String),
    #[error("invalid batch JSON: {0}")]
    DecodeBatch(#[source] serde_json::Error),
    #[error("batch must contain at least one request")]
    EmptyBatch,
    #[error("batch has {actual} requests; maximum is {maximum}")]
    BatchTooLarge { actual: usize, maximum: usize },
    #[error("batch item {index} is invalid: {message}")]
    InvalidBatchItem { index: usize, message: String },
    #[error("cannot encode server response: {0}")]
    Encode(String),
    #[error("cannot encode JSON: {0}")]
    EncodeJson(#[source] serde_json::Error),
    #[error("response count {responses} does not match request count {requests}")]
    ResponseCountMismatch { requests: usize, responses: usize },
}

impl ProtocolError {
    /// HTTP status associated with the stable protocol failure.
    pub const fn status(&self) -> u16 {
        match self {
            Self::NotFound => 404,
            Self::MethodNotAllowed { .. } => 405,
            Self::BodyTooLarge { .. } | Self::BatchTooLarge { .. } => 413,
            Self::InvalidLimits(_)
            | Self::InvalidRequest(_)
            | Self::DecodeBatch(_)
            | Self::EmptyBatch
            | Self::InvalidBatchItem { .. } => 400,
            Self::Encode(_) | Self::EncodeJson(_) | Self::ResponseCountMismatch { .. } => 500,
        }
    }

    /// Stable machine-readable error code.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidLimits(_) => "INVALID_LIMITS",
            Self::NotFound => "NOT_FOUND",
            Self::MethodNotAllowed { .. } => "METHOD_NOT_ALLOWED",
            Self::BodyTooLarge { .. } => "BODY_TOO_LARGE",
            Self::InvalidRequest(_) => "INVALID_REQUEST",
            Self::DecodeBatch(_) => "INVALID_BATCH_JSON",
            Self::EmptyBatch => "EMPTY_BATCH",
            Self::BatchTooLarge { .. } => "BATCH_TOO_LARGE",
            Self::InvalidBatchItem { .. } => "INVALID_BATCH_ITEM",
            Self::Encode(_) | Self::EncodeJson(_) => "ENCODE_ERROR",
            Self::ResponseCountMismatch { .. } => "RESPONSE_COUNT_MISMATCH",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &str = r#"{
        "state":"package missing",
        "questions":{
            "route":{
                "type":"choice",
                "instructions":"route",
                "criteria":{"billing":"billing","shipping":"shipping"}
            }
        }
    }"#;

    #[test]
    fn route_aliases_do_not_change_semantics() {
        assert_eq!(resolve_route("GET", "/readyz").unwrap(), ServerRoute::Ready);
        assert_eq!(
            resolve_route("POST", "/v1/openrouter/evaluate").unwrap(),
            ServerRoute::Evaluate(ApiProfile::TurboJev)
        );
        assert_eq!(
            resolve_route("POST", "/v1/vercel/evaluate").unwrap(),
            ServerRoute::Evaluate(ApiProfile::AiSdkEvaluationV4)
        );
    }

    #[test]
    fn batch_preserves_request_order() {
        let body = format!(r#"{{"requests":[{REQUEST},{REQUEST}]}}"#);
        let command = decode_command("POST", "/v1/batch", &body, ServerLimits::default()).unwrap();
        match command {
            ServerCommand::Batch { requests, .. } => assert_eq!(requests.len(), 2),
            _ => panic!("expected batch command"),
        }
    }

    #[test]
    fn method_is_rejected_before_body_parsing() {
        let error =
            decode_command("GET", "/v1/evaluate", "not-json", ServerLimits::default()).unwrap_err();
        assert_eq!(error.status(), 405);
        assert_eq!(error.code(), "METHOD_NOT_ALLOWED");
    }
}
