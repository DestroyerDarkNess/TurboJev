#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Compatibility adapters for external evaluation wire contracts.
//!
//! This crate translates wire shapes only. It never performs inference,
//! tokenization, prompt rendering, calibration, filesystem access, networking,
//! or platform-specific work.

use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use thiserror::Error;
use turbojev_schema::{Answer, EvaluationRequest, EvaluationResponse, Question};

/// Stable compatibility profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireProfile {
    /// TurboJev's canonical native JSON contract.
    TurboJev,
    /// TypeSafe-style `/v1/systemone` contract (`noul`, `choice`, `score`).
    TypeSafeSystemOne,
    /// AI SDK v4 evaluation contract (`boolean`, `choice`, `score`).
    AiSdkEvaluationV4,
}

impl WireProfile {
    /// Stable lowercase profile identifier suitable for configuration files.
    pub const fn id(self) -> &'static str {
        match self {
            Self::TurboJev => "turbojev",
            Self::TypeSafeSystemOne => "typesafe-systemone-v1",
            Self::AiSdkEvaluationV4 => "ai-sdk-evaluation-v4",
        }
    }
}

/// Decode an external request into TurboJev's canonical request.
///
/// The canonical schema already accepts `boolean` as an alias of `noul`, so
/// decoding is intentionally strict and small: compatibility is a wire concern,
/// not a second domain model.
pub fn decode_request(profile: WireProfile, input: &str) -> Result<EvaluationRequest, CompatError> {
    let mut value: Value = serde_json::from_str(input).map_err(CompatError::Decode)?;
    normalize_request_tags(profile, &mut value)?;
    let request: EvaluationRequest = serde_json::from_value(value).map_err(CompatError::Decode)?;
    request.validate().map_err(CompatError::InvalidRequest)?;
    Ok(request)
}

/// Encode a canonical request using one compatibility profile.
pub fn encode_request(
    profile: WireProfile,
    request: &EvaluationRequest,
) -> Result<String, CompatError> {
    request.validate().map_err(CompatError::InvalidRequest)?;
    let value = request_to_profile(profile, request);
    serde_json::to_string(&value).map_err(CompatError::Encode)
}

/// Encode a canonical response using one compatibility profile.
///
/// `request` is required because some external contracts shape answers based on
/// the declared question type rather than carrying an answer type tag.
pub fn encode_response(
    profile: WireProfile,
    request: &EvaluationRequest,
    response: &EvaluationResponse,
) -> Result<String, CompatError> {
    ensure_answer_keys(request, response)?;
    let value = match profile {
        WireProfile::TurboJev => serde_json::to_value(response).map_err(CompatError::Encode)?,
        WireProfile::TypeSafeSystemOne => typesafe_response(response),
        WireProfile::AiSdkEvaluationV4 => ai_sdk_response(request, response)?,
    };
    serde_json::to_string(&value).map_err(CompatError::Encode)
}

fn normalize_request_tags(profile: WireProfile, value: &mut Value) -> Result<(), CompatError> {
    let questions = value
        .as_object_mut()
        .and_then(|root| root.get_mut("questions"))
        .and_then(Value::as_object_mut)
        .ok_or(CompatError::MissingQuestions)?;

    for question in questions.values_mut() {
        let Some(object) = question.as_object_mut() else {
            return Err(CompatError::InvalidQuestionShape);
        };
        let Some(kind) = object.get_mut("type") else {
            return Err(CompatError::MissingQuestionType);
        };
        let Some(kind_text) = kind.as_str().map(str::to_owned) else {
            return Err(CompatError::InvalidQuestionType);
        };
        match profile {
            WireProfile::AiSdkEvaluationV4 if kind_text == "boolean" => {
                *kind = Value::String("noul".to_owned());
            }
            WireProfile::TypeSafeSystemOne | WireProfile::TurboJev if kind_text == "boolean" => {
                // TurboJev intentionally accepts the AI SDK alias on input too.
                *kind = Value::String("noul".to_owned());
            }
            _ => {}
        }
    }
    Ok(())
}

fn request_to_profile(profile: WireProfile, request: &EvaluationRequest) -> Value {
    let mut questions = Map::new();
    for (id, question) in &request.questions {
        let value = match question {
            Question::Choice {
                instructions,
                criteria,
            } => json!({
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }),
            Question::Noul { instructions } => json!({
                "type": if profile == WireProfile::AiSdkEvaluationV4 { "boolean" } else { "noul" },
                "instructions": instructions,
            }),
            Question::Score {
                instructions,
                criteria,
            } => json!({
                "type": "score",
                "instructions": instructions,
                "criteria": criteria,
            }),
        };
        questions.insert(id.clone(), value);
    }

    let mut root = Map::new();
    if let Some(model) = &request.model {
        root.insert("model".to_owned(), Value::String(model.clone()));
    }
    root.insert("state".to_owned(), request.state.clone());
    root.insert("questions".to_owned(), Value::Object(questions));
    Value::Object(root)
}

fn typesafe_response(response: &EvaluationResponse) -> Value {
    let answers = response
        .answers
        .iter()
        .map(|(id, answer)| (id.clone(), typesafe_answer(answer)))
        .collect::<Map<String, Value>>();

    let mut usage = Map::new();
    if let Some(input_tokens) = response.usage.input_tokens {
        usage.insert("input_tokens".to_owned(), json!(input_tokens));
    }

    let mut root = Map::new();
    root.insert("answers".to_owned(), Value::Object(answers));
    if !usage.is_empty() {
        root.insert("usage".to_owned(), Value::Object(usage));
    }
    Value::Object(root)
}

fn typesafe_answer(answer: &Answer) -> Value {
    match answer {
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => json!({
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities,
        }),
        Answer::Noul { noul } => json!({
            "type": "noul",
            "noul": noul,
        }),
        Answer::Score {
            score,
            confidence,
            probabilities,
            legend,
        } => json!({
            "type": "score",
            "score": score,
            "confidence": confidence,
            "probabilities": probabilities,
            "legend": legend,
        }),
    }
}

fn ai_sdk_response(
    request: &EvaluationRequest,
    response: &EvaluationResponse,
) -> Result<Value, CompatError> {
    let mut answers = Map::new();
    let mut confidence = Map::new();

    for (id, question) in &request.questions {
        let answer = response
            .answers
            .get(id)
            .ok_or_else(|| CompatError::MissingAnswer(id.clone()))?;
        let value = match (question, answer) {
            (
                Question::Choice { .. },
                Answer::Choice {
                    choice,
                    confidence: c,
                    probabilities,
                },
            ) => {
                confidence.insert(id.clone(), json!(c));
                json!({ "choice": choice, "probabilities": probabilities })
            }
            (Question::Noul { .. }, Answer::Noul { noul }) => {
                json!({ "probability": noul })
            }
            (
                Question::Score { .. },
                Answer::Score {
                    score,
                    confidence: c,
                    probabilities,
                    ..
                },
            ) => {
                confidence.insert(id.clone(), json!(c));
                json!({ "score": score, "probabilities": probabilities })
            }
            _ => return Err(CompatError::AnswerTypeMismatch(id.clone())),
        };
        answers.insert(id.clone(), value);
    }

    let mut usage = Map::new();
    if let Some(input_tokens) = response.usage.input_tokens {
        usage.insert("inputTokens".to_owned(), json!(input_tokens));
    }

    // AI SDK's TypeSafe provider exposes Choice/Score confidence in provider
    // metadata. TurboJev mirrors that compatibility field, but also records
    // provenance so callers can see that generic-model confidence is a
    // TurboJev candidate-distribution proxy unless a fitted calibrator is used.
    let provider_metadata = json!({
        "typesafe": { "confidence": confidence },
        "turbojev": {
            "compatibilityProfile": WireProfile::AiSdkEvaluationV4.id(),
            "confidenceSource": response.metadata.confidence_method,
            "calibration": response.metadata.calibration,
            "calibrated": response.metadata.calibrated,
            "contract": response.metadata.contract,
        }
    });

    let mut root = Map::new();
    root.insert("answers".to_owned(), Value::Object(answers));
    if !usage.is_empty() {
        root.insert("usage".to_owned(), Value::Object(usage));
    }
    root.insert("providerMetadata".to_owned(), provider_metadata);
    Ok(Value::Object(root))
}

fn ensure_answer_keys(
    request: &EvaluationRequest,
    response: &EvaluationResponse,
) -> Result<(), CompatError> {
    if request.questions.len() != response.answers.len() {
        return Err(CompatError::AnswerCountMismatch {
            expected: request.questions.len(),
            actual: response.answers.len(),
        });
    }
    for id in request.questions.keys() {
        if !response.answers.contains_key(id) {
            return Err(CompatError::MissingAnswer(id.clone()));
        }
    }
    Ok(())
}

/// Compatibility translation failure.
#[derive(Debug, Error)]
pub enum CompatError {
    /// Invalid JSON.
    #[error("invalid compatibility JSON: {0}")]
    Decode(#[source] serde_json::Error),
    /// JSON serialization failure.
    #[error("failed to encode compatibility JSON: {0}")]
    Encode(#[source] serde_json::Error),
    /// Canonical request validation failed.
    #[error("invalid evaluation request: {0}")]
    InvalidRequest(#[source] turbojev_schema::SchemaError),
    /// Questions member missing or invalid.
    #[error("request must contain a questions object")]
    MissingQuestions,
    /// One question is not an object.
    #[error("each question must be an object")]
    InvalidQuestionShape,
    /// One question has no type.
    #[error("each question must declare a type")]
    MissingQuestionType,
    /// Question type is not a string.
    #[error("question type must be a string")]
    InvalidQuestionType,
    /// Response answer count differs from the request.
    #[error("response has {actual} answers; expected {expected}")]
    AnswerCountMismatch { expected: usize, actual: usize },
    /// Response omitted a named answer.
    #[error("response omitted answer {0:?}")]
    MissingAnswer(String),
    /// Canonical answer type differs from the declared question type.
    #[error("answer type does not match question {0:?}")]
    AnswerTypeMismatch(String),
}

/// Helper for bindings that want a stable ordered map from a serialized answer
/// distribution without exposing a concrete map implementation.
pub fn ordered_probabilities(answer: &Answer) -> Option<IndexMap<String, f64>> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            Some(probabilities.clone())
        }
        Answer::Noul { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::indexmap;
    use turbojev_schema::{EngineMetadata, ExecutionMetadata, Usage};

    fn request() -> EvaluationRequest {
        EvaluationRequest {
            model: Some("open-model".to_owned()),
            state: json!({"message": "hola"}),
            questions: indexmap! {
                "route".into() => Question::choice("ruta", indexmap! {
                    "a".into() => "A".into(),
                    "b".into() => "B".into(),
                }),
                "ok".into() => Question::noul("¿es correcto?"),
                "score".into() => Question::score("nivel", vec!["bajo".into(), "alto".into()]),
            },
        }
    }

    fn response() -> EvaluationResponse {
        EvaluationResponse {
            model: Some("open-model".to_owned()),
            answers: indexmap! {
                "route".into() => Answer::Choice {
                    choice: "b".into(), confidence: 0.9,
                    probabilities: indexmap!{"a".into()=>0.1,"b".into()=>0.9},
                },
                "ok".into() => Answer::Noul { noul: 0.8 },
                "score".into() => Answer::Score {
                    score: 0.75, confidence: 0.75,
                    probabilities: indexmap!{"0".into()=>0.25,"1".into()=>0.75},
                    legend: indexmap!{"0".into()=>"bajo".into(),"1".into()=>"alto".into()},
                },
            },
            usage: Usage {
                decisions: 3,
                input_tokens: Some(42),
                generated_tokens: 0,
            },
            metadata: EngineMetadata {
                backend: "fixture".into(),
                calibration: "softmax".into(),
                calibrated: false,
                confidence_method: "selected-candidate-probability".into(),
                contract: "turbojev.core.v4".into(),
                execution: ExecutionMetadata {
                    prefix_reuse: "none".into(),
                    scheduling: "serial".into(),
                    parallel_task_limit: 1,
                    projection: "full-vocabulary".into(),
                },
            },
        }
    }

    #[test]
    fn ai_sdk_boolean_round_trip_uses_boolean_request_and_probability_answer() {
        let req = request();
        let encoded = encode_request(WireProfile::AiSdkEvaluationV4, &req).unwrap();
        assert!(
            encoded.contains("\\\"type\\\":\\\"boolean\\\"")
                || encoded.contains("\"type\":\"boolean\"")
        );
        let decoded = decode_request(WireProfile::AiSdkEvaluationV4, &encoded).unwrap();
        assert_eq!(decoded, req);

        let output = encode_response(WireProfile::AiSdkEvaluationV4, &req, &response()).unwrap();
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["answers"]["ok"]["probability"], json!(0.8));
        assert_eq!(
            value["providerMetadata"]["typesafe"]["confidence"]["route"],
            json!(0.9)
        );
    }

    #[test]
    fn typesafe_profile_uses_noul() {
        let encoded = encode_request(WireProfile::TypeSafeSystemOne, &request()).unwrap();
        let value: Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(value["questions"]["ok"]["type"], json!("noul"));
    }
}
