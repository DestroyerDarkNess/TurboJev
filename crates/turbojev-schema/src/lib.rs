#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Public, backend-neutral TurboJev request and response types.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub mod media;

/// Maximum number of questions accepted by the portable core in one request.
pub const DEFAULT_MAX_QUESTIONS: usize = 256;

/// Maximum number of candidates accepted by the portable core for a choice.
pub const DEFAULT_MAX_CHOICES: usize = 256;

/// One TurboJev evaluation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRequest {
    /// Optional requested model identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Shared state/evidence. Strings, objects, and arrays are all supported.
    pub state: Value,
    /// Named typed questions. Input order is preserved.
    pub questions: IndexMap<String, Question>,
}

impl EvaluationRequest {
    /// Validate portable invariants before any backend work starts.
    pub fn validate(&self) -> Result<(), SchemaError> {
        validate_state(&self.state)?;

        if self.questions.is_empty() {
            return Err(SchemaError::NoQuestions);
        }
        if self.questions.len() > DEFAULT_MAX_QUESTIONS {
            return Err(SchemaError::TooManyQuestions {
                actual: self.questions.len(),
                max: DEFAULT_MAX_QUESTIONS,
            });
        }

        for (id, question) in &self.questions {
            validate_identifier("question", id)?;
            question.validate(id)?;
        }

        Ok(())
    }
}

/// A typed semantic question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Question {
    /// Choose exactly one named criterion.
    #[serde(rename = "choice")]
    Choice {
        /// Natural-language decision instruction.
        instructions: String,
        /// Stable output key -> human-readable criterion description.
        criteria: IndexMap<String, String>,
    },
    /// Boolean/noul decision. `boolean` is accepted as a compatibility alias.
    #[serde(rename = "noul", alias = "boolean")]
    Noul {
        /// Natural-language condition to judge.
        instructions: String,
    },
    /// Ordered score decision.
    #[serde(rename = "score")]
    Score {
        /// Natural-language scoring instruction.
        instructions: String,
        /// Ordered level descriptions, starting at zero.
        criteria: Vec<String>,
    },
}

impl Question {
    /// Construct a choice question.
    pub fn choice(instructions: impl Into<String>, criteria: IndexMap<String, String>) -> Self {
        Self::Choice {
            instructions: instructions.into(),
            criteria,
        }
    }

    /// Construct a noul/boolean question.
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
        }
    }

    /// Construct an ordered score question.
    pub fn score(instructions: impl Into<String>, criteria: Vec<String>) -> Self {
        Self::Score {
            instructions: instructions.into(),
            criteria,
        }
    }

    /// Return the decision instruction.
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Noul { instructions }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    fn validate(&self, question_id: &str) -> Result<(), SchemaError> {
        validate_nonempty("instructions", question_id, self.instructions())?;

        match self {
            Self::Choice { criteria, .. } => {
                if criteria.len() < 2 {
                    return Err(SchemaError::TooFewChoices {
                        question: question_id.to_owned(),
                    });
                }
                if criteria.len() > DEFAULT_MAX_CHOICES {
                    return Err(SchemaError::TooManyChoices {
                        question: question_id.to_owned(),
                        actual: criteria.len(),
                        max: DEFAULT_MAX_CHOICES,
                    });
                }
                for (key, description) in criteria {
                    validate_identifier("choice", key)?;
                    validate_nonempty("choice description", question_id, description)?;
                }
            }
            Self::Noul { .. } => {}
            Self::Score { criteria, .. } => {
                if criteria.len() < 2 {
                    return Err(SchemaError::TooFewScoreLevels {
                        question: question_id.to_owned(),
                    });
                }
                if criteria.len() > DEFAULT_MAX_CHOICES {
                    return Err(SchemaError::TooManyChoices {
                        question: question_id.to_owned(),
                        actual: criteria.len(),
                        max: DEFAULT_MAX_CHOICES,
                    });
                }
                for description in criteria {
                    validate_nonempty("score level", question_id, description)?;
                }
            }
        }

        Ok(())
    }
}

/// One typed answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Answer {
    /// Choice result.
    #[serde(rename = "choice")]
    Choice {
        /// Selected stable criterion key.
        choice: String,
        /// Probability assigned to the selected criterion.
        confidence: f64,
        /// Conditional distribution across all supplied criteria.
        probabilities: IndexMap<String, f64>,
    },
    /// Boolean/noul result. The value is P(true).
    #[serde(rename = "noul")]
    Noul {
        /// Conditional probability that the condition is true.
        noul: f64,
    },
    /// Ordered score result.
    #[serde(rename = "score")]
    Score {
        /// Probability-weighted expected level index.
        score: f64,
        /// Maximum probability among supplied levels.
        confidence: f64,
        /// Conditional distribution keyed by level index as a string.
        probabilities: IndexMap<String, f64>,
        /// Level-index -> user-supplied level description.
        legend: IndexMap<String, String>,
    },
}

/// Portable usage metadata produced by the core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Number of typed decisions completed.
    pub decisions: usize,
    /// Number of input tokens processed by the runtime, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<usize>,
    /// Number of autoregressively generated answer tokens.
    ///
    /// The TurboJev direct-logit fast path keeps this at zero.
    pub generated_tokens: usize,
}

/// Stable, backend-neutral description of the execution path used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// `none` or `shared`.
    pub prefix_reuse: String,
    /// `serial` or `parallel`.
    pub scheduling: String,
    /// Maximum task count submitted together when scheduling is parallel.
    pub parallel_task_limit: usize,
    /// `full-vocabulary` or `candidates-only`.
    pub projection: String,
}

/// TurboJev-specific metadata. Compatibility adapters may omit this on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineMetadata {
    /// Backend implementation name.
    pub backend: String,
    /// Calibration method applied to candidate logits.
    pub calibration: String,
    /// Whether probabilities were produced by a calibrator fitted on held-out data.
    pub calibrated: bool,
    /// How the generic confidence field is derived.
    pub confidence_method: String,
    /// Stable architecture contract identifier.
    pub contract: String,
    /// Portable execution strategy chosen through capability negotiation.
    pub execution: ExecutionMetadata,
}

/// Evaluation result in TurboJev's native representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResponse {
    /// Model identifier resolved by the backend, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Typed answers, preserving input question order.
    pub answers: IndexMap<String, Answer>,
    /// Portable usage counters.
    pub usage: Usage,
    /// TurboJev-specific execution metadata.
    #[serde(rename = "_turbojev")]
    pub metadata: EngineMetadata,
}

/// Schema validation error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// State must be string, object, or array and cannot be empty/null.
    #[error("state must be a non-empty string, object, or array")]
    InvalidState,
    /// Explicit media evidence failed portable validation.
    #[error("invalid media evidence: {0}")]
    InvalidMedia(String),
    /// No questions were supplied.
    #[error("at least one question is required")]
    NoQuestions,
    /// Too many questions were supplied.
    #[error("too many questions: {actual}; maximum is {max}")]
    TooManyQuestions { actual: usize, max: usize },
    /// Identifier was empty or whitespace-only.
    #[error("{kind} identifier must not be empty")]
    EmptyIdentifier { kind: &'static str },
    /// A text field was empty.
    #[error("{field} for question {question:?} must not be empty")]
    EmptyText {
        field: &'static str,
        question: String,
    },
    /// A choice question requires at least two candidates.
    #[error("choice question {question:?} requires at least two criteria")]
    TooFewChoices { question: String },
    /// A score question requires at least two levels.
    #[error("score question {question:?} requires at least two levels")]
    TooFewScoreLevels { question: String },
    /// Portable candidate limit exceeded.
    #[error("question {question:?} has {actual} candidates; maximum is {max}")]
    TooManyChoices {
        question: String,
        actual: usize,
        max: usize,
    },
}

fn validate_state(state: &Value) -> Result<(), SchemaError> {
    media::MediaEvidence::from_state(state).map_err(SchemaError::InvalidMedia)?;
    let valid = match state {
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    };

    valid.then_some(()).ok_or(SchemaError::InvalidState)
}

fn validate_identifier(kind: &'static str, value: &str) -> Result<(), SchemaError> {
    if value.trim().is_empty() {
        Err(SchemaError::EmptyIdentifier { kind })
    } else {
        Ok(())
    }
}

fn validate_nonempty(field: &'static str, question: &str, value: &str) -> Result<(), SchemaError> {
    if value.trim().is_empty() {
        Err(SchemaError::EmptyText {
            field,
            question: question.to_owned(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn boolean_alias_deserializes_to_noul() {
        let raw = json!({
            "model": "test",
            "state": "hello",
            "questions": {
                "ok": {
                    "type": "boolean",
                    "instructions": "Is this valid?"
                }
            }
        });

        let request: EvaluationRequest = serde_json::from_value(raw).unwrap();
        assert!(matches!(request.questions["ok"], Question::Noul { .. }));
    }

    #[test]
    fn rejects_empty_choice_set() {
        let request = EvaluationRequest {
            model: None,
            state: json!("hello"),
            questions: IndexMap::from([(
                "route".to_owned(),
                Question::choice("route", IndexMap::new()),
            )]),
        };

        assert!(matches!(
            request.validate(),
            Err(SchemaError::TooFewChoices { .. })
        ));
    }
}
