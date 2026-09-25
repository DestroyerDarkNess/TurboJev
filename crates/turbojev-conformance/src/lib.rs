#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Backend-independent conformance and numerical parity checks.
//!
//! This crate compares completed typed responses. It therefore works for every
//! runtime, model format and platform without knowing how inference happened.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_schema::{Answer, EvaluationResponse};

/// Numerical tolerances used when comparing a reference path with an optimized path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    /// Maximum absolute difference accepted for candidate probabilities.
    pub probability_abs: f64,
    /// Maximum absolute difference accepted for expected score values.
    pub score_abs: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            probability_abs: 1e-6,
            score_abs: 1e-6,
        }
    }
}

/// Per-question parity information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuestionParity {
    /// Stable question identifier.
    pub id: String,
    /// Whether both paths selected the same discrete outcome.
    pub decision_agrees: bool,
    /// Largest absolute probability difference found for this question.
    pub max_probability_delta: f64,
    /// Absolute score difference for score questions.
    pub score_delta: Option<f64>,
}

/// Aggregate comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    /// True when all semantic and numerical checks satisfy the tolerance.
    pub passed: bool,
    /// Number of compared questions.
    pub questions: usize,
    /// Number whose discrete decision agrees.
    pub decision_agreements: usize,
    /// Maximum candidate probability delta across the request.
    pub max_probability_delta: f64,
    /// Maximum expected-score delta across score questions.
    pub max_score_delta: f64,
    /// Ordered per-question details.
    pub details: Vec<QuestionParity>,
}

/// Compare a reference response against an optimized/runtime-specific response.
pub fn compare_responses(
    reference: &EvaluationResponse,
    candidate: &EvaluationResponse,
    tolerance: Tolerance,
) -> Result<ParityReport, ConformanceError> {
    validate_tolerance(tolerance)?;
    if reference.answers.len() != candidate.answers.len() {
        return Err(ConformanceError::AnswerCountMismatch {
            reference: reference.answers.len(),
            candidate: candidate.answers.len(),
        });
    }

    let mut details = Vec::with_capacity(reference.answers.len());
    let mut agreements = 0usize;
    let mut max_probability_delta = 0.0f64;
    let mut max_score_delta = 0.0f64;

    for (id, expected) in &reference.answers {
        let actual = candidate
            .answers
            .get(id)
            .ok_or_else(|| ConformanceError::MissingAnswer(id.clone()))?;
        let parity = compare_answer(id, expected, actual)?;
        if parity.decision_agrees {
            agreements += 1;
        }
        max_probability_delta = max_probability_delta.max(parity.max_probability_delta);
        if let Some(delta) = parity.score_delta {
            max_score_delta = max_score_delta.max(delta);
        }
        details.push(parity);
    }

    let passed = agreements == reference.answers.len()
        && max_probability_delta <= tolerance.probability_abs
        && max_score_delta <= tolerance.score_abs;

    Ok(ParityReport {
        passed,
        questions: reference.answers.len(),
        decision_agreements: agreements,
        max_probability_delta,
        max_score_delta,
        details,
    })
}

/// Validate distributions and score ranges in one response without comparing it
/// to another backend.
pub fn validate_response(response: &EvaluationResponse) -> Result<(), ConformanceError> {
    for (id, answer) in &response.answers {
        match answer {
            Answer::Choice {
                choice,
                probabilities,
                ..
            } => {
                validate_distribution(id, probabilities)?;
                if !probabilities.contains_key(choice) {
                    return Err(ConformanceError::SelectedChoiceMissing(id.clone()));
                }
            }
            Answer::Noul { noul } => validate_probability(id, *noul)?,
            Answer::Score {
                score,
                probabilities,
                legend,
                ..
            } => {
                validate_distribution(id, probabilities)?;
                if probabilities.len() != legend.len() {
                    return Err(ConformanceError::LegendMismatch(id.clone()));
                }
                if !score.is_finite() || *score < 0.0 || *score > (probabilities.len() - 1) as f64 {
                    return Err(ConformanceError::InvalidScore(id.clone()));
                }
            }
        }
    }
    Ok(())
}

fn compare_answer(
    id: &str,
    expected: &Answer,
    actual: &Answer,
) -> Result<QuestionParity, ConformanceError> {
    match (expected, actual) {
        (
            Answer::Choice {
                choice: a,
                probabilities: pa,
                ..
            },
            Answer::Choice {
                choice: b,
                probabilities: pb,
                ..
            },
        ) => Ok(QuestionParity {
            id: id.to_owned(),
            decision_agrees: a == b,
            max_probability_delta: distribution_delta(id, pa, pb)?,
            score_delta: None,
        }),
        (Answer::Noul { noul: a }, Answer::Noul { noul: b }) => Ok(QuestionParity {
            id: id.to_owned(),
            decision_agrees: (*a >= 0.5) == (*b >= 0.5),
            max_probability_delta: (a - b).abs(),
            score_delta: None,
        }),
        (
            Answer::Score {
                score: a,
                probabilities: pa,
                ..
            },
            Answer::Score {
                score: b,
                probabilities: pb,
                ..
            },
        ) => {
            let winner_a = distribution_argmax(pa)
                .ok_or_else(|| ConformanceError::EmptyDistribution(id.to_owned()))?;
            let winner_b = distribution_argmax(pb)
                .ok_or_else(|| ConformanceError::EmptyDistribution(id.to_owned()))?;
            Ok(QuestionParity {
                id: id.to_owned(),
                decision_agrees: winner_a == winner_b,
                max_probability_delta: distribution_delta(id, pa, pb)?,
                score_delta: Some((a - b).abs()),
            })
        }
        _ => Err(ConformanceError::AnswerTypeMismatch(id.to_owned())),
    }
}

fn distribution_delta(
    id: &str,
    a: &IndexMap<String, f64>,
    b: &IndexMap<String, f64>,
) -> Result<f64, ConformanceError> {
    if a.len() != b.len() || a.keys().any(|key| !b.contains_key(key)) {
        return Err(ConformanceError::CandidateSetMismatch(id.to_owned()));
    }
    let mut delta = 0.0f64;
    for (key, av) in a {
        let bv = b
            .get(key)
            .ok_or_else(|| ConformanceError::CandidateSetMismatch(id.to_owned()))?;
        validate_probability(id, *av)?;
        validate_probability(id, *bv)?;
        delta = delta.max((av - bv).abs());
    }
    Ok(delta)
}

fn validate_distribution(
    id: &str,
    probabilities: &IndexMap<String, f64>,
) -> Result<(), ConformanceError> {
    if probabilities.len() < 2 {
        return Err(ConformanceError::EmptyDistribution(id.to_owned()));
    }
    let mut total = 0.0;
    for value in probabilities.values().copied() {
        validate_probability(id, value)?;
        total += value;
    }
    if (total - 1.0).abs() > 1e-6 {
        return Err(ConformanceError::DistributionNotNormalized {
            id: id.to_owned(),
            total,
        });
    }
    Ok(())
}

fn validate_probability(id: &str, value: f64) -> Result<(), ConformanceError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(ConformanceError::InvalidProbability {
            id: id.to_owned(),
            value,
        })
    } else {
        Ok(())
    }
}

fn distribution_argmax(values: &IndexMap<String, f64>) -> Option<&str> {
    values
        .iter()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(key, _)| key.as_str())
}

fn validate_tolerance(tolerance: Tolerance) -> Result<(), ConformanceError> {
    if !tolerance.probability_abs.is_finite()
        || tolerance.probability_abs < 0.0
        || !tolerance.score_abs.is_finite()
        || tolerance.score_abs < 0.0
    {
        Err(ConformanceError::InvalidTolerance)
    } else {
        Ok(())
    }
}

/// Conformance failure.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConformanceError {
    /// Comparison tolerances must be finite and non-negative.
    #[error("conformance tolerances must be finite and non-negative")]
    InvalidTolerance,
    /// Response answer counts differ.
    #[error("candidate response has {candidate} answers; reference has {reference}")]
    AnswerCountMismatch { reference: usize, candidate: usize },
    /// Candidate response omitted one answer.
    #[error("candidate response omitted answer {0:?}")]
    MissingAnswer(String),
    /// Answer variants differ.
    #[error("answer type mismatch for question {0:?}")]
    AnswerTypeMismatch(String),
    /// Candidate keys differ.
    #[error("candidate set mismatch for question {0:?}")]
    CandidateSetMismatch(String),
    /// Distribution is missing/too small.
    #[error("question {0:?} has an invalid/empty distribution")]
    EmptyDistribution(String),
    /// Probability is invalid.
    #[error("question {id:?} has invalid probability {value}")]
    InvalidProbability { id: String, value: f64 },
    /// Distribution does not sum to one.
    #[error("question {id:?} probabilities sum to {total}, expected 1")]
    DistributionNotNormalized { id: String, total: f64 },
    /// Selected choice is not in its distribution.
    #[error("selected choice for question {0:?} is missing from its probability distribution")]
    SelectedChoiceMissing(String),
    /// Score legend and distribution lengths differ.
    #[error("score legend does not match probability levels for question {0:?}")]
    LegendMismatch(String),
    /// Score is outside the supplied rubric range.
    #[error("score is outside the allowed rubric range for question {0:?}")]
    InvalidScore(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::indexmap;
    use turbojev_schema::{EngineMetadata, ExecutionMetadata, Usage};

    fn response(choice: &str, p: f64) -> EvaluationResponse {
        EvaluationResponse {
            model: None,
            answers: indexmap! {
                "route".into() => Answer::Choice {
                    choice: choice.into(), confidence: p,
                    probabilities: indexmap!{"a".into()=>1.0-p,"b".into()=>p},
                }
            },
            usage: Usage {
                decisions: 1,
                input_tokens: None,
                generated_tokens: 0,
            },
            metadata: EngineMetadata {
                backend: "test".into(),
                calibration: "softmax".into(),
                calibrated: false,
                confidence_method: "selected-candidate-probability".into(),
                contract: "v".into(),
                order_robustness: "none".into(),
                order_evaluations: 0,
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
    fn catches_numerical_drift_but_reports_decision_agreement() {
        let a = response("b", 0.9);
        let b = response("b", 0.88);
        let report = compare_responses(
            &a,
            &b,
            Tolerance {
                probability_abs: 0.01,
                score_abs: 1e-6,
            },
        )
        .unwrap();
        assert_eq!(report.decision_agreements, 1);
        assert!(!report.passed);
    }
}
