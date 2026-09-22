#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Reproducible benchmark/fixture protocol shared by all TurboJev runtimes.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_backend::ExecutionPlan;
use turbojev_schema::EvaluationRequest;

/// Benchmark file contract.
pub const BENCHMARK_CONTRACT: &str = "turbojev.benchmark.v1";

/// One semantic decision case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkCase {
    pub id: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub request: EvaluationRequest,
}

/// Portable benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub contract: String,
    pub id: String,
    pub cases: Vec<BenchmarkCase>,
}

impl BenchmarkSuite {
    /// Validate the suite before it is used for a backend comparison.
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.contract != BENCHMARK_CONTRACT {
            return Err(BenchmarkError::Contract(self.contract.clone()));
        }
        if self.id.trim().is_empty() {
            return Err(BenchmarkError::EmptySuiteId);
        }
        if self.cases.is_empty() {
            return Err(BenchmarkError::EmptySuite);
        }
        let mut ids = HashSet::with_capacity(self.cases.len());
        for case in &self.cases {
            if case.id.trim().is_empty() {
                return Err(BenchmarkError::EmptyCaseId);
            }
            if !ids.insert(case.id.as_str()) {
                return Err(BenchmarkError::DuplicateCase(case.id.clone()));
            }
            case.request
                .validate()
                .map_err(|error| BenchmarkError::Request(case.id.clone(), error.to_string()))?;
        }
        Ok(())
    }
}

/// Exact environment metadata attached to one benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRunMetadata {
    pub suite: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    pub backend: String,
    pub prompt_profile: String,
    pub tokenizer_profile: String,
    pub plan: ExecutionPlan,
    pub warmup_runs: u32,
    pub repetitions: u32,
    /// Opaque host description supplied by the benchmark runner. The protocol
    /// does not inspect OS/device details itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
}

impl BenchmarkRunMetadata {
    /// Validate required reproducibility fields.
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        for (field, value) in [
            ("suite", self.suite.as_str()),
            ("model", self.model.as_str()),
            ("backend", self.backend.as_str()),
            ("prompt_profile", self.prompt_profile.as_str()),
            ("tokenizer_profile", self.tokenizer_profile.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BenchmarkError::EmptyMetadata(field));
            }
        }
        if self.repetitions == 0 {
            return Err(BenchmarkError::ZeroRepetitions);
        }
        Ok(())
    }
}

/// One measured case repetition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSample {
    pub case_id: String,
    pub repetition: u32,
    pub total_seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<usize>,
    /// Direct TurboJev inference emits no model-generated output tokens.
    pub generated_tokens: u64,
    /// Stable hash/fingerprint supplied by the runner for semantic parity.
    pub answer_fingerprint: String,
}

/// Aggregate timing summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingSummary {
    pub samples: usize,
    pub min_seconds: f64,
    pub median_seconds: f64,
    pub p95_seconds: f64,
    pub max_seconds: f64,
    pub mean_seconds: f64,
}

/// Complete machine-readable benchmark report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub contract: String,
    pub metadata: BenchmarkRunMetadata,
    pub samples: Vec<BenchmarkSample>,
    pub timing: TimingSummary,
}

impl BenchmarkReport {
    /// Build a report and validate direct-inference invariants.
    pub fn new(
        metadata: BenchmarkRunMetadata,
        samples: Vec<BenchmarkSample>,
    ) -> Result<Self, BenchmarkError> {
        metadata.validate()?;
        if samples.is_empty() {
            return Err(BenchmarkError::NoSamples);
        }
        for sample in &samples {
            if sample.case_id.trim().is_empty()
                || sample.answer_fingerprint.trim().is_empty()
                || sample.repetition >= metadata.repetitions
                || !valid_seconds(sample.total_seconds)
                || sample
                    .prefill_seconds
                    .is_some_and(|value| !valid_seconds(value))
                || sample
                    .scoring_seconds
                    .is_some_and(|value| !valid_seconds(value))
                || sample.generated_tokens != 0
            {
                return Err(BenchmarkError::InvalidSample);
            }
        }
        let timing = summarize(samples.iter().map(|sample| sample.total_seconds))?;
        Ok(Self {
            contract: BENCHMARK_CONTRACT.to_owned(),
            metadata,
            samples,
            timing,
        })
    }
}

/// Summarize a finite non-empty latency series.
pub fn summarize(values: impl IntoIterator<Item = f64>) -> Result<TimingSummary, BenchmarkError> {
    let mut values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return Err(BenchmarkError::NoSamples);
    }
    if values.iter().any(|value| !valid_seconds(*value)) {
        return Err(BenchmarkError::InvalidSample);
    }
    values.sort_by(f64::total_cmp);
    let sum: f64 = values.iter().sum();
    Ok(TimingSummary {
        samples: values.len(),
        min_seconds: values[0],
        median_seconds: percentile_sorted(&values, 0.50),
        p95_seconds: percentile_sorted(&values, 0.95),
        max_seconds: values[values.len() - 1],
        mean_seconds: sum / values.len() as f64,
    })
}

fn valid_seconds(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn percentile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BenchmarkError {
    #[error("unsupported benchmark contract {0:?}")]
    Contract(String),
    #[error("benchmark suite id is empty")]
    EmptySuiteId,
    #[error("benchmark suite contains no cases")]
    EmptySuite,
    #[error("benchmark case id is empty")]
    EmptyCaseId,
    #[error("duplicate benchmark case {0:?}")]
    DuplicateCase(String),
    #[error("benchmark case {0:?} has invalid request: {1}")]
    Request(String, String),
    #[error("benchmark metadata field {0:?} is empty")]
    EmptyMetadata(&'static str),
    #[error("benchmark repetitions must be at least one")]
    ZeroRepetitions,
    #[error("benchmark report has no samples")]
    NoSamples,
    #[error("benchmark sample is invalid or reports generated tokens")]
    InvalidSample,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_summary_is_deterministic() {
        let summary = summarize([0.4, 0.1, 0.2, 0.3, 0.5]).unwrap();
        assert_eq!(summary.min_seconds, 0.1);
        assert_eq!(summary.median_seconds, 0.3);
        assert_eq!(summary.p95_seconds, 0.5);
        assert_eq!(summary.max_seconds, 0.5);
    }

    #[test]
    fn metadata_requires_repetitions_and_identity() {
        let metadata = BenchmarkRunMetadata {
            suite: "suite".into(),
            model: "model".into(),
            revision: None,
            quantization: None,
            backend: "backend".into(),
            prompt_profile: "prompt".into(),
            tokenizer_profile: "tokenizer".into(),
            plan: ExecutionPlan::default(),
            warmup_runs: 0,
            repetitions: 0,
            host: None,
        };
        assert_eq!(
            metadata.validate().unwrap_err(),
            BenchmarkError::ZeroRepetitions
        );
    }

    #[test]
    fn generated_tokens_must_remain_zero() {
        let metadata = BenchmarkRunMetadata {
            suite: "x".into(),
            model: "m".into(),
            revision: None,
            quantization: None,
            backend: "b".into(),
            prompt_profile: "p".into(),
            tokenizer_profile: "t".into(),
            plan: ExecutionPlan::default(),
            warmup_runs: 1,
            repetitions: 1,
            host: None,
        };
        let error = BenchmarkReport::new(
            metadata,
            vec![BenchmarkSample {
                case_id: "x".into(),
                repetition: 0,
                total_seconds: 1.0,
                prefill_seconds: None,
                scoring_seconds: Some(1.0),
                input_tokens: Some(12),
                generated_tokens: 1,
                answer_fingerprint: "bad".into(),
            }],
        )
        .unwrap_err();
        assert_eq!(error, BenchmarkError::InvalidSample);
    }
}
