#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable conformance kit for TurboJev runtime adapters.
//!
//! First-party and third-party runtimes can use this crate to verify the same
//! invariants before being treated as conforming: exact answer-boundary
//! tokenization, candidate-code round trips, capability/plan consistency,
//! finite score shapes and numerical parity against a reference path.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_backend::{CandidateProjection, ExecutionPlan};
use turbojev_projection::compare_logits as compare_projection_logits;
use turbojev_prompt::RenderedMessages;
use turbojev_runtime::{
    FusedRuntimeExecutor, PromptCodec, RuntimeBatch, RuntimeCapabilities, RuntimeError,
    RuntimeOutput, TokenId,
};
use turbojev_sequence::CandidateTrie;

/// Absolute-logit tolerance for parity checks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTolerance {
    /// Maximum absolute difference between corresponding candidate scores.
    pub logit_abs: f32,
}

impl Default for RuntimeTolerance {
    fn default() -> Self {
        Self { logit_abs: 1e-4 }
    }
}

/// Exact multi-token sequence-scoring conformance report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceScoringReport {
    /// Whether runtime scores match the exact reference within tolerance.
    pub passed: bool,
    /// Number of candidate sequences compared.
    pub candidates: usize,
    /// Largest absolute candidate log-score delta.
    pub max_log_score_delta: f32,
    /// Unique candidate-prefix states required by the reference trie.
    pub unique_prefix_evaluations: usize,
    /// Candidate token steps avoided by sharing prefix states.
    pub shared_token_steps_saved: usize,
}

/// Per-task numerical parity details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTaskParity {
    /// Stable task identifier.
    pub id: String,
    /// Largest absolute score delta for the task.
    pub max_logit_delta: f32,
}

/// Runtime-level parity report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeParityReport {
    /// Whether every checked invariant and tolerance passed.
    pub passed: bool,
    /// Number of compared tasks.
    pub tasks: usize,
    /// Largest absolute score delta across all tasks.
    pub max_logit_delta: f32,
    /// Per-task details.
    pub details: Vec<RuntimeTaskParity>,
}

/// Conformance report specific to candidate-only output projection.
///
/// In addition to numerical parity, this verifies that an optimized runtime
/// reports exactly the number of candidate rows it claims to have projected.
/// This prevents a backend from computing a full vocabulary head, gathering a
/// few logits and advertising that work as a selective projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectiveProjectionReport {
    /// Whether both numerical and accounting invariants passed.
    pub passed: bool,
    /// Number of runtime tasks compared.
    pub tasks: usize,
    /// Candidate rows expected from the prepared batch.
    pub expected_candidate_rows: usize,
    /// Candidate rows reported by the optimized backend.
    pub reported_candidate_rows: usize,
    /// Largest absolute logit delta across all candidates.
    pub max_logit_delta: f32,
    /// Per-task numerical details.
    pub details: Vec<RuntimeTaskParity>,
}

/// Cross-request runtime parity summary for one fused conformance probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedRuntimeParityReport {
    /// Whether every request matched the independent reference path.
    pub passed: bool,
    /// Number of independent request roots compared.
    pub requests: usize,
    /// Largest candidate-logit delta observed across all requests/tasks.
    pub max_logit_delta: f32,
    /// Per-request runtime parity reports in input order.
    pub reports: Vec<RuntimeParityReport>,
}

/// Verify a runtime's true fused path against its ordinary independent path.
pub fn verify_fused_runtime_executor<E: FusedRuntimeExecutor>(
    executor: &E,
    batches: &[RuntimeBatch],
    plans: &[ExecutionPlan],
    tolerance: RuntimeTolerance,
) -> Result<FusedRuntimeParityReport, RuntimeConformanceError> {
    let capabilities = executor
        .fused_capabilities()
        .validate()
        .map_err(|error| RuntimeConformanceError::InvalidConfiguration(error.to_string()))?;
    if !capabilities.fused_batch {
        return Err(RuntimeConformanceError::FusedExecutionUnsupported);
    }
    if batches.len() < 2 || batches.len() != plans.len() {
        return Err(RuntimeConformanceError::InvalidConfiguration(
            "fused runtime conformance requires matching arrays with at least two requests"
                .to_owned(),
        ));
    }
    if let Some(maximum) = capabilities.max_fused_items {
        if batches.len() > maximum {
            return Err(RuntimeConformanceError::InvalidConfiguration(format!(
                "fused conformance received {} requests; runtime maximum is {}",
                batches.len(),
                maximum
            )));
        }
    }

    let runtime_capabilities = executor.capabilities();
    let mut reference = Vec::with_capacity(batches.len());
    for (batch, plan) in batches.iter().zip(plans) {
        validate_runtime_batch(runtime_capabilities, batch, *plan)?;
        reference.push(executor.score(batch, *plan)?);
    }
    let fused = executor.score_fused(batches, plans)?;
    if fused.len() != reference.len() {
        return Err(RuntimeConformanceError::FusedOutputCountMismatch {
            expected: reference.len(),
            actual: fused.len(),
        });
    }

    let mut reports = Vec::with_capacity(batches.len());
    let mut max_logit_delta = 0.0_f32;
    for ((batch, expected), actual) in batches.iter().zip(&reference).zip(&fused) {
        let report = compare_runtime_outputs(batch, expected, actual, tolerance)?;
        max_logit_delta = max_logit_delta.max(report.max_logit_delta);
        reports.push(report);
    }

    Ok(FusedRuntimeParityReport {
        passed: reports.iter().all(|report| report.passed),
        requests: reports.len(),
        max_logit_delta,
        reports,
    })
}

/// Verified candidate code at one exact answer boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedCandidateCode {
    /// Model-facing candidate symbol.
    pub symbol: String,
    /// Exact answer token sequence.
    pub token_ids: Vec<TokenId>,
}

/// Verify candidate symbols using the codec's real chat-template answer boundary.
pub fn verify_candidate_boundary<C: PromptCodec>(
    codec: &C,
    messages: &RenderedMessages,
    symbols: &[String],
    max_tokens_per_code: usize,
) -> Result<Vec<VerifiedCandidateCode>, RuntimeConformanceError> {
    if symbols.len() < 2 {
        return Err(RuntimeConformanceError::TooFewCandidates);
    }
    if max_tokens_per_code == 0 {
        return Err(RuntimeConformanceError::InvalidConfiguration(
            "max_tokens_per_code must be at least 1".to_owned(),
        ));
    }

    let base = codec.encode_prompt(messages)?;
    let mut seen_tokens = HashSet::with_capacity(symbols.len());
    let mut verified = Vec::with_capacity(symbols.len());

    for symbol in symbols {
        let full = codec.encode_prompt_with_answer(messages, symbol)?;
        if full.len() <= base.len() || !full.starts_with(&base) {
            return Err(RuntimeConformanceError::AnswerBoundaryMismatch(
                symbol.clone(),
            ));
        }
        let suffix = full[base.len()..].to_vec();
        if suffix.is_empty() || suffix.len() > max_tokens_per_code {
            return Err(RuntimeConformanceError::CandidateWidth {
                symbol: symbol.clone(),
                width: suffix.len(),
                max: max_tokens_per_code,
            });
        }
        if codec.decode(&suffix)? != *symbol {
            return Err(RuntimeConformanceError::RoundTripMismatch(symbol.clone()));
        }
        if !seen_tokens.insert(suffix.clone()) {
            return Err(RuntimeConformanceError::TokenCollision(symbol.clone()));
        }
        verified.push(VerifiedCandidateCode {
            symbol: symbol.clone(),
            token_ids: suffix,
        });
    }

    if !is_prefix_free(&verified) {
        return Err(RuntimeConformanceError::NonPrefixFreeCodebook);
    }
    Ok(verified)
}

/// Verify runtime multi-token scores against exact full-vocabulary sequence math.
///
/// `next_logits(prefix)` must return the complete next-token vocabulary logits
/// after appending `prefix` to the already prepared model prompt. The reference
/// path shares each unique candidate prefix through [`CandidateTrie`].
pub fn verify_exact_sequence_scores<F>(
    candidate_sequences: &[Vec<TokenId>],
    runtime_scores: &[f32],
    tolerance: RuntimeTolerance,
    mut next_logits: F,
) -> Result<SequenceScoringReport, RuntimeConformanceError>
where
    F: FnMut(&[TokenId]) -> Result<Vec<f32>, RuntimeError>,
{
    if !tolerance.logit_abs.is_finite() || tolerance.logit_abs < 0.0 {
        return Err(RuntimeConformanceError::InvalidTolerance);
    }
    if runtime_scores.len() != candidate_sequences.len() {
        return Err(RuntimeConformanceError::SequenceScoreCountMismatch {
            expected: candidate_sequences.len(),
            actual: runtime_scores.len(),
        });
    }
    if runtime_scores.iter().any(|score| !score.is_finite()) {
        return Err(RuntimeConformanceError::NonFiniteSequenceScore);
    }

    let trie = CandidateTrie::build(candidate_sequences)
        .map_err(|error| RuntimeConformanceError::SequenceReference(error.to_string()))?;
    let work = trie.work();
    let reference = trie
        .score_with::<RuntimeError, _>(|prefix| next_logits(prefix))
        .map_err(|error| RuntimeConformanceError::SequenceReference(error.to_string()))?;

    let max_log_score_delta = reference
        .iter()
        .zip(runtime_scores)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);

    Ok(SequenceScoringReport {
        passed: max_log_score_delta <= tolerance.logit_abs,
        candidates: candidate_sequences.len(),
        max_log_score_delta,
        unique_prefix_evaluations: work.unique_prefix_evaluations,
        shared_token_steps_saved: work.shared_token_steps_saved,
    })
}

/// Validate a prepared runtime batch against declared capabilities and plan.
pub fn validate_runtime_batch(
    capabilities: RuntimeCapabilities,
    batch: &RuntimeBatch,
    plan: ExecutionPlan,
) -> Result<(), RuntimeConformanceError> {
    if !capabilities.backend_capabilities().supports(plan) {
        return Err(RuntimeConformanceError::UnsupportedPlan);
    }
    if batch.tasks.is_empty() {
        return Err(RuntimeConformanceError::EmptyBatch);
    }

    let mut ids = HashSet::with_capacity(batch.tasks.len());
    for task in &batch.tasks {
        if !ids.insert(task.id.as_str()) {
            return Err(RuntimeConformanceError::DuplicateTask(task.id.clone()));
        }
        if task.candidates.len() < 2 {
            return Err(RuntimeConformanceError::TooFewTaskCandidates(
                task.id.clone(),
            ));
        }
        let has_multi_token = task
            .candidates
            .iter()
            .any(|candidate| candidate.token_ids.len() != 1);
        if has_multi_token && !capabilities.sequence_scores {
            return Err(RuntimeConformanceError::MissingSequenceScoring(
                task.id.clone(),
            ));
        }
        if task
            .candidates
            .iter()
            .any(|candidate| candidate.token_ids.is_empty())
        {
            return Err(RuntimeConformanceError::EmptyCandidateCode(task.id.clone()));
        }
    }
    Ok(())
}

/// Validate one runtime output against its prepared input batch.
pub fn validate_runtime_output(
    batch: &RuntimeBatch,
    output: &RuntimeOutput,
) -> Result<(), RuntimeConformanceError> {
    if output.scores.len() != batch.tasks.len() {
        return Err(RuntimeConformanceError::TaskCountMismatch {
            expected: batch.tasks.len(),
            actual: output.scores.len(),
        });
    }

    let tasks: HashMap<&str, usize> = batch
        .tasks
        .iter()
        .map(|task| (task.id.as_str(), task.candidates.len()))
        .collect();
    let mut seen = HashSet::with_capacity(output.scores.len());

    for row in &output.scores {
        if !seen.insert(row.id.as_str()) {
            return Err(RuntimeConformanceError::DuplicateScoreRow(row.id.clone()));
        }
        let expected = tasks
            .get(row.id.as_str())
            .copied()
            .ok_or_else(|| RuntimeConformanceError::UnknownScoreRow(row.id.clone()))?;
        if row.logits.len() != expected {
            return Err(RuntimeConformanceError::CandidateCountMismatch {
                id: row.id.clone(),
                expected,
                actual: row.logits.len(),
            });
        }
        if row.logits.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeConformanceError::NonFiniteScore(row.id.clone()));
        }
    }

    for task in &batch.tasks {
        if !seen.contains(task.id.as_str()) {
            return Err(RuntimeConformanceError::MissingScoreRow(task.id.clone()));
        }
    }
    Ok(())
}

/// Compare an optimized runtime output against a reference output.
pub fn compare_runtime_outputs(
    batch: &RuntimeBatch,
    reference: &RuntimeOutput,
    candidate: &RuntimeOutput,
    tolerance: RuntimeTolerance,
) -> Result<RuntimeParityReport, RuntimeConformanceError> {
    if !tolerance.logit_abs.is_finite() || tolerance.logit_abs < 0.0 {
        return Err(RuntimeConformanceError::InvalidTolerance);
    }
    validate_runtime_output(batch, reference)?;
    validate_runtime_output(batch, candidate)?;

    let reference_rows: HashMap<&str, &[f32]> = reference
        .scores
        .iter()
        .map(|row| (row.id.as_str(), row.logits.as_slice()))
        .collect();
    let candidate_rows: HashMap<&str, &[f32]> = candidate
        .scores
        .iter()
        .map(|row| (row.id.as_str(), row.logits.as_slice()))
        .collect();

    let mut details = Vec::with_capacity(batch.tasks.len());
    let mut max_logit_delta = 0.0f32;
    for task in &batch.tasks {
        let expected = reference_rows
            .get(task.id.as_str())
            .ok_or_else(|| RuntimeConformanceError::MissingScoreRow(task.id.clone()))?;
        let actual = candidate_rows
            .get(task.id.as_str())
            .ok_or_else(|| RuntimeConformanceError::MissingScoreRow(task.id.clone()))?;
        let mut task_delta = 0.0f32;
        for (left, right) in expected.iter().zip(actual.iter()) {
            task_delta = task_delta.max((left - right).abs());
        }
        max_logit_delta = max_logit_delta.max(task_delta);
        details.push(RuntimeTaskParity {
            id: task.id.clone(),
            max_logit_delta: task_delta,
        });
    }

    Ok(RuntimeParityReport {
        passed: max_logit_delta <= tolerance.logit_abs,
        tasks: batch.tasks.len(),
        max_logit_delta,
        details,
    })
}

/// Verify a true candidate-only projection against the full-vocabulary baseline.
///
/// The candidate output must contain the same logits within `tolerance`, and
/// must report one projected row for every candidate in the prepared batch.
/// Callers should use an [`ExecutionPlan`] whose projection is
/// [`CandidateProjection::CandidatesOnly`] for the optimized run.
pub fn verify_selective_projection(
    batch: &RuntimeBatch,
    reference: &RuntimeOutput,
    candidate: &RuntimeOutput,
    tolerance: RuntimeTolerance,
) -> Result<SelectiveProjectionReport, RuntimeConformanceError> {
    let parity = compare_runtime_outputs(batch, reference, candidate, tolerance)?;
    let expected_candidate_rows = batch.tasks.iter().try_fold(0usize, |total, task| {
        total
            .checked_add(task.candidates.len())
            .ok_or(RuntimeConformanceError::CandidateRowCountOverflow)
    })?;
    let reported_candidate_rows = candidate
        .metrics
        .candidate_rows_projected
        .ok_or(RuntimeConformanceError::MissingCandidateProjectionMetric)?;
    if reported_candidate_rows != expected_candidate_rows {
        return Err(RuntimeConformanceError::CandidateProjectionMetricMismatch {
            expected: expected_candidate_rows,
            actual: reported_candidate_rows,
        });
    }

    // Re-run each task through the projection-specific comparator. This keeps
    // the semantics of candidate-only parity owned by one primitive and makes
    // the worst-candidate behavior reusable by runtime adapters.
    let reference_rows: HashMap<&str, &[f32]> = reference
        .scores
        .iter()
        .map(|row| (row.id.as_str(), row.logits.as_slice()))
        .collect();
    let candidate_rows: HashMap<&str, &[f32]> = candidate
        .scores
        .iter()
        .map(|row| (row.id.as_str(), row.logits.as_slice()))
        .collect();
    for task in &batch.tasks {
        let expected = reference_rows
            .get(task.id.as_str())
            .ok_or_else(|| RuntimeConformanceError::MissingScoreRow(task.id.clone()))?;
        let actual = candidate_rows
            .get(task.id.as_str())
            .ok_or_else(|| RuntimeConformanceError::MissingScoreRow(task.id.clone()))?;
        compare_projection_logits(expected, actual, tolerance.logit_abs)
            .map_err(|error| RuntimeConformanceError::ProjectionParity(error.to_string()))?;
    }

    Ok(SelectiveProjectionReport {
        passed: parity.passed,
        tasks: parity.tasks,
        expected_candidate_rows,
        reported_candidate_rows,
        max_logit_delta: parity.max_logit_delta,
        details: parity.details,
    })
}

/// Validate that a plan is a genuine candidate-only projection plan before an
/// adapter uses [`verify_selective_projection`].
pub fn require_candidate_only_plan(plan: ExecutionPlan) -> Result<(), RuntimeConformanceError> {
    if plan.projection != CandidateProjection::CandidatesOnly {
        return Err(RuntimeConformanceError::NotCandidateOnlyPlan);
    }
    Ok(())
}

fn is_prefix_free(codes: &[VerifiedCandidateCode]) -> bool {
    for (left_index, left) in codes.iter().enumerate() {
        for (right_index, right) in codes.iter().enumerate() {
            if left_index != right_index && right.token_ids.starts_with(&left.token_ids) {
                return false;
            }
        }
    }
    true
}

/// Runtime adapter conformance failure.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum RuntimeConformanceError {
    /// Runtime returned the wrong number of sequence scores.
    #[error("runtime returned {actual} sequence scores; expected {expected}")]
    SequenceScoreCountMismatch { expected: usize, actual: usize },
    /// Runtime emitted a non-finite sequence score.
    #[error("runtime returned a non-finite sequence score")]
    NonFiniteSequenceScore,
    /// Exact sequence reference evaluation failed.
    #[error("sequence reference failed: {0}")]
    SequenceReference(String),
    /// Runtime bridge error from the codec.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// At least two candidates are required.
    #[error("conformance checks require at least two candidates")]
    TooFewCandidates,
    /// General invalid configuration.
    #[error("invalid conformance configuration: {0}")]
    InvalidConfiguration(String),
    /// Candidate answer did not extend the exact encoded prompt.
    #[error("candidate {0:?} does not extend the exact answer boundary")]
    AnswerBoundaryMismatch(String),
    /// Candidate code width is unsupported.
    #[error("candidate {symbol:?} encoded to {width} tokens; maximum is {max}")]
    CandidateWidth {
        symbol: String,
        width: usize,
        max: usize,
    },
    /// Candidate failed decode round trip.
    #[error("candidate {0:?} failed token decode round trip")]
    RoundTripMismatch(String),
    /// Two candidate symbols map to the same token sequence.
    #[error("candidate {0:?} collides with another candidate token sequence")]
    TokenCollision(String),
    /// Candidate code set is not prefix-free.
    #[error("candidate codebook is not prefix-free")]
    NonPrefixFreeCodebook,
    /// Planner requested undeclared runtime capabilities.
    #[error("runtime capabilities do not support the requested execution plan")]
    UnsupportedPlan,
    /// Runtime batch has no tasks.
    #[error("runtime batch is empty")]
    EmptyBatch,
    /// Task IDs must be unique.
    #[error("runtime batch contains duplicate task {0:?}")]
    DuplicateTask(String),
    /// A task has fewer than two candidates.
    #[error("runtime task {0:?} has fewer than two candidates")]
    TooFewTaskCandidates(String),
    /// Multi-token candidates require exact sequence scoring.
    #[error("runtime task {0:?} requires sequence scoring")]
    MissingSequenceScoring(String),
    /// Candidate code is empty.
    #[error("runtime task {0:?} contains an empty candidate code")]
    EmptyCandidateCode(String),
    /// Output row count differs from input task count.
    #[error("runtime returned {actual} rows; expected {expected}")]
    TaskCountMismatch { expected: usize, actual: usize },
    /// Runtime emitted duplicate task rows.
    #[error("runtime returned duplicate score row {0:?}")]
    DuplicateScoreRow(String),
    /// Runtime returned an unknown task row.
    #[error("runtime returned unknown score row {0:?}")]
    UnknownScoreRow(String),
    /// Runtime omitted a task row.
    #[error("runtime omitted score row {0:?}")]
    MissingScoreRow(String),
    /// Candidate count differs from prepared task.
    #[error("runtime row {id:?} has {actual} scores; expected {expected}")]
    CandidateCountMismatch {
        id: String,
        expected: usize,
        actual: usize,
    },
    /// Runtime emitted NaN or infinity.
    #[error("runtime row {0:?} contains a non-finite score")]
    NonFiniteScore(String),
    /// Candidate row-count summation overflowed.
    #[error("candidate row count overflow")]
    CandidateRowCountOverflow,
    /// Selective backend did not report its projected row count.
    #[error("candidate-only backend did not report candidate_rows_projected")]
    MissingCandidateProjectionMetric,
    /// Selective backend reported a row count inconsistent with the batch.
    #[error("candidate-only backend reported {actual} projected rows; expected {expected}")]
    CandidateProjectionMetricMismatch { expected: usize, actual: usize },
    /// Projection-specific parity primitive rejected the compared logits.
    #[error("candidate projection parity failed: {0}")]
    ProjectionParity(String),
    /// Conformance helper received a non-selective plan.
    #[error("execution plan is not a candidate-only projection plan")]
    NotCandidateOnlyPlan,
    /// Runtime does not advertise a fused cross-request path.
    #[error("runtime does not advertise fused cross-request execution")]
    FusedExecutionUnsupported,
    /// Fused runtime returned a different number of request roots.
    #[error("fused runtime returned {actual} outputs; expected {expected}")]
    FusedOutputCountMismatch { expected: usize, actual: usize },
    /// Numerical tolerance is invalid.
    #[error("runtime parity tolerance must be finite and non-negative")]
    InvalidTolerance,
}

#[cfg(test)]
mod tests {
    use turbojev_backend::{
        BackendMetrics, CandidateProjection, ExecutionPlan, FusedBackendCapabilities, PrefixReuse,
        TaskScheduling,
    };
    use turbojev_runtime::{
        FusedRuntimeExecutor, RuntimeCandidate, RuntimeExecutor, RuntimeTask, RuntimeTaskScores,
    };

    use super::*;

    struct CharacterCodec;

    impl PromptCodec for CharacterCodec {
        fn profile(&self) -> &str {
            "char"
        }

        fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError> {
            Ok(messages.user.chars().map(u32::from).collect())
        }

        fn encode_prompt_with_answer(
            &self,
            messages: &RenderedMessages,
            answer: &str,
        ) -> Result<Vec<TokenId>, RuntimeError> {
            Ok(format!("{}{answer}", messages.user)
                .chars()
                .map(u32::from)
                .collect())
        }

        fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError> {
            token_ids
                .iter()
                .map(|value| {
                    char::from_u32(*value)
                        .ok_or_else(|| RuntimeError::Codec("invalid scalar".to_owned()))
                })
                .collect()
        }
    }

    fn batch() -> RuntimeBatch {
        RuntimeBatch {
            state: serde_json::Value::Null,
            common_prefix: vec![1, 2],
            tasks: vec![RuntimeTask {
                messages: RenderedMessages::default(),
                id: "route".to_owned(),
                prompt_suffix: vec![3],
                candidates: vec![
                    RuntimeCandidate {
                        symbol: "A".to_owned(),
                        token_ids: vec![65],
                    },
                    RuntimeCandidate {
                        symbol: "B".to_owned(),
                        token_ids: vec![66],
                    },
                ],
                codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
            }],
        }
    }

    fn output(values: [f32; 2]) -> RuntimeOutput {
        RuntimeOutput {
            model: Some("fixture".to_owned()),
            scores: vec![RuntimeTaskScores {
                id: "route".to_owned(),
                logits: values.to_vec(),
            }],
            metrics: BackendMetrics::default(),
        }
    }

    #[test]
    fn verifies_exact_answer_boundary() {
        let messages = RenderedMessages {
            assistant_prefix: String::new(),
            system: None,
            user: "answer=".to_owned(),
            profile: "fixture".to_owned(),
        };
        let codes = verify_candidate_boundary(
            &CharacterCodec,
            &messages,
            &["A".to_owned(), "B".to_owned()],
            1,
        )
        .unwrap();
        assert_eq!(codes[0].token_ids, vec![65]);
    }

    #[test]
    fn validates_runtime_contract_and_numerical_parity() {
        let batch = batch();
        let capabilities = RuntimeCapabilities {
            next_token_logits: true,
            sequence_scores: false,
            shared_prefix: true,
            parallel_suffixes: true,
            selective_output_head: true,
            local_inference: true,
            image_input: false,
            audio_input: false,
            video_input: false,
        };
        let plan = ExecutionPlan {
            prefix_reuse: PrefixReuse::Shared,
            scheduling: TaskScheduling::Parallel,
            parallel_task_limit: 2,
            projection: CandidateProjection::CandidatesOnly,
        };
        validate_runtime_batch(capabilities, &batch, plan).unwrap();

        let report = compare_runtime_outputs(
            &batch,
            &output([0.0, 2.0]),
            &output([0.00001, 1.99999]),
            RuntimeTolerance { logit_abs: 0.001 },
        )
        .unwrap();
        assert!(report.passed);
    }

    #[test]
    fn selective_projection_requires_exact_row_accounting() {
        let batch = batch();
        let reference = output([0.0, 2.0]);
        let mut candidate = output([0.0, 2.0]);
        candidate.metrics.candidate_rows_projected = Some(2);
        let report = verify_selective_projection(
            &batch,
            &reference,
            &candidate,
            RuntimeTolerance { logit_abs: 0.0 },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.expected_candidate_rows, 2);
        assert_eq!(report.reported_candidate_rows, 2);
    }

    #[test]
    fn exact_sequence_conformance_uses_full_vocab_and_shared_prefixes() {
        let candidates = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let logits_for = |prefix: &[TokenId]| -> Result<Vec<f32>, RuntimeError> {
            Ok(match prefix {
                [] => vec![3.0, 0.0, 0.0, 2.0],
                [0] => vec![0.0, 2.5, 1.0, 0.0],
                [3] => vec![0.0, 1.5, 0.0, 0.0],
                _ => {
                    return Err(RuntimeError::Inference(
                        "unexpected reference prefix".to_owned(),
                    ))
                }
            })
        };
        let trie = CandidateTrie::build(&candidates).unwrap();
        let expected = trie.score_with::<RuntimeError, _>(logits_for).unwrap();
        let report = verify_exact_sequence_scores(
            &candidates,
            &expected,
            RuntimeTolerance { logit_abs: 0.0 },
            |prefix| {
                Ok(match prefix {
                    [] => vec![3.0, 0.0, 0.0, 2.0],
                    [0] => vec![0.0, 2.5, 1.0, 0.0],
                    [3] => vec![0.0, 1.5, 0.0, 0.0],
                    _ => {
                        return Err(RuntimeError::Inference(
                            "unexpected conformance prefix".to_owned(),
                        ))
                    }
                })
            },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.unique_prefix_evaluations, 3);
        assert_eq!(report.shared_token_steps_saved, 1);
    }

    #[test]
    fn selective_projection_rejects_gather_disguised_as_selective() {
        let batch = batch();
        let reference = output([0.0, 2.0]);
        let candidate = output([0.0, 2.0]);
        let error = verify_selective_projection(
            &batch,
            &reference,
            &candidate,
            RuntimeTolerance { logit_abs: 0.0 },
        )
        .unwrap_err();
        assert_eq!(
            error,
            RuntimeConformanceError::MissingCandidateProjectionMetric
        );
    }

    struct FusedFixtureExecutor;

    impl RuntimeExecutor for FusedFixtureExecutor {
        fn name(&self) -> &'static str {
            "fused-fixture"
        }

        fn capabilities(&self) -> RuntimeCapabilities {
            RuntimeCapabilities {
                next_token_logits: true,
                sequence_scores: false,
                shared_prefix: false,
                parallel_suffixes: false,
                selective_output_head: false,
                local_inference: true,
                image_input: false,
                audio_input: false,
                video_input: false,
            }
        }

        fn score(
            &self,
            batch: &RuntimeBatch,
            _plan: ExecutionPlan,
        ) -> Result<RuntimeOutput, RuntimeError> {
            Ok(RuntimeOutput {
                model: Some("fused-fixture".to_owned()),
                scores: batch
                    .tasks
                    .iter()
                    .map(|task| RuntimeTaskScores {
                        id: task.id.clone(),
                        logits: (0..task.candidates.len())
                            .map(|index| index as f32)
                            .collect(),
                    })
                    .collect(),
                metrics: BackendMetrics::default(),
            })
        }
    }

    impl FusedRuntimeExecutor for FusedFixtureExecutor {
        fn fused_capabilities(&self) -> FusedBackendCapabilities {
            FusedBackendCapabilities {
                fused_batch: true,
                max_fused_items: Some(8),
                estimated_fused_item_bytes: Some(1024),
            }
        }

        fn score_fused(
            &self,
            batches: &[RuntimeBatch],
            plans: &[ExecutionPlan],
        ) -> Result<Vec<RuntimeOutput>, RuntimeError> {
            if batches.len() != plans.len() {
                return Err(RuntimeError::Inference("batch/plan mismatch".to_owned()));
            }
            batches
                .iter()
                .zip(plans)
                .map(|(batch, plan)| self.score(batch, *plan))
                .collect()
        }
    }

    #[test]
    fn fused_runtime_conformance_matches_independent_reference() {
        let batches = vec![batch(), batch()];
        let plans = vec![ExecutionPlan::default(), ExecutionPlan::default()];
        let report = verify_fused_runtime_executor(
            &FusedFixtureExecutor,
            &batches,
            &plans,
            RuntimeTolerance::default(),
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.requests, 2);
        assert_eq!(report.max_logit_delta, 0.0);
    }
}
