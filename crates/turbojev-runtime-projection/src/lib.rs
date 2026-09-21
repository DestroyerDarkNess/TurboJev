#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable candidate-only projection runtime.
//!
//! This crate is the reference implementation of TurboJev's selective output
//! head contract. It is deliberately model-format and operating-system neutral.
//! A concrete model runtime supplies the final post-normalization hidden state
//! and the requested output-head rows; this crate performs only the candidate
//! matrix multiplication and returns raw logits to TurboJev core.
//!
//! Device-native runtimes are free to implement the same contract without
//! copying rows to CPU. They must still pass the same conformance checks.

use thiserror::Error;
use turbojev_backend::{
    BackendMetrics, BackendResourceHints, CandidateProjection, ExecutionPlan, PrefixReuse,
    TaskScheduling,
};
use turbojev_projection::{
    project_candidates_f32, DenseCandidateRows, ProjectionConfig, ProjectionError,
};
use turbojev_runtime::{
    RuntimeBatch, RuntimeCapabilities, RuntimeError, RuntimeExecutor, RuntimeOutput,
    RuntimeTaskScores, TokenId,
};

/// Low-level model/runtime data needed for a dense selective projection.
///
/// The full-vocabulary method exists only as a correctness baseline. The
/// candidate-only path never calls it.
pub trait DenseProjectionSource {
    /// Stable runtime/model identifier.
    fn model_id(&self) -> &str;

    /// Whether model execution is local to the host process/device.
    fn local_inference(&self) -> bool {
        true
    }

    /// Portable scheduling/resource hints.
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints::default()
    }

    /// Ordinary full-vocabulary reference path, returning only gathered
    /// candidate logits after the runtime has computed its normal vocabulary
    /// head. Candidate IDs are guaranteed to be one token each.
    fn full_vocabulary_candidate_logits(
        &self,
        prompt_tokens: &[TokenId],
        candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, ProjectionRuntimeError>;

    /// Final post-normalization hidden state at the decision position, before
    /// the ordinary LM output head.
    fn final_hidden_state(
        &self,
        prompt_tokens: &[TokenId],
    ) -> Result<Vec<f32>, ProjectionRuntimeError>;

    /// Fetch only output-head rows corresponding to the requested token IDs.
    /// Rows must be returned in exactly the same order, including duplicates.
    fn candidate_output_rows(
        &self,
        candidate_token_ids: &[TokenId],
    ) -> Result<DenseCandidateRows, ProjectionRuntimeError>;

    /// Optional model-specific scalar applied to output-head dot products.
    fn projection_config(&self) -> ProjectionConfig {
        ProjectionConfig::default()
    }
}

/// Runtime executor that owns the full-vocabulary/selective switch while
/// delegating model-specific hidden-state and row access to a source.
pub struct DenseProjectionExecutor<S> {
    source: S,
}

impl<S> DenseProjectionExecutor<S> {
    /// Wrap a loaded model/runtime source.
    pub const fn new(source: S) -> Self {
        Self { source }
    }

    /// Borrow the underlying model/runtime source.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Recover the source.
    pub fn into_source(self) -> S {
        self.source
    }
}

impl<S: DenseProjectionSource> RuntimeExecutor for DenseProjectionExecutor<S> {
    fn name(&self) -> &'static str {
        "dense-candidate-projection"
    }

    fn capabilities(&self) -> RuntimeCapabilities {
        RuntimeCapabilities {
            next_token_logits: true,
            sequence_scores: false,
            shared_prefix: false,
            parallel_suffixes: false,
            selective_output_head: true,
            local_inference: self.source.local_inference(),
        }
    }

    fn resource_hints(&self) -> BackendResourceHints {
        self.source.resource_hints()
    }

    fn score(
        &self,
        batch: &RuntimeBatch,
        plan: ExecutionPlan,
    ) -> Result<RuntimeOutput, RuntimeError> {
        validate_plan(plan)?;
        if !batch.common_prefix.is_empty() {
            return Err(RuntimeError::Unsupported(
                "dense projection reference executor does not own shared-prefix state".to_owned(),
            ));
        }

        let selective = plan.projection == CandidateProjection::CandidatesOnly;
        let mut scores = Vec::with_capacity(batch.tasks.len());
        let mut input_tokens = 0usize;
        let mut projected_rows = 0usize;

        for task in &batch.tasks {
            let candidate_ids = single_token_ids(task)?;
            input_tokens = input_tokens.saturating_add(task.prompt_suffix.len());
            let logits = if selective {
                let hidden = self
                    .source
                    .final_hidden_state(&task.prompt_suffix)
                    .map_err(map_source_error)?;
                let rows = self
                    .source
                    .candidate_output_rows(&candidate_ids)
                    .map_err(map_source_error)?;
                if rows.candidate_count() != candidate_ids.len() {
                    return Err(RuntimeError::Inference(format!(
                        "candidate row source returned {} rows for {} candidates in task {:?}",
                        rows.candidate_count(),
                        candidate_ids.len(),
                        task.id
                    )));
                }
                projected_rows = projected_rows.saturating_add(rows.candidate_count());
                project_candidates_f32(&hidden, &rows, self.source.projection_config())
                    .map_err(map_projection_error)?
            } else {
                self.source
                    .full_vocabulary_candidate_logits(&task.prompt_suffix, &candidate_ids)
                    .map_err(map_source_error)?
            };
            validate_logits(&task.id, candidate_ids.len(), &logits)?;
            scores.push(RuntimeTaskScores {
                id: task.id.clone(),
                logits,
            });
        }

        Ok(RuntimeOutput {
            model: Some(self.source.model_id().to_owned()),
            scores,
            metrics: BackendMetrics {
                input_tokens: Some(input_tokens),
                candidate_rows_projected: selective.then_some(projected_rows),
                ..BackendMetrics::default()
            },
        })
    }
}

fn validate_plan(plan: ExecutionPlan) -> Result<(), RuntimeError> {
    if !plan.scheduling_is_valid() {
        return Err(RuntimeError::Preparation(
            "invalid scheduling/parallel_task_limit combination".to_owned(),
        ));
    }
    if plan.prefix_reuse != PrefixReuse::None {
        return Err(RuntimeError::Unsupported(
            "dense projection reference executor does not implement shared-prefix reuse".to_owned(),
        ));
    }
    if plan.scheduling != TaskScheduling::Serial {
        return Err(RuntimeError::Unsupported(
            "dense projection reference executor is serial".to_owned(),
        ));
    }
    Ok(())
}

fn single_token_ids(task: &turbojev_runtime::RuntimeTask) -> Result<Vec<TokenId>, RuntimeError> {
    let mut result = Vec::with_capacity(task.candidates.len());
    for candidate in &task.candidates {
        if candidate.token_ids.len() != 1 {
            return Err(RuntimeError::Unsupported(format!(
                "task {:?} uses a multi-token candidate; dense output-row projection is next-token only",
                task.id
            )));
        }
        result.push(candidate.token_ids[0]);
    }
    Ok(result)
}

fn validate_logits(task_id: &str, expected: usize, logits: &[f32]) -> Result<(), RuntimeError> {
    if logits.len() != expected {
        return Err(RuntimeError::Inference(format!(
            "task {task_id:?} returned {} logits for {expected} candidates",
            logits.len()
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Inference(format!(
            "task {task_id:?} returned non-finite logits"
        )));
    }
    Ok(())
}

fn map_projection_error(error: ProjectionError) -> RuntimeError {
    RuntimeError::Inference(error.to_string())
}

fn map_source_error(error: ProjectionRuntimeError) -> RuntimeError {
    match error {
        ProjectionRuntimeError::Unsupported(message) => RuntimeError::Unsupported(message),
        ProjectionRuntimeError::InvalidInput(message) => RuntimeError::Preparation(message),
        ProjectionRuntimeError::Inference(message) => RuntimeError::Inference(message),
    }
}

/// Stable error category implemented by candidate-row model sources.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ProjectionRuntimeError {
    /// Loaded model/runtime cannot expose a requested primitive.
    #[error("unsupported projection source operation: {0}")]
    Unsupported(String),
    /// Prompt/candidate/model input was invalid.
    #[error("invalid projection source input: {0}")]
    InvalidInput(String),
    /// Model/runtime execution failed.
    #[error("projection source inference failed: {0}")]
    Inference(String),
}

#[cfg(test)]
mod tests {
    use turbojev_backend::CandidateProjection;
    use turbojev_projection::{gather_candidate_rows_f32, project_full_vocab_f32};
    use turbojev_runtime::{RuntimeCandidate, RuntimeTask};
    use turbojev_runtime_conformance::{
        verify_selective_projection, RuntimeTolerance,
    };
    use turbojev_tokenizer::CodebookKind;

    use super::*;

    #[derive(Clone)]
    struct FixtureSource {
        hidden: Vec<f32>,
        matrix: Vec<f32>,
        bias: Vec<f32>,
    }

    impl FixtureSource {
        fn new() -> Self {
            Self {
                hidden: vec![0.5, -1.25, 2.0, 0.75],
                matrix: vec![
                    0.1, 0.2, 0.3, 0.4,
                    -0.5, 0.6, -0.7, 0.8,
                    1.0, 0.0, 0.25, -0.5,
                    0.33, -0.2, 0.75, 1.1,
                    -1.0, 0.5, 0.5, -0.25,
                ],
                bias: vec![0.0, 0.1, -0.2, 0.3, -0.4],
            }
        }
    }

    impl DenseProjectionSource for FixtureSource {
        fn model_id(&self) -> &str {
            "fixture-dense-head"
        }

        fn full_vocabulary_candidate_logits(
            &self,
            _prompt_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, ProjectionRuntimeError> {
            let full = project_full_vocab_f32(
                &self.hidden,
                5,
                &self.matrix,
                Some(&self.bias),
                ProjectionConfig::default(),
            )
            .map_err(|error| ProjectionRuntimeError::Inference(error.to_string()))?;
            candidate_token_ids
                .iter()
                .map(|id| {
                    full.get(*id as usize).copied().ok_or_else(|| {
                        ProjectionRuntimeError::InvalidInput(format!(
                            "candidate token {id} outside fixture vocabulary"
                        ))
                    })
                })
                .collect()
        }

        fn final_hidden_state(
            &self,
            _prompt_tokens: &[TokenId],
        ) -> Result<Vec<f32>, ProjectionRuntimeError> {
            Ok(self.hidden.clone())
        }

        fn candidate_output_rows(
            &self,
            candidate_token_ids: &[TokenId],
        ) -> Result<DenseCandidateRows, ProjectionRuntimeError> {
            gather_candidate_rows_f32(
                5,
                4,
                &self.matrix,
                Some(&self.bias),
                candidate_token_ids,
            )
            .map_err(|error| ProjectionRuntimeError::Inference(error.to_string()))
        }
    }

    fn batch() -> RuntimeBatch {
        RuntimeBatch {
            common_prefix: Vec::new(),
            tasks: vec![RuntimeTask {
                id: "route".to_owned(),
                prompt_suffix: vec![10, 11, 12],
                candidates: vec![
                    RuntimeCandidate { symbol: "A".to_owned(), token_ids: vec![4] },
                    RuntimeCandidate { symbol: "B".to_owned(), token_ids: vec![1] },
                    RuntimeCandidate { symbol: "C".to_owned(), token_ids: vec![3] },
                ],
                codebook_kind: CodebookKind::SingleToken,
            }],
        }
    }

    fn plan(projection: CandidateProjection) -> ExecutionPlan {
        ExecutionPlan {
            prefix_reuse: PrefixReuse::None,
            scheduling: TaskScheduling::Serial,
            parallel_task_limit: 1,
            projection,
        }
    }

    #[test]
    fn selective_path_matches_full_vocabulary_reference_exactly() {
        let executor = DenseProjectionExecutor::new(FixtureSource::new());
        let batch = batch();
        let reference = executor
            .score(&batch, plan(CandidateProjection::FullVocabulary))
            .unwrap();
        let selected = executor
            .score(&batch, plan(CandidateProjection::CandidatesOnly))
            .unwrap();
        let report = verify_selective_projection(
            &batch,
            &reference,
            &selected,
            RuntimeTolerance { logit_abs: 0.0 },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.expected_candidate_rows, 3);
        assert_eq!(selected.metrics.candidate_rows_projected, Some(3));
    }

    #[test]
    fn selective_capability_is_truthful_and_full_path_does_not_claim_rows() {
        let executor = DenseProjectionExecutor::new(FixtureSource::new());
        assert!(executor.capabilities().selective_output_head);
        let full = executor
            .score(&batch(), plan(CandidateProjection::FullVocabulary))
            .unwrap();
        assert_eq!(full.metrics.candidate_rows_projected, None);
    }
}
