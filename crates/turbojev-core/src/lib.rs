#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! TurboJev's portable, backend-neutral decision engine.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use serde_json::Value;
use thiserror::Error;
use turbojev_backend::{
    AsyncScoringBackend, BackendError, BackendOutput, Candidate, DecisionBatch, DecisionIntent,
    DecisionTask, ExecutionPlan, FusedScoringBackend, ScoringBackend, TaskScores,
};
use turbojev_calibration::{CalibrationError, Calibrator, SoftmaxCalibrator};
use turbojev_engine::{ExecutionBudget, ExecutionPlanner, PlanningPolicy};
use turbojev_schema::{
    Answer, EngineMetadata, EvaluationRequest, EvaluationResponse, ExecutionMetadata, Question,
    SchemaError, Usage,
};

/// Stable native core contract identifier.
pub const CORE_CONTRACT: &str = "turbojev.core.v4";

/// Convenience result returned by [`TurboJevEngine::classify`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    /// Selected option.
    pub choice: String,
    /// Conditional distribution across supplied options.
    pub probabilities: IndexMap<String, f64>,
}

/// Backend-neutral TurboJev engine.
///
/// The engine owns schema validation, execution planning, probability
/// normalization/calibration and typed output construction. Backends only
/// implement model-facing inference and return ordered candidate logits.
pub struct TurboJevEngine<B, C = SoftmaxCalibrator> {
    backend: B,
    calibrator: C,
    planner: ExecutionPlanner,
}

impl<B> TurboJevEngine<B, SoftmaxCalibrator>
where
    B: ScoringBackend,
{
    /// Create an engine using uncalibrated softmax and automatic planning.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            calibrator: SoftmaxCalibrator,
            planner: ExecutionPlanner::default(),
        }
    }
}

impl<B, C> TurboJevEngine<B, C>
where
    B: ScoringBackend,
    C: Calibrator,
{
    /// Create an engine with an explicit calibrator and automatic planning.
    pub fn with_calibrator(backend: B, calibrator: C) -> Self {
        Self {
            backend,
            calibrator,
            planner: ExecutionPlanner::default(),
        }
    }

    /// Override execution planning policy without changing the backend.
    pub fn with_planning_policy(mut self, policy: PlanningPolicy) -> Self {
        self.planner = ExecutionPlanner::with_budget(policy, self.planner.budget());
        self
    }

    /// Apply a portable resource budget to automatic execution planning.
    pub fn with_execution_budget(mut self, budget: ExecutionBudget) -> Self {
        self.planner = self.planner.set_budget(budget);
        self
    }

    /// Access the configured inference backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Evaluate a full typed decision request.
    pub fn evaluate(&self, request: EvaluationRequest) -> Result<EvaluationResponse, Error> {
        request.validate()?;
        let batch = compile_decision_batch(&request);
        let (backend_output, indexed_scores, plan) = self.score_batch(&batch)?;
        build_evaluation_response(
            &request,
            &batch,
            backend_output,
            &indexed_scores,
            plan,
            self.backend.name(),
            &self.calibrator,
        )
    }

    /// Ergonomic single-choice helper similar to classifier SDKs.
    ///
    /// This helper carries a semantic `BestMatch` intent rather than injecting
    /// an English instruction. The selected model adapter owns the human-language
    /// rendering for that intent.
    pub fn classify<I, S>(
        &self,
        state: impl Into<String>,
        options: I,
    ) -> Result<ClassificationResult, Error>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let options: Vec<String> = options.into_iter().map(Into::into).collect();
        if options.len() < 2 {
            return Err(Error::InvalidClassification(
                "classification requires at least two options".to_owned(),
            ));
        }
        if options.iter().any(|option| option.trim().is_empty()) {
            return Err(Error::InvalidClassification(
                "classification options must not be empty".to_owned(),
            ));
        }
        let unique: HashSet<&str> = options.iter().map(String::as_str).collect();
        if unique.len() != options.len() {
            return Err(Error::InvalidClassification(
                "classification options must be unique".to_owned(),
            ));
        }

        let candidates: Vec<Candidate> = options
            .into_iter()
            .map(|option| Candidate {
                key: option.clone(),
                description: option,
            })
            .collect();
        let task = DecisionTask {
            id: "classification".to_owned(),
            intent: DecisionIntent::BestMatch,
            instruction: None,
            candidates,
        };
        let batch = DecisionBatch {
            state: Value::String(state.into()),
            tasks: vec![task.clone()],
        };

        let (_backend_output, indexed_scores, _plan) = self.score_batch(&batch)?;
        let logits = indexed_scores
            .get(&task.id)
            .ok_or_else(|| Error::MissingBackendScores(task.id.clone()))?;
        if logits.len() != task.candidates.len() {
            return Err(Error::CandidateCountMismatch {
                question: task.id.clone(),
                expected: task.candidates.len(),
                actual: logits.len(),
            });
        }
        let probabilities = self.calibrator.probabilities(logits)?;
        let winner = argmax(&probabilities)
            .ok_or_else(|| Error::Internal("classification produced no probabilities".to_owned()))?;
        let distribution = task
            .candidates
            .iter()
            .zip(&probabilities)
            .map(|(candidate, probability)| (candidate.key.clone(), *probability))
            .collect();

        Ok(ClassificationResult {
            choice: task.candidates[winner].key.clone(),
            probabilities: distribution,
        })
    }

    fn score_batch(
        &self,
        batch: &DecisionBatch,
    ) -> Result<(BackendOutput, HashMap<String, Vec<f32>>, ExecutionPlan), Error> {
        let capabilities = self.backend.capabilities();
        let resource_hints = self.backend.resource_hints();
        let plan = self
            .planner
            .plan_with_hints(capabilities, resource_hints, batch);
        if !capabilities.supports(plan) {
            return Err(Error::Internal(
                "execution planner produced a plan unsupported by the backend".to_owned(),
            ));
        }

        let backend_output = self.backend.score(batch, plan)?;
        let effective_plan = backend_output.metrics.effective_plan.unwrap_or(plan);
        let indexed_scores = index_backend_scores(batch, &backend_output.scores)?;
        Ok((backend_output, indexed_scores, effective_plan))
    }
}

impl<B, C> TurboJevEngine<B, C>
where
    B: FusedScoringBackend,
    C: Calibrator,
{
    /// Return whether this exact request group is eligible for true fused execution.
    /// This is shape-aware: a runtime may support fused batching globally while a
    /// particular request group must fall back to the independent reference path.
    pub fn can_evaluate_fused(
        &self,
        requests: &[EvaluationRequest],
    ) -> Result<bool, Error> {
        if requests.len() < 2 {
            return Ok(false);
        }
        let capabilities = self.backend.fused_capabilities().validate()?;
        if !capabilities.fused_batch {
            return Ok(false);
        }
        if let Some(maximum) = capabilities.max_fused_items {
            if requests.len() > maximum {
                return Ok(false);
            }
        }

        let mut batches = Vec::with_capacity(requests.len());
        let mut plans = Vec::with_capacity(requests.len());
        for request in requests {
            request.validate()?;
            let batch = compile_decision_batch(request);
            let backend_capabilities = self.backend.capabilities();
            let hints = self.backend.resource_hints();
            let plan = self.planner.plan_with_hints(backend_capabilities, hints, &batch);
            if !backend_capabilities.supports(plan) {
                return Ok(false);
            }
            batches.push(batch);
            plans.push(plan);
        }
        self.backend.can_score_fused(&batches, &plans).map_err(Into::into)
    }

    /// Evaluate multiple independent requests through one fused backend call.
    ///
    /// This method is available only to backends that explicitly implement
    /// [`FusedScoringBackend`]. It never emulates fusion by repeatedly calling
    /// the ordinary single-request backend method.
    pub fn evaluate_fused(
        &self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<EvaluationResponse>, Error> {
        if requests.len() < 2 {
            return Err(Error::InvalidFusedBatch(
                "fused evaluation requires at least two independent requests".to_owned(),
            ));
        }
        if !self.can_evaluate_fused(requests)? {
            return Err(Error::InvalidFusedBatch(
                "this request group is not eligible for the backend fused path".to_owned(),
            ));
        }
        let capabilities = self.backend.fused_capabilities().validate()?;
        if !capabilities.fused_batch {
            return Err(Error::InvalidFusedBatch(
                "backend does not advertise fused cross-request scoring".to_owned(),
            ));
        }
        if let Some(maximum) = capabilities.max_fused_items {
            if requests.len() > maximum {
                return Err(Error::InvalidFusedBatch(format!(
                    "fused evaluation received {} requests; backend maximum is {}",
                    requests.len(), maximum
                )));
            }
        }

        let mut batches = Vec::with_capacity(requests.len());
        let mut plans = Vec::with_capacity(requests.len());
        for request in requests {
            request.validate()?;
            let batch = compile_decision_batch(request);
            let capabilities = self.backend.capabilities();
            let hints = self.backend.resource_hints();
            let plan = self.planner.plan_with_hints(capabilities, hints, &batch);
            if !capabilities.supports(plan) {
                return Err(Error::Internal(
                    "execution planner produced a plan unsupported by the backend".to_owned(),
                ));
            }
            batches.push(batch);
            plans.push(plan);
        }

        let outputs = self.backend.score_fused(&batches, &plans)?;
        if outputs.len() != requests.len() {
            return Err(Error::FusedResponseCountMismatch {
                expected: requests.len(),
                actual: outputs.len(),
            });
        }

        let mut responses = Vec::with_capacity(requests.len());
        for (((request, batch), output), requested_plan) in requests
            .iter()
            .cloned()
            .zip(batches.iter())
            .zip(outputs.into_iter())
            .zip(plans.into_iter())
        {
            let effective_plan = output.metrics.effective_plan.unwrap_or(requested_plan);
            let indexed_scores = index_backend_scores(batch, &output.scores)?;
            responses.push(build_evaluation_response(
                &request,
                batch,
                output,
                &indexed_scores,
                effective_plan,
                self.backend.name(),
                &self.calibrator,
            )?);
        }
        Ok(responses)
    }
}

/// Asynchronous backend-neutral TurboJev engine.
///
/// The async engine shares the same schema, planning, calibration and typed
/// answer construction as [`TurboJevEngine`]. Only backend execution is
/// asynchronous, so no executor framework is pulled into the portable core.
pub struct TurboJevAsyncEngine<B, C = SoftmaxCalibrator> {
    backend: B,
    calibrator: C,
    planner: ExecutionPlanner,
}

impl<B> TurboJevAsyncEngine<B, SoftmaxCalibrator>
where
    B: AsyncScoringBackend,
{
    /// Create an async engine using uncalibrated softmax and automatic planning.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            calibrator: SoftmaxCalibrator,
            planner: ExecutionPlanner::default(),
        }
    }
}

impl<B, C> TurboJevAsyncEngine<B, C>
where
    B: AsyncScoringBackend,
    C: Calibrator,
{
    /// Create an async engine with an explicit calibrator.
    pub fn with_calibrator(backend: B, calibrator: C) -> Self {
        Self {
            backend,
            calibrator,
            planner: ExecutionPlanner::default(),
        }
    }

    /// Override execution planning policy without changing the backend.
    pub fn with_planning_policy(mut self, policy: PlanningPolicy) -> Self {
        self.planner = ExecutionPlanner::with_budget(policy, self.planner.budget());
        self
    }

    /// Apply a portable resource budget to automatic execution planning.
    pub fn with_execution_budget(mut self, budget: ExecutionBudget) -> Self {
        self.planner = self.planner.set_budget(budget);
        self
    }

    /// Access the configured inference backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Evaluate a full typed decision request without blocking the host runtime.
    pub async fn evaluate(
        &self,
        request: EvaluationRequest,
    ) -> Result<EvaluationResponse, Error> {
        request.validate()?;
        let batch = compile_decision_batch(&request);
        let (backend_output, indexed_scores, plan) = self.score_batch(&batch).await?;
        build_evaluation_response(
            &request,
            &batch,
            backend_output,
            &indexed_scores,
            plan,
            self.backend.name(),
            &self.calibrator,
        )
    }

    /// Ergonomic async single-choice helper.
    pub async fn classify<I, S>(
        &self,
        state: impl Into<String>,
        options: I,
    ) -> Result<ClassificationResult, Error>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let (batch, task) = compile_classification(state.into(), options)?;
        let (_backend_output, indexed_scores, _plan) = self.score_batch(&batch).await?;
        build_classification_result(&task, &indexed_scores, &self.calibrator)
    }

    async fn score_batch(
        &self,
        batch: &DecisionBatch,
    ) -> Result<(BackendOutput, HashMap<String, Vec<f32>>, ExecutionPlan), Error> {
        let capabilities = self.backend.capabilities();
        let resource_hints = self.backend.resource_hints();
        let plan = self
            .planner
            .plan_with_hints(capabilities, resource_hints, batch);
        if !capabilities.supports(plan) {
            return Err(Error::Internal(
                "execution planner produced a plan unsupported by the backend".to_owned(),
            ));
        }

        let backend_output = self.backend.score_async(batch, plan).await?;
        let effective_plan = backend_output.metrics.effective_plan.unwrap_or(plan);
        let indexed_scores = index_backend_scores(batch, &backend_output.scores)?;
        Ok((backend_output, indexed_scores, effective_plan))
    }
}

fn build_evaluation_response<C: Calibrator>(
    request: &EvaluationRequest,
    batch: &DecisionBatch,
    backend_output: BackendOutput,
    indexed_scores: &HashMap<String, Vec<f32>>,
    plan: ExecutionPlan,
    backend_name: &str,
    calibrator: &C,
) -> Result<EvaluationResponse, Error> {
    let mut answers = IndexMap::with_capacity(request.questions.len());
    for (question_id, question) in &request.questions {
        let task = batch
            .tasks
            .iter()
            .find(|task| task.id == *question_id)
            .ok_or_else(|| Error::Internal(format!("compiled task missing: {question_id}")))?;
        let logits = indexed_scores
            .get(question_id)
            .ok_or_else(|| Error::MissingBackendScores(question_id.clone()))?;

        if logits.len() != task.candidates.len() {
            return Err(Error::CandidateCountMismatch {
                question: question_id.clone(),
                expected: task.candidates.len(),
                actual: logits.len(),
            });
        }

        let probabilities = calibrator.probabilities(logits)?;
        let answer = build_answer(question, task, &probabilities)?;
        answers.insert(question_id.clone(), answer);
    }

    let BackendOutput { model, metrics, .. } = backend_output;
    Ok(EvaluationResponse {
        model: model.or_else(|| request.model.clone()),
        answers,
        usage: Usage {
            decisions: batch.tasks.len(),
            input_tokens: metrics.input_tokens,
            generated_tokens: 0,
        },
        metadata: EngineMetadata {
            backend: backend_name.to_owned(),
            calibration: calibrator.name().to_owned(),
            calibrated: calibrator.is_fitted(),
            confidence_method: "selected-candidate-probability".to_owned(),
            contract: CORE_CONTRACT.to_owned(),
            execution: execution_metadata(plan),
        },
    })
}

fn execution_metadata(plan: ExecutionPlan) -> ExecutionMetadata {
    ExecutionMetadata {
        prefix_reuse: match plan.prefix_reuse {
            turbojev_backend::PrefixReuse::None => "none",
            turbojev_backend::PrefixReuse::Shared => "shared",
        }
        .to_owned(),
        scheduling: match plan.scheduling {
            turbojev_backend::TaskScheduling::Serial => "serial",
            turbojev_backend::TaskScheduling::Parallel => "parallel",
        }
        .to_owned(),
        parallel_task_limit: plan.parallel_task_limit,
        projection: match plan.projection {
            turbojev_backend::CandidateProjection::FullVocabulary => "full-vocabulary",
            turbojev_backend::CandidateProjection::CandidatesOnly => "candidates-only",
        }
        .to_owned(),
    }
}

fn compile_classification<I, S>(
    state: String,
    options: I,
) -> Result<(DecisionBatch, DecisionTask), Error>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let options: Vec<String> = options.into_iter().map(Into::into).collect();
    if options.len() < 2 {
        return Err(Error::InvalidClassification(
            "classification requires at least two options".to_owned(),
        ));
    }
    if options.iter().any(|option| option.trim().is_empty()) {
        return Err(Error::InvalidClassification(
            "classification options must not be empty".to_owned(),
        ));
    }
    let unique: HashSet<&str> = options.iter().map(String::as_str).collect();
    if unique.len() != options.len() {
        return Err(Error::InvalidClassification(
            "classification options must be unique".to_owned(),
        ));
    }

    let candidates = options
        .into_iter()
        .map(|option| Candidate {
            key: option.clone(),
            description: option,
        })
        .collect();
    let task = DecisionTask {
        id: "classification".to_owned(),
        intent: DecisionIntent::BestMatch,
        instruction: None,
        candidates,
    };
    let batch = DecisionBatch {
        state: Value::String(state),
        tasks: vec![task.clone()],
    };
    Ok((batch, task))
}

fn build_classification_result<C: Calibrator>(
    task: &DecisionTask,
    indexed_scores: &HashMap<String, Vec<f32>>,
    calibrator: &C,
) -> Result<ClassificationResult, Error> {
    let logits = indexed_scores
        .get(&task.id)
        .ok_or_else(|| Error::MissingBackendScores(task.id.clone()))?;
    if logits.len() != task.candidates.len() {
        return Err(Error::CandidateCountMismatch {
            question: task.id.clone(),
            expected: task.candidates.len(),
            actual: logits.len(),
        });
    }
    let probabilities = calibrator.probabilities(logits)?;
    let winner = argmax(&probabilities)
        .ok_or_else(|| Error::Internal("classification produced no probabilities".to_owned()))?;
    let distribution = task
        .candidates
        .iter()
        .zip(&probabilities)
        .map(|(candidate, probability)| (candidate.key.clone(), *probability))
        .collect();

    Ok(ClassificationResult {
        choice: task.candidates[winner].key.clone(),
        probabilities: distribution,
    })
}

/// Compile a validated public request into the exact semantic decision batch used by inference.
///
/// This is public so specialization/training tooling can reuse the same candidate
/// normalization as inference instead of maintaining a second implementation.
pub fn compile_decision_batch(request: &EvaluationRequest) -> DecisionBatch {
    let tasks = request
        .questions
        .iter()
        .map(|(id, question)| DecisionTask {
            id: id.clone(),
            intent: DecisionIntent::Criterion,
            instruction: Some(question.instructions().to_owned()),
            candidates: candidates_for(question),
        })
        .collect();

    DecisionBatch {
        state: request.state.clone(),
        tasks,
    }
}

fn candidates_for(question: &Question) -> Vec<Candidate> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, description)| Candidate {
                key: key.clone(),
                description: description.clone(),
            })
            .collect(),
        Question::Noul { .. } => vec![
            Candidate {
                key: "false".to_owned(),
                description: "false".to_owned(),
            },
            Candidate {
                key: "true".to_owned(),
                description: "true".to_owned(),
            },
        ],
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(index, description)| Candidate {
                key: index.to_string(),
                description: description.clone(),
            })
            .collect(),
    }
}

fn index_backend_scores(
    batch: &DecisionBatch,
    scores: &[TaskScores],
) -> Result<HashMap<String, Vec<f32>>, Error> {
    if scores.len() != batch.tasks.len() {
        return Err(Error::BackendTaskCountMismatch {
            expected: batch.tasks.len(),
            actual: scores.len(),
        });
    }

    let mut indexed = HashMap::with_capacity(scores.len());
    for score in scores {
        if indexed
            .insert(score.id.clone(), score.logits.clone())
            .is_some()
        {
            return Err(Error::DuplicateBackendScores(score.id.clone()));
        }
    }

    for task in &batch.tasks {
        if !indexed.contains_key(&task.id) {
            return Err(Error::MissingBackendScores(task.id.clone()));
        }
    }

    Ok(indexed)
}

fn build_answer(
    question: &Question,
    task: &DecisionTask,
    probabilities: &[f64],
) -> Result<Answer, Error> {
    if probabilities.len() != task.candidates.len() {
        return Err(Error::CandidateCountMismatch {
            question: task.id.clone(),
            expected: task.candidates.len(),
            actual: probabilities.len(),
        });
    }

    let winner = argmax(probabilities)
        .ok_or_else(|| Error::Internal(format!("no probabilities for task {}", task.id)))?;

    match question {
        Question::Choice { .. } => {
            let distribution = task
                .candidates
                .iter()
                .zip(probabilities)
                .map(|(candidate, probability)| (candidate.key.clone(), *probability))
                .collect();

            Ok(Answer::Choice {
                choice: task.candidates[winner].key.clone(),
                confidence: probabilities[winner],
                probabilities: distribution,
            })
        }
        Question::Noul { .. } => Ok(Answer::Noul {
            noul: probabilities[1],
        }),
        Question::Score { criteria, .. } => {
            let probabilities_map: IndexMap<String, f64> = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| (index.to_string(), *probability))
                .collect();
            let legend: IndexMap<String, String> = criteria
                .iter()
                .enumerate()
                .map(|(index, description)| (index.to_string(), description.clone()))
                .collect();
            let expected_score = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();

            Ok(Answer::Score {
                score: expected_score,
                confidence: probabilities[winner],
                probabilities: probabilities_map,
                legend,
            })
        }
    }
}

fn argmax(values: &[f64]) -> Option<usize> {
    let mut best_index = 0;
    let first = *values.first()?;
    let mut best_value = first;

    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }

    Some(best_index)
}

/// Core engine error.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid public request.
    #[error(transparent)]
    Schema(#[from] SchemaError),
    /// Backend failure.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Probability/calibration failure.
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
    /// Backend did not return the same number of tasks it received.
    #[error("backend returned {actual} score rows; expected {expected}")]
    BackendTaskCountMismatch { expected: usize, actual: usize },
    /// Backend returned duplicate scores for one task identifier.
    #[error("backend returned duplicate score row for question {0:?}")]
    DuplicateBackendScores(String),
    /// Backend omitted one task.
    #[error("backend omitted scores for question {0:?}")]
    MissingBackendScores(String),
    /// Backend candidate/logit counts do not match.
    #[error(
        "question {question:?} has {expected} candidates but backend returned {actual} logits"
    )]
    CandidateCountMismatch {
        question: String,
        expected: usize,
        actual: usize,
    },
    /// Invalid convenience-classification input.
    #[error("invalid classification request: {0}")]
    InvalidClassification(String),
    /// Invalid cross-request fused-evaluation request.
    #[error("invalid fused evaluation request: {0}")]
    InvalidFusedBatch(String),
    /// Fused backend returned the wrong number of independent responses.
    #[error("fused backend returned {actual} outputs; expected {expected}")]
    FusedResponseCountMismatch { expected: usize, actual: usize },
    /// Internal invariant failure.
    #[error("internal TurboJev invariant failed: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use indexmap::indexmap;
    use turbojev_backend_mock::FixtureBackend;

    use super::*;

    #[test]
    fn produces_typed_choice_from_backend_logits() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![0.1, 4.0, -2.0])
            .build();
        let engine = TurboJevEngine::new(backend);
        let request = EvaluationRequest {
            model: None,
            state: serde_json::json!({"message": "package missing"}),
            questions: indexmap! {
                "route".into() => Question::choice(
                    "Where should this go?",
                    indexmap! {
                        "billing".into() => "billing".into(),
                        "shipping".into() => "shipping".into(),
                        "other".into() => "other".into(),
                    },
                )
            },
        };

        let response = engine.evaluate(request).unwrap();
        match &response.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "shipping"),
            _ => panic!("wrong answer type"),
        }
        assert_eq!(response.usage.generated_tokens, 0);
    }

    struct AsyncFixtureBackend;

    impl turbojev_backend::AsyncScoringBackend for AsyncFixtureBackend {
        fn name(&self) -> &'static str {
            "async-fixture"
        }

        fn capabilities(&self) -> turbojev_backend::BackendCapabilities {
            turbojev_backend::BackendCapabilities::default()
        }

        fn score_async<'a>(
            &'a self,
            batch: &'a DecisionBatch,
            _plan: ExecutionPlan,
        ) -> turbojev_backend::BackendFuture<'a, Result<BackendOutput, BackendError>> {
            Box::pin(std::future::ready(Ok(BackendOutput {
                    model: Some("async-open-model".to_owned()),
                    scores: batch
                        .tasks
                        .iter()
                        .map(|task| TaskScores {
                            id: task.id.clone(),
                            logits: (0..task.candidates.len())
                                .map(|index| index as f32)
                                .collect(),
                        })
                        .collect(),
                    metrics: turbojev_backend::BackendMetrics::default(),
                })))
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }

        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(output) => return output,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn async_engine_uses_same_typed_decision_contract() {
        let engine = TurboJevAsyncEngine::new(AsyncFixtureBackend);
        let result = block_on(engine.classify("hola", ["uno", "dos", "tres"]))
            .expect("async classification should succeed");
        assert_eq!(result.choice, "tres");
    }

    #[test]
    fn reference_policy_remains_available() {
        let backend = FixtureBackend::builder()
            .with_logits("classification", vec![2.0, 1.0])
            .build();
        let engine = TurboJevEngine::new(backend)
            .with_planning_policy(PlanningPolicy::Reference);
        let result = engine.classify("hola", ["si", "no"]).unwrap();
        assert_eq!(result.choice, "si");
    }

    #[test]
    fn fused_fixture_engine_matches_independent_evaluation() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![0.1, 4.0, -2.0])
            .build();
        let engine = TurboJevEngine::new(backend);
        let make_request = |state: &str| EvaluationRequest {
            model: None,
            state: serde_json::json!({"message": state}),
            questions: indexmap! {
                "route".into() => Question::choice(
                    "Where should this go?",
                    indexmap! {
                        "billing".into() => "billing".into(),
                        "shipping".into() => "shipping".into(),
                        "other".into() => "other".into(),
                    },
                )
            },
        };
        let requests = vec![make_request("one"), make_request("two")];
        let independent: Vec<_> = requests
            .iter()
            .cloned()
            .map(|request| engine.evaluate(request).unwrap())
            .collect();
        let fused = engine.evaluate_fused(&requests).unwrap();
        assert_eq!(fused, independent);
    }

}
