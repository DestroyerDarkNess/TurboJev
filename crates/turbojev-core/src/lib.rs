#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! TurboJev's portable, backend-neutral decision engine.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
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

/// Mitigation for positional bias in unordered `choice` questions.
///
/// `None` scores the supplied order once. Opt-in `Cyclic` evaluates each cyclic
/// candidate order, maps scores back to stable keys, and averages raw logits;
/// it costs one model evaluation per candidate. Ordered score levels and
/// `noul` remain single-pass because their order has semantic meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderRobustness {
    /// Score the caller's supplied order once.
    #[default]
    None,
    /// Average every cyclic rotation for each unordered choice question.
    Cyclic,
}

impl OrderRobustness {
    const MAX_CYCLIC_CANDIDATES: usize = 32;

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Cyclic => "cyclic",
        }
    }
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
    order_robustness: OrderRobustness,
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
            order_robustness: OrderRobustness::None,
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
            order_robustness: OrderRobustness::None,
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

    /// Set the candidate-order policy for unordered choices.
    pub fn with_order_robustness(mut self, value: OrderRobustness) -> Self {
        self.order_robustness = value;
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
        let (backend_output, indexed_scores, plan, order_evaluations) = match self.order_robustness
        {
            OrderRobustness::None => {
                let (output, scores, plan) = self.score_batch(&batch)?;
                (output, scores, plan, 0)
            }
            OrderRobustness::Cyclic => self.score_cyclic_choices(&request, &batch)?,
        };
        let mut response = build_evaluation_response(
            &request,
            &batch,
            backend_output,
            &indexed_scores,
            plan,
            self.backend.name(),
            &self.calibrator,
        )?;
        response.metadata.order_robustness = self.order_robustness.name().to_owned();
        response.metadata.order_evaluations = order_evaluations;
        Ok(response)
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

    fn score_cyclic_choices(
        &self,
        request: &EvaluationRequest,
        batch: &DecisionBatch,
    ) -> Result<
        (
            BackendOutput,
            HashMap<String, Vec<f32>>,
            ExecutionPlan,
            usize,
        ),
        Error,
    > {
        let choice_ids: HashSet<&str> = request
            .questions
            .iter()
            .filter_map(|(id, question)| {
                matches!(question, Question::Choice { .. }).then_some(id.as_str())
            })
            .collect();
        if choice_ids.is_empty() {
            let (output, scores, plan) = self.score_batch(batch)?;
            return Ok((output, scores, plan, 0));
        }
        for task in batch
            .tasks
            .iter()
            .filter(|task| choice_ids.contains(task.id.as_str()))
        {
            if task.candidates.len() > OrderRobustness::MAX_CYCLIC_CANDIDATES {
                return Err(Error::OrderRobustnessCandidateLimit {
                    question: task.id.clone(),
                    actual: task.candidates.len(),
                    max: OrderRobustness::MAX_CYCLIC_CANDIDATES,
                });
            }
        }

        let mut indexed = HashMap::with_capacity(batch.tasks.len());
        let mut model = None;
        let mut input_tokens = 0usize;
        let mut input_known = false;
        let mut plan = None;
        let mut evaluations = 0usize;
        let ordinary = DecisionBatch {
            state: batch.state.clone(),
            tasks: batch
                .tasks
                .iter()
                .filter(|task| !choice_ids.contains(task.id.as_str()))
                .cloned()
                .collect(),
        };
        if !ordinary.tasks.is_empty() {
            let (output, scores, effective_plan) = self.score_batch(&ordinary)?;
            model = output.model;
            if let Some(value) = output.metrics.input_tokens {
                input_tokens += value;
                input_known = true;
            }
            indexed.extend(scores);
            plan = Some(effective_plan);
            evaluations += 1;
        }
        for task in batch
            .tasks
            .iter()
            .filter(|task| choice_ids.contains(task.id.as_str()))
        {
            let count = task.candidates.len();
            let mut total = vec![0.0f32; count];
            for rotation in 0..count {
                let mut rotated = task.clone();
                rotated.candidates.rotate_left(rotation);
                let rotated_batch = DecisionBatch {
                    state: batch.state.clone(),
                    tasks: vec![rotated],
                };
                let (output, scores, effective_plan) = self.score_batch(&rotated_batch)?;
                if model.is_none() {
                    model = output.model;
                }
                if let Some(value) = output.metrics.input_tokens {
                    input_tokens += value;
                    input_known = true;
                }
                plan.get_or_insert(effective_plan);
                let logits = scores
                    .get(&task.id)
                    .ok_or_else(|| Error::MissingBackendScores(task.id.clone()))?;
                if logits.len() != count {
                    return Err(Error::CandidateCountMismatch {
                        question: task.id.clone(),
                        expected: count,
                        actual: logits.len(),
                    });
                }
                for (position, value) in logits.iter().enumerate() {
                    total[(position + rotation) % count] += *value;
                }
                evaluations += 1;
            }
            for value in &mut total {
                *value /= count as f32;
            }
            indexed.insert(task.id.clone(), total);
        }
        Ok((
            BackendOutput {
                model,
                scores: Vec::new(),
                metrics: turbojev_backend::BackendMetrics {
                    input_tokens: input_known.then_some(input_tokens),
                    ..Default::default()
                },
            },
            indexed,
            plan.ok_or_else(|| {
                Error::Internal("order robustness did not create an execution plan".to_owned())
            })?,
            evaluations,
        ))
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
    pub fn can_evaluate_fused(&self, requests: &[EvaluationRequest]) -> Result<bool, Error> {
        if self.order_robustness != OrderRobustness::None {
            return Ok(false);
        }
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
            let plan = self
                .planner
                .plan_with_hints(backend_capabilities, hints, &batch);
            if !backend_capabilities.supports(plan) {
                return Ok(false);
            }
            batches.push(batch);
            plans.push(plan);
        }
        self.backend
            .can_score_fused(&batches, &plans)
            .map_err(Into::into)
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
                    requests.len(),
                    maximum
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
    order_robustness: OrderRobustness,
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
            order_robustness: OrderRobustness::None,
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
            order_robustness: OrderRobustness::None,
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

    /// Set the candidate-order policy for unordered choices.
    pub fn with_order_robustness(mut self, value: OrderRobustness) -> Self {
        self.order_robustness = value;
        self
    }

    /// Access the configured inference backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Evaluate a full typed decision request without blocking the host runtime.
    pub async fn evaluate(&self, request: EvaluationRequest) -> Result<EvaluationResponse, Error> {
        request.validate()?;
        let batch = compile_decision_batch(&request);
        let (backend_output, indexed_scores, plan, order_evaluations) = match self.order_robustness
        {
            OrderRobustness::None => {
                let (output, scores, plan) = self.score_batch(&batch).await?;
                (output, scores, plan, 0)
            }
            OrderRobustness::Cyclic => self.score_cyclic_choices(&request, &batch).await?,
        };
        let mut response = build_evaluation_response(
            &request,
            &batch,
            backend_output,
            &indexed_scores,
            plan,
            self.backend.name(),
            &self.calibrator,
        )?;
        response.metadata.order_robustness = self.order_robustness.name().to_owned();
        response.metadata.order_evaluations = order_evaluations;
        Ok(response)
    }

    async fn score_cyclic_choices(
        &self,
        request: &EvaluationRequest,
        batch: &DecisionBatch,
    ) -> Result<
        (
            BackendOutput,
            HashMap<String, Vec<f32>>,
            ExecutionPlan,
            usize,
        ),
        Error,
    > {
        let choice_ids: HashSet<&str> = request
            .questions
            .iter()
            .filter_map(|(id, question)| {
                matches!(question, Question::Choice { .. }).then_some(id.as_str())
            })
            .collect();
        if choice_ids.is_empty() {
            let (output, scores, plan) = self.score_batch(batch).await?;
            return Ok((output, scores, plan, 0));
        }
        for task in batch
            .tasks
            .iter()
            .filter(|task| choice_ids.contains(task.id.as_str()))
        {
            if task.candidates.len() > OrderRobustness::MAX_CYCLIC_CANDIDATES {
                return Err(Error::OrderRobustnessCandidateLimit {
                    question: task.id.clone(),
                    actual: task.candidates.len(),
                    max: OrderRobustness::MAX_CYCLIC_CANDIDATES,
                });
            }
        }

        let mut indexed = HashMap::with_capacity(batch.tasks.len());
        let mut model = None;
        let mut input_tokens = 0usize;
        let mut input_known = false;
        let mut plan = None;
        let mut evaluations = 0usize;
        let ordinary = DecisionBatch {
            state: batch.state.clone(),
            tasks: batch
                .tasks
                .iter()
                .filter(|task| !choice_ids.contains(task.id.as_str()))
                .cloned()
                .collect(),
        };
        if !ordinary.tasks.is_empty() {
            let (output, scores, effective_plan) = self.score_batch(&ordinary).await?;
            model = output.model;
            if let Some(value) = output.metrics.input_tokens {
                input_tokens += value;
                input_known = true;
            }
            indexed.extend(scores);
            plan = Some(effective_plan);
            evaluations += 1;
        }
        for task in batch
            .tasks
            .iter()
            .filter(|task| choice_ids.contains(task.id.as_str()))
        {
            let count = task.candidates.len();
            let mut total = vec![0.0f32; count];
            for rotation in 0..count {
                let mut rotated = task.clone();
                rotated.candidates.rotate_left(rotation);
                let rotated_batch = DecisionBatch {
                    state: batch.state.clone(),
                    tasks: vec![rotated],
                };
                let (output, scores, effective_plan) = self.score_batch(&rotated_batch).await?;
                if model.is_none() {
                    model = output.model;
                }
                if let Some(value) = output.metrics.input_tokens {
                    input_tokens += value;
                    input_known = true;
                }
                plan.get_or_insert(effective_plan);
                let logits = scores
                    .get(&task.id)
                    .ok_or_else(|| Error::MissingBackendScores(task.id.clone()))?;
                if logits.len() != count {
                    return Err(Error::CandidateCountMismatch {
                        question: task.id.clone(),
                        expected: count,
                        actual: logits.len(),
                    });
                }
                for (position, value) in logits.iter().enumerate() {
                    total[(position + rotation) % count] += *value;
                }
                evaluations += 1;
            }
            for value in &mut total {
                *value /= count as f32;
            }
            indexed.insert(task.id.clone(), total);
        }
        Ok((
            BackendOutput {
                model,
                scores: Vec::new(),
                metrics: turbojev_backend::BackendMetrics {
                    input_tokens: input_known.then_some(input_tokens),
                    ..Default::default()
                },
            },
            indexed,
            plan.ok_or_else(|| {
                Error::Internal("order robustness did not create an execution plan".to_owned())
            })?,
            evaluations,
        ))
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
            order_robustness: "none".to_owned(),
            order_evaluations: 0,
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
    /// Cyclic order correction would exceed its bounded inference cost.
    #[error(
        "order robustness for question {question:?} has {actual} candidates; maximum is {max}"
    )]
    OrderRobustnessCandidateLimit {
        question: String,
        actual: usize,
        max: usize,
    },
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
        assert_eq!(OrderRobustness::default(), OrderRobustness::None);
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
        assert_eq!(response.metadata.order_robustness, "none");
        assert_eq!(response.metadata.order_evaluations, 0);
    }

    struct PositionBiasedBackend;

    impl ScoringBackend for PositionBiasedBackend {
        fn name(&self) -> &'static str {
            "position-biased-fixture"
        }

        fn capabilities(&self) -> turbojev_backend::BackendCapabilities {
            turbojev_backend::BackendCapabilities::default()
        }

        fn score(
            &self,
            batch: &DecisionBatch,
            _plan: ExecutionPlan,
        ) -> Result<BackendOutput, BackendError> {
            let position_bias = [5.0, 0.0, -5.0];
            Ok(BackendOutput {
                model: Some("position-biased-fixture".to_owned()),
                scores: batch
                    .tasks
                    .iter()
                    .map(|task| TaskScores {
                        id: task.id.clone(),
                        logits: task
                            .candidates
                            .iter()
                            .enumerate()
                            .map(|(position, candidate)| {
                                let semantic_logit = match candidate.key.as_str() {
                                    "a" => 0.1,
                                    "b" => 1.0,
                                    "c" => 0.2,
                                    _ => 0.0,
                                };
                                semantic_logit + position_bias[position]
                            })
                            .collect(),
                    })
                    .collect(),
                metrics: turbojev_backend::BackendMetrics::default(),
            })
        }
    }

    impl AsyncScoringBackend for PositionBiasedBackend {
        fn name(&self) -> &'static str {
            "position-biased-fixture"
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
                model: Some("position-biased-fixture".to_owned()),
                scores: batch
                    .tasks
                    .iter()
                    .map(|task| TaskScores {
                        id: task.id.clone(),
                        logits: task
                            .candidates
                            .iter()
                            .enumerate()
                            .map(|(position, candidate)| {
                                let semantic_logit = match candidate.key.as_str() {
                                    "a" => 0.1,
                                    "b" => 1.0,
                                    "c" => 0.2,
                                    _ => 0.0,
                                };
                                semantic_logit + [5.0, 0.0, -5.0][position]
                            })
                            .collect(),
                    })
                    .collect(),
                metrics: turbojev_backend::BackendMetrics::default(),
            })))
        }
    }

    fn position_biased_request() -> EvaluationRequest {
        EvaluationRequest {
            model: None,
            state: serde_json::json!({"message": "route this"}),
            questions: indexmap! {
                "route".into() => Question::choice(
                    "Choose the best route.",
                    indexmap! {
                        "a".into() => "option a".into(),
                        "b".into() => "option b".into(),
                        "c".into() => "option c".into(),
                    },
                )
            },
        }
    }

    #[test]
    fn cyclic_order_robustness_cancels_position_bias_for_evaluate() {
        let baseline = TurboJevEngine::new(PositionBiasedBackend)
            .with_order_robustness(OrderRobustness::None)
            .evaluate(position_biased_request())
            .unwrap();
        match &baseline.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "a"),
            _ => panic!("wrong answer type"),
        }

        let corrected = TurboJevEngine::new(PositionBiasedBackend)
            .with_order_robustness(OrderRobustness::Cyclic)
            .evaluate(position_biased_request())
            .unwrap();
        match &corrected.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "b"),
            _ => panic!("wrong answer type"),
        }
        assert_eq!(corrected.metadata.order_robustness, "cyclic");
        assert_eq!(corrected.metadata.order_evaluations, 3);
    }

    struct AsyncFixtureBackend;

    impl AsyncScoringBackend for AsyncFixtureBackend {
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
    fn async_engine_uses_typed_decision_contract() {
        let engine = TurboJevAsyncEngine::new(AsyncFixtureBackend)
            .with_order_robustness(OrderRobustness::None);
        let result = block_on(engine.evaluate(position_biased_request()))
            .expect("async evaluation should succeed");
        match &result.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "c"),
            _ => panic!("wrong answer type"),
        }
    }

    #[test]
    fn async_cyclic_order_robustness_is_invariant_to_candidate_rotation() {
        let result = block_on(
            TurboJevAsyncEngine::new(PositionBiasedBackend)
                .with_order_robustness(OrderRobustness::Cyclic)
                .evaluate(position_biased_request()),
        )
        .unwrap();
        match &result.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "b"),
            _ => panic!("wrong answer type"),
        }
    }

    #[test]
    fn reference_policy_remains_available() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![2.0, 1.0, 0.0])
            .build();
        let engine = TurboJevEngine::new(backend).with_planning_policy(PlanningPolicy::Reference);
        let result = engine.evaluate(position_biased_request()).unwrap();
        match &result.answers["route"] {
            Answer::Choice { choice, .. } => assert_eq!(choice, "a"),
            _ => panic!("wrong answer type"),
        }
    }

    #[test]
    fn fused_fixture_engine_matches_independent_evaluation() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![0.1, 4.0, -2.0])
            .build();
        let engine = TurboJevEngine::new(backend).with_order_robustness(OrderRobustness::None);
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
