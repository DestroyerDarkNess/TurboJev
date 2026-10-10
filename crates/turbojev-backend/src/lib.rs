#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Stable, operating-system-neutral backend SPI.
//!
//! Backends may wrap GGUF, safetensors, WebGPU, a pure Rust runtime, an
//! application-provided accelerator, or any future open-weight execution
//! environment. The portable contract intentionally has no filesystem,
//! networking, process, environment-variable, or target-OS concepts.

use std::{future::Future, pin::Pin};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// A normalized candidate passed to an inference backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// Stable output key returned to callers.
    pub key: String,
    /// Human-readable semantics shown to the model.
    pub description: String,
}

/// Semantic intent of a decision task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionIntent {
    /// Apply explicit caller-supplied criterion text.
    Criterion,
    /// Choose the candidate that best matches the supplied state. The runtime's
    /// prompt profile decides how to express this intent to the model.
    BestMatch,
}

/// One independent semantic decision over a shared state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionTask {
    /// Stable question identifier.
    pub id: String,
    /// Semantic operation.
    pub intent: DecisionIntent,
    /// Caller-supplied criterion text for `Criterion` tasks. It may use any
    /// human language. `BestMatch` tasks intentionally need no English default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    /// Allowed outputs in deterministic order.
    pub candidates: Vec<Candidate>,
}

/// A batch of typed decisions sharing the same state/evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionBatch {
    /// Original shared state. Backends must not silently mutate/truncate it.
    pub state: Value,
    /// Independent decision tasks in request order.
    pub tasks: Vec<DecisionTask>,
}

/// Whether a shared prefix may be reused across decision tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefixReuse {
    /// Recompute each complete task independently.
    None,
    /// Prefill the shared state/prefix once and branch from it.
    Shared,
}

/// Task scheduling requested from a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskScheduling {
    /// Evaluate tasks one by one.
    Serial,
    /// Evaluate independent task suffixes as bounded batches.
    Parallel,
}

/// Output projection requested from a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateProjection {
    /// Compute the model's ordinary vocabulary head and gather candidate rows.
    FullVocabulary,
    /// Compute only output rows required by the candidates.
    CandidatesOnly,
}

/// Semantic execution plan chosen by the portable planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Shared-state reuse strategy.
    pub prefix_reuse: PrefixReuse,
    /// Task batching strategy.
    pub scheduling: TaskScheduling,
    /// Maximum number of tasks allowed in one parallel suffix batch.
    ///
    /// Serial plans always use `1`. This value is deliberately model/runtime
    /// neutral; it is derived from portable caller budgets and backend hints.
    pub parallel_task_limit: usize,
    /// Candidate projection strategy.
    pub projection: CandidateProjection,
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self {
            prefix_reuse: PrefixReuse::None,
            scheduling: TaskScheduling::Serial,
            parallel_task_limit: 1,
            projection: CandidateProjection::FullVocabulary,
        }
    }
}

impl ExecutionPlan {
    /// Return whether the plan's quantitative scheduling fields are coherent.
    pub const fn scheduling_is_valid(self) -> bool {
        match self.scheduling {
            TaskScheduling::Serial => self.parallel_task_limit == 1,
            TaskScheduling::Parallel => self.parallel_task_limit >= 2,
        }
    }
}

/// Raw logits for one task. TurboJev core owns normalization/calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskScores {
    /// Question identifier matching a task in the input batch.
    pub id: String,
    /// Candidate logits in exactly the same order as the input candidate list.
    pub logits: Vec<f32>,
}

/// Optional backend counters. All values are data, never platform assumptions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BackendMetrics {
    /// Actual model tokens generated before candidate scoring (optional reasoning).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<usize>,
    /// Shared-state prefill time, when measured by the host runtime.
    pub prefill_seconds: Option<f64>,
    /// Candidate scoring time, when measured by the host runtime.
    pub scoring_seconds: Option<f64>,
    /// Total backend execution time, when measured by the host runtime.
    pub total_seconds: Option<f64>,
    /// Input tokens consumed, when known.
    pub input_tokens: Option<usize>,
    /// Number of parallel suffix batches actually submitted, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_batches: Option<usize>,
    /// Largest number of tasks submitted together, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_parallel_tasks: Option<usize>,
    /// Number of candidate output rows explicitly projected by a selective
    /// output-head implementation. `None` means the runtime did not expose the
    /// counter; zero is a valid measured value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_rows_projected: Option<usize>,
    /// Total candidate-code tokens scored by an exact multi-token path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_candidate_tokens: Option<usize>,
    /// Unique candidate-prefix states evaluated by trie/sequence scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_prefix_evaluations: Option<usize>,
    /// Candidate token steps avoided through prefix sharing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_token_steps_saved: Option<usize>,
    /// Number of independent semantic requests sharing one fused backend operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_request_count: Option<usize>,
    /// Number of independent model roots submitted by that fused operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_root_count: Option<usize>,
    /// Actual plan executed when a runtime must conservatively downgrade an
    /// optimization after candidate codebook discovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_plan: Option<ExecutionPlan>,
}

/// Raw backend result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendOutput {
    /// Resolved model identity or checkpoint name.
    pub model: Option<String>,
    /// One score row per input task.
    pub scores: Vec<TaskScores>,
    /// Backend metrics.
    pub metrics: BackendMetrics,
}

/// Feature flags used for capability negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BackendCapabilities {
    /// Backend can prefill shared state once and reuse it across tasks.
    pub shared_prefix: bool,
    /// Suffixes/tasks can be evaluated in one batch.
    pub parallel_suffixes: bool,
    /// Backend can project only candidate output-head rows.
    pub selective_output_head: bool,
    /// Backend can exactly score multi-token candidate codes.
    pub multi_token_candidates: bool,
    /// Backend executes model inference locally to the host application.
    ///
    /// `false` remains legal so applications may provide remote/custom adapters,
    /// but TurboJev itself never requires a remote service.
    pub local_inference: bool,
}

impl BackendCapabilities {
    /// Verify that a plan only requests declared capabilities and is internally
    /// coherent.
    pub fn supports(self, plan: ExecutionPlan) -> bool {
        if !plan.scheduling_is_valid() {
            return false;
        }
        let prefix_ok = plan.prefix_reuse == PrefixReuse::None || self.shared_prefix;
        let scheduling_ok = match plan.scheduling {
            TaskScheduling::Serial => true,
            TaskScheduling::Parallel => self.parallel_suffixes,
        };
        let projection_ok =
            plan.projection == CandidateProjection::FullVocabulary || self.selective_output_head;
        prefix_ok && scheduling_ok && projection_ok
    }
}

/// Quantitative runtime hints used only to bound an execution plan.
///
/// These values describe one loaded model/runtime instance and contain no OS or
/// device-brand assumptions. `None` means the backend cannot provide a safe
/// estimate, in which case the planner simply omits that constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BackendResourceHints {
    /// Hard backend/runtime limit for tasks in one parallel batch.
    pub max_parallel_tasks: Option<usize>,
    /// Conservative incremental memory estimate for one parallel task branch.
    pub estimated_parallel_task_bytes: Option<u64>,
}

impl BackendResourceHints {
    /// Normalize zero/invalid hints into "unknown" so they can never create a
    /// zero-sized execution batch.
    pub const fn normalized(self) -> Self {
        Self {
            max_parallel_tasks: match self.max_parallel_tasks {
                Some(0) => None,
                other => other,
            },
            estimated_parallel_task_bytes: match self.estimated_parallel_task_bytes {
                Some(0) => None,
                other => other,
            },
        }
    }
}

/// Inference backend contract.
///
/// This trait deliberately describes *semantic inference*, not a concrete model
/// format or OS API. A runtime adapter is responsible for prompt rendering,
/// tokenization and model execution, then returns candidate logits.
/// One field after schema validation and dependency resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldTask {
    pub name: String,
    pub state: Value,
    pub instructions: String,
    pub output_schema: Value,
    /// None requests bounded open-value generation; Some scores these values.
    pub values: Option<Vec<Value>>,
    pub thinking: bool,
    pub thinking_budget: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default)]
    pub seed: u64,
}

impl FieldTask {
    /// Finite fields reuse the existing tokenizer-independent decision contract.
    pub fn decision_batch(&self) -> Result<DecisionBatch, BackendError> {
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| BackendError::Unsupported("open-value generation".into()))?;
        Ok(DecisionBatch {
            state: self.state.clone(),
            tasks: vec![DecisionTask {
                id: self.name.clone(),
                intent: DecisionIntent::Criterion,
                instruction: Some(self.instructions.clone()),
                candidates: values
                    .iter()
                    .map(|value| Candidate {
                        key: value.to_string(),
                        description: serde_json::to_string(value).expect("JSON value serializes"),
                    })
                    .collect(),
            }],
        })
    }
}

/// Model evidence for one field; the portable core validates and types it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldOutput {
    /// Missing means logits; explicit JSON null is a generated nullable value.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_field_value"
    )]
    pub value: Option<Value>,
    pub logits: Vec<f32>,
    pub thinking: String,
    pub input_tokens: Option<usize>,
    pub generated_tokens: usize,
}

fn present_field_value<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

pub fn field_output_from_scores(
    task: &FieldTask,
    output: BackendOutput,
) -> Result<FieldOutput, BackendError> {
    if output.scores.len() != 1 || output.scores[0].id != task.name {
        return Err(BackendError::Inference(
            "field scorer returned missing, duplicate, or unexpected rows".into(),
        ));
    }
    Ok(FieldOutput {
        value: None,
        logits: output.scores.into_iter().next().unwrap().logits,
        thinking: String::new(),
        input_tokens: output.metrics.input_tokens,
        generated_tokens: output.metrics.generated_tokens.unwrap_or(0),
    })
}

pub trait ScoringBackend {
    /// Typed field inference. Existing scorers support finite fields by default.
    /// Open values or reasoning overrides require a concrete implementation.
    fn infer_field(&self, task: &FieldTask) -> Result<FieldOutput, BackendError> {
        if task.thinking {
            return Err(BackendError::Unsupported("per-field thinking".into()));
        }
        let batch = task.decision_batch()?;
        let output = self.score(&batch, ExecutionPlan::default())?;
        field_output_from_scores(task, output)
    }
    /// Stable backend implementation identifier.
    fn name(&self) -> &'static str;

    /// Runtime capabilities for the currently loaded model/runtime combination.
    fn capabilities(&self) -> BackendCapabilities;

    /// Quantitative limits/estimates used by portable scheduling.
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints::default()
    }

    /// Produce raw candidate logits for every task using the supplied plan.
    fn score(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError>;
}

/// Cross-request fused-scoring capabilities exposed by a backend.
///
/// This is deliberately separate from [`BackendCapabilities`]: parallel suffixes
/// describe multiple decision tasks *inside one semantic request*, whereas this
/// contract describes independent request roots entering one backend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FusedBackendCapabilities {
    /// Independent decision batches can be executed in one backend operation.
    pub fused_batch: bool,
    /// Hard maximum independent request roots accepted by one fused operation.
    pub max_fused_items: Option<usize>,
    /// Conservative additional bytes per fused request root, when known.
    pub estimated_fused_item_bytes: Option<u64>,
}

impl FusedBackendCapabilities {
    /// Conservative default for ordinary backends.
    pub const fn disabled() -> Self {
        Self {
            fused_batch: false,
            max_fused_items: None,
            estimated_fused_item_bytes: None,
        }
    }

    /// Validate capability combinations before a host advertises them.
    pub fn validate(self) -> Result<Self, BackendError> {
        if !self.fused_batch
            && (self.max_fused_items.is_some() || self.estimated_fused_item_bytes.is_some())
        {
            return Err(BackendError::InvalidInput(
                "fused backend limits require fusedBatch=true".to_owned(),
            ));
        }
        if matches!(self.max_fused_items, Some(0 | 1)) {
            return Err(BackendError::InvalidInput(
                "maxFusedItems must be at least two when fused batching is enabled".to_owned(),
            ));
        }
        if matches!(self.estimated_fused_item_bytes, Some(0)) {
            return Err(BackendError::InvalidInput(
                "estimatedFusedItemBytes must be greater than zero when present".to_owned(),
            ));
        }
        Ok(self)
    }
}

impl Default for FusedBackendCapabilities {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Optional cross-request scoring SPI.
///
/// Implementing this trait is the only portable-core path that may be described
/// as fused cross-request execution. Calling `score` repeatedly is not fusion.
/// Backends must preserve request order and return one [`BackendOutput`] for
/// every input [`DecisionBatch`].
pub trait FusedScoringBackend: ScoringBackend {
    /// Truthful capabilities for the loaded backend/model combination.
    fn fused_capabilities(&self) -> FusedBackendCapabilities;

    /// Return whether this specific prepared workload can use the fused path.
    /// Runtime capabilities may be shape-dependent (for example, single-token
    /// candidate codebooks or native sequence/token limits).
    fn can_score_fused(
        &self,
        _batches: &[DecisionBatch],
        _plans: &[ExecutionPlan],
    ) -> Result<bool, BackendError> {
        Ok(self.fused_capabilities().validate()?.fused_batch)
    }

    /// Score multiple independent decision batches in one backend operation.
    fn score_fused(
        &self,
        batches: &[DecisionBatch],
        plans: &[ExecutionPlan],
    ) -> Result<Vec<BackendOutput>, BackendError>;
}

/// Runtime-agnostic future returned by asynchronous backends.
///
/// The future deliberately has no `Send` bound. Browser/WASM and some mobile
/// runtimes own thread-affine host objects, so thread mobility is a host-level
/// policy rather than a TurboJev core requirement.
pub type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Asynchronous inference backend contract.
///
/// This is separate from [`ScoringBackend`] so synchronous native runtimes do
/// not pay for an async runtime, while browser/WebGPU/JS-hosted executors can
/// integrate without blocking or depending on Tokio/async-std.
pub trait AsyncScoringBackend {
    /// Async counterpart of typed field inference.
    fn infer_field_async<'a>(
        &'a self,
        task: &'a FieldTask,
    ) -> BackendFuture<'a, Result<FieldOutput, BackendError>> {
        Box::pin(async move {
            if task.thinking {
                return Err(BackendError::Unsupported("per-field thinking".into()));
            }
            let batch = task.decision_batch()?;
            field_output_from_scores(
                task,
                self.score_async(&batch, ExecutionPlan::default()).await?,
            )
        })
    }
    /// Stable backend implementation identifier.
    fn name(&self) -> &'static str;

    /// Runtime capabilities for the currently loaded model/runtime combination.
    fn capabilities(&self) -> BackendCapabilities;

    /// Quantitative limits/estimates used by portable scheduling.
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints::default()
    }

    /// Produce raw candidate logits asynchronously.
    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>>;
}

/// Marker for hosts that specifically require a backend to be shareable
/// across threads. TurboJev core deliberately does not impose this on all
/// backends because browser/WASM runtimes may own non-Send host handles.
pub trait ThreadSafeScoringBackend: ScoringBackend + Send + Sync {}

impl<T> ThreadSafeScoringBackend for T where T: ScoringBackend + Send + Sync {}

/// Error returned by inference backends.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// Backend is not initialized or has no model loaded.
    #[error("backend is not ready: {0}")]
    NotReady(String),
    /// The selected model/runtime cannot satisfy the requested operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// Tokenization/prompt preparation failed inside the backend.
    #[error("backend input preparation failed: {0}")]
    InvalidInput(String),
    /// Inference failed.
    #[error("backend inference failed: {0}")]
    Inference(String),
    /// Planner requested capabilities the backend did not advertise.
    #[error("backend does not support the requested execution plan")]
    UnsupportedPlan,
}
