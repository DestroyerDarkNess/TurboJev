#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable cross-request execution contracts for TurboJev servers.
//!
//! This crate does not create threads, read clocks, open sockets, or own model
//! sessions. It defines the capability negotiation and exact-parity rules a host
//! must satisfy before it may advertise fused cross-request execution.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_schema::{EvaluationRequest, EvaluationResponse};

/// Stable contract identifier for server execution capability metadata.
pub const CROSS_REQUEST_EXECUTION_CONTRACT: &str = "turbojev.cross-request-execution.v1";

/// Cross-request execution mode selected for one scheduler turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CrossRequestMode {
    /// Requests are evaluated independently. A host may still round-robin jobs.
    Serial,
    /// Multiple independent requests enter one backend batch operation.
    Fused,
}

impl CrossRequestMode {
    /// Stable response metadata value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial-round-robin",
            Self::Fused => "fused-batch",
        }
    }
}

/// Capabilities of a concrete server-side executor for one loaded model/runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CrossRequestCapabilities {
    /// Backend can evaluate independent TurboJev requests in one fused operation.
    pub fused_batch: bool,
    /// Backend can observe cancellation while an already-started operation runs.
    pub cooperative_cancellation: bool,
    /// Backend can enforce a deadline after execution has already started.
    pub hard_deadlines: bool,
    /// Maximum fused request count advertised by the backend, when known.
    pub max_fused_items: Option<usize>,
    /// Conservative incremental memory estimate per fused request root.
    pub estimated_fused_item_bytes: Option<u64>,
}

impl CrossRequestCapabilities {
    /// Conservative capabilities for a normal thread-affine session.
    pub const fn serial_only() -> Self {
        Self {
            fused_batch: false,
            cooperative_cancellation: false,
            hard_deadlines: false,
            max_fused_items: None,
            estimated_fused_item_bytes: None,
        }
    }

    /// Validate truthful internal combinations.
    pub fn validate(self) -> Result<Self, ExecutionContractError> {
        if matches!(self.max_fused_items, Some(0)) {
            return Err(ExecutionContractError::InvalidCapabilities(
                "maxFusedItems must be greater than zero when present".to_owned(),
            ));
        }
        if !self.fused_batch
            && (self.max_fused_items.is_some() || self.estimated_fused_item_bytes.is_some())
        {
            return Err(ExecutionContractError::InvalidCapabilities(
                "fused limits require fusedBatch=true".to_owned(),
            ));
        }
        if matches!(self.estimated_fused_item_bytes, Some(0)) {
            return Err(ExecutionContractError::InvalidCapabilities(
                "estimatedFusedItemBytes must be greater than zero when present".to_owned(),
            ));
        }
        Ok(self)
    }
}

impl Default for CrossRequestCapabilities {
    fn default() -> Self {
        Self::serial_only()
    }
}

/// Host policy controlling whether fused batching is permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum BatchPlanningPolicy {
    /// Use fused execution only when the executor explicitly advertises it.
    #[default]
    Auto,
    /// Force the independent reference path.
    ReferenceSerial,
}

/// Concrete cross-request plan for one scheduler turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CrossRequestPlan {
    pub mode: CrossRequestMode,
    /// Maximum requests executed by the turn before fairness rotates to another job.
    pub turn_items: usize,
}

/// Host-side fused-batch resource budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CrossRequestBudget {
    pub max_fused_items: usize,
    pub memory_budget_bytes: Option<u64>,
}

impl CrossRequestBudget {
    pub const fn items(max_fused_items: usize) -> Self {
        Self {
            max_fused_items,
            memory_budget_bytes: None,
        }
    }
}

/// Compatibility planner using only a host item-count limit.
pub fn plan_cross_request(
    policy: BatchPlanningPolicy,
    capabilities: CrossRequestCapabilities,
    remaining_items: usize,
    fairness_quantum: usize,
    host_max_fused_items: usize,
) -> Result<CrossRequestPlan, ExecutionContractError> {
    plan_cross_request_with_budget(
        policy,
        capabilities,
        remaining_items,
        fairness_quantum,
        CrossRequestBudget::items(host_max_fused_items),
    )
}

/// Plan one scheduler turn from truthful capabilities and host resource limits.
pub fn plan_cross_request_with_budget(
    policy: BatchPlanningPolicy,
    capabilities: CrossRequestCapabilities,
    remaining_items: usize,
    fairness_quantum: usize,
    budget: CrossRequestBudget,
) -> Result<CrossRequestPlan, ExecutionContractError> {
    let capabilities = capabilities.validate()?;
    if remaining_items == 0 {
        return Err(ExecutionContractError::EmptyWork);
    }
    if fairness_quantum == 0 || budget.max_fused_items == 0 {
        return Err(ExecutionContractError::InvalidLimits(
            "fairnessQuantum and maxFusedItems must be greater than zero".to_owned(),
        ));
    }

    let maximum_turn = remaining_items.min(fairness_quantum);
    if policy == BatchPlanningPolicy::ReferenceSerial || !capabilities.fused_batch {
        return Ok(CrossRequestPlan {
            mode: CrossRequestMode::Serial,
            turn_items: maximum_turn,
        });
    }

    let backend_max = capabilities.max_fused_items.unwrap_or(usize::MAX);
    let mut fused_items = maximum_turn.min(budget.max_fused_items).min(backend_max);
    if let (Some(memory_budget), Some(bytes_per_item)) = (
        budget.memory_budget_bytes,
        capabilities.estimated_fused_item_bytes,
    ) {
        let by_memory = memory_budget / bytes_per_item;
        let by_memory = usize::try_from(by_memory).unwrap_or(usize::MAX);
        fused_items = fused_items.min(by_memory);
    }
    if fused_items < 2 {
        return Ok(CrossRequestPlan {
            mode: CrossRequestMode::Serial,
            turn_items: maximum_turn.min(1),
        });
    }

    Ok(CrossRequestPlan {
        mode: CrossRequestMode::Fused,
        turn_items: fused_items,
    })
}

/// Host-provided cancellation/deadline view. Portable executors may poll it,
/// but no portable crate owns a clock or thread primitive.
pub trait ExecutionControl {
    fn is_cancelled(&self) -> bool;
    fn deadline_expired(&self) -> bool;
}

/// No-op control used by conformance/reference callers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Uncontrolled;

impl ExecutionControl for Uncontrolled {
    fn is_cancelled(&self) -> bool {
        false
    }
    fn deadline_expired(&self) -> bool {
        false
    }
}

/// SPI implemented by host/runtime adapters that want cross-request execution.
pub trait CrossRequestExecutor {
    type Error;

    fn capabilities(&self) -> CrossRequestCapabilities;

    fn execute_one(
        &mut self,
        request: &EvaluationRequest,
        control: &dyn ExecutionControl,
    ) -> Result<EvaluationResponse, Self::Error>;

    /// Return whether this exact scheduler slice is eligible for fused execution.
    /// The default follows the global capability; shape-aware runtimes should
    /// override it so unsupported request groups fall back before execution.
    fn can_execute_fused(
        &mut self,
        _requests: &[EvaluationRequest],
        _control: &dyn ExecutionControl,
    ) -> Result<bool, Self::Error> {
        Ok(self.capabilities().fused_batch)
    }

    /// Execute multiple independent requests in one backend operation.
    ///
    /// Implementations must preserve input order and exact public semantics.
    /// This method is called only when `fused_batch=true` was advertised.
    fn execute_fused(
        &mut self,
        requests: &[EvaluationRequest],
        control: &dyn ExecutionControl,
    ) -> Result<Vec<EvaluationResponse>, Self::Error>;
}

/// Execute one already-planned scheduler turn.
pub fn execute_turn<E: CrossRequestExecutor>(
    executor: &mut E,
    requests: &[EvaluationRequest],
    plan: CrossRequestPlan,
    control: &dyn ExecutionControl,
) -> Result<Vec<EvaluationResponse>, TurnError<E::Error>> {
    if requests.is_empty() {
        return Err(TurnError::Contract(ExecutionContractError::EmptyWork));
    }
    if plan.turn_items == 0 || plan.turn_items > requests.len() {
        return Err(TurnError::Contract(ExecutionContractError::InvalidPlan));
    }
    if control.is_cancelled() {
        return Err(TurnError::Cancelled);
    }
    if control.deadline_expired() {
        return Err(TurnError::DeadlineExceeded);
    }

    match plan.mode {
        CrossRequestMode::Serial => {
            let mut responses = Vec::with_capacity(plan.turn_items);
            for request in &requests[..plan.turn_items] {
                if control.is_cancelled() {
                    return Err(TurnError::Cancelled);
                }
                if control.deadline_expired() {
                    return Err(TurnError::DeadlineExceeded);
                }
                responses.push(
                    executor
                        .execute_one(request, control)
                        .map_err(TurnError::Executor)?,
                );
            }
            Ok(responses)
        }
        CrossRequestMode::Fused => {
            let capabilities = executor
                .capabilities()
                .validate()
                .map_err(TurnError::Contract)?;
            if !capabilities.fused_batch {
                return Err(TurnError::Contract(
                    ExecutionContractError::UnsupportedFusedBatch,
                ));
            }
            let responses = executor
                .execute_fused(&requests[..plan.turn_items], control)
                .map_err(TurnError::Executor)?;
            if responses.len() != plan.turn_items {
                return Err(TurnError::Contract(
                    ExecutionContractError::ResponseCountMismatch {
                        expected: plan.turn_items,
                        actual: responses.len(),
                    },
                ));
            }
            Ok(responses)
        }
    }
}

/// Verify exact fused-vs-reference response parity before a backend may claim
/// fused cross-request execution in conformance evidence.
pub fn verify_fused_parity<E>(
    executor: &mut E,
    requests: &[EvaluationRequest],
) -> Result<(), FusedParityError<E::Error>>
where
    E: CrossRequestExecutor,
{
    if requests.len() < 2 {
        return Err(FusedParityError::Contract(
            ExecutionContractError::InvalidLimits(
                "fused parity requires at least two requests".to_owned(),
            ),
        ));
    }
    let capabilities = executor
        .capabilities()
        .validate()
        .map_err(FusedParityError::Contract)?;
    if !capabilities.fused_batch {
        return Err(FusedParityError::Contract(
            ExecutionContractError::UnsupportedFusedBatch,
        ));
    }
    let control = Uncontrolled;
    let mut reference = Vec::with_capacity(requests.len());
    for request in requests {
        reference.push(
            executor
                .execute_one(request, &control)
                .map_err(FusedParityError::Executor)?,
        );
    }
    let fused = executor
        .execute_fused(requests, &control)
        .map_err(FusedParityError::Executor)?;
    if fused.len() != reference.len() {
        return Err(FusedParityError::Contract(
            ExecutionContractError::ResponseCountMismatch {
                expected: reference.len(),
                actual: fused.len(),
            },
        ));
    }
    for (index, (left, right)) in reference.iter().zip(&fused).enumerate() {
        if left != right {
            return Err(FusedParityError::Mismatch { index });
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum ExecutionContractError {
    #[error("cross-request capabilities are invalid: {0}")]
    InvalidCapabilities(String),
    #[error("cross-request execution limits are invalid: {0}")]
    InvalidLimits(String),
    #[error("cross-request work must contain at least one item")]
    EmptyWork,
    #[error("cross-request execution plan is invalid for the supplied work")]
    InvalidPlan,
    #[error("executor does not advertise fused cross-request batching")]
    UnsupportedFusedBatch,
    #[error("executor returned {actual} responses; expected {expected}")]
    ResponseCountMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Error)]
pub enum TurnError<E> {
    #[error(transparent)]
    Contract(#[from] ExecutionContractError),
    #[error("request was cancelled")]
    Cancelled,
    #[error("request deadline expired")]
    DeadlineExceeded,
    #[error("executor failed")]
    Executor(E),
}

#[derive(Debug, Error)]
pub enum FusedParityError<E> {
    #[error(transparent)]
    Contract(#[from] ExecutionContractError),
    #[error("executor failed")]
    Executor(E),
    #[error("fused response differs from serial reference at request index {index}")]
    Mismatch { index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_schema::EvaluationResponse;

    #[derive(Default)]
    struct EchoExecutor;

    impl CrossRequestExecutor for EchoExecutor {
        type Error = &'static str;

        fn capabilities(&self) -> CrossRequestCapabilities {
            CrossRequestCapabilities {
                fused_batch: true,
                cooperative_cancellation: false,
                hard_deadlines: false,
                max_fused_items: Some(8),
                estimated_fused_item_bytes: Some(1024),
            }
        }

        fn execute_one(
            &mut self,
            request: &EvaluationRequest,
            _control: &dyn ExecutionControl,
        ) -> Result<EvaluationResponse, Self::Error> {
            Ok(response_for(request))
        }

        fn execute_fused(
            &mut self,
            requests: &[EvaluationRequest],
            _control: &dyn ExecutionControl,
        ) -> Result<Vec<EvaluationResponse>, Self::Error> {
            Ok(requests.iter().map(response_for).collect())
        }
    }

    fn request(id: &str) -> EvaluationRequest {
        serde_json::from_value(serde_json::json!({
            "state": id,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "route",
                    "criteria": {"a": "A", "b": "B"}
                }
            }
        }))
        .unwrap()
    }

    fn response_for(_request: &EvaluationRequest) -> EvaluationResponse {
        serde_json::from_value(serde_json::json!({
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "a",
                    "confidence": 0.75,
                    "probabilities": {"a": 0.75, "b": 0.25}
                }
            },
            "usage": {
                "decisions": 1,
                "generated_tokens": 0
            },
            "_turbojev": {
                "backend": "echo",
                "calibration": "softmax",
                "calibrated": false,
                "confidence_method": "candidate-argmax",
                "contract": "turbojev.response.v1",
                "execution": {
                    "prefix_reuse": "none",
                    "scheduling": "serial",
                    "parallel_task_limit": 1,
                    "projection": "full-vocabulary"
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn serial_executor_never_gets_a_false_fused_plan() {
        let plan = plan_cross_request(
            BatchPlanningPolicy::Auto,
            CrossRequestCapabilities::serial_only(),
            8,
            4,
            8,
        )
        .unwrap();
        assert_eq!(plan.mode, CrossRequestMode::Serial);
        assert_eq!(plan.turn_items, 4);
    }

    #[test]
    fn fused_plan_is_bounded_by_backend_host_and_fairness() {
        let caps = CrossRequestCapabilities {
            fused_batch: true,
            cooperative_cancellation: false,
            hard_deadlines: false,
            max_fused_items: Some(6),
            estimated_fused_item_bytes: None,
        };
        let plan = plan_cross_request(BatchPlanningPolicy::Auto, caps, 20, 5, 4).unwrap();
        assert_eq!(plan.mode, CrossRequestMode::Fused);
        assert_eq!(plan.turn_items, 4);
    }

    #[test]
    fn fused_path_must_match_serial_reference_exactly() {
        let mut executor = EchoExecutor;
        verify_fused_parity(&mut executor, &[request("a"), request("b")]).unwrap();
    }

    #[test]
    fn fused_plan_respects_runtime_memory_estimate() {
        let caps = CrossRequestCapabilities {
            fused_batch: true,
            cooperative_cancellation: false,
            hard_deadlines: false,
            max_fused_items: Some(16),
            estimated_fused_item_bytes: Some(4 * 1024),
        };
        let plan = plan_cross_request_with_budget(
            BatchPlanningPolicy::Auto,
            caps,
            20,
            10,
            CrossRequestBudget {
                max_fused_items: 12,
                memory_budget_bytes: Some(12 * 1024),
            },
        )
        .unwrap();
        assert_eq!(plan.mode, CrossRequestMode::Fused);
        assert_eq!(plan.turn_items, 3);
    }
}
