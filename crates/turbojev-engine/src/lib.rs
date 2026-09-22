#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Backend-neutral execution planning.
//!
//! The planner uses only semantic workload shape, declared backend capabilities,
//! quantitative backend hints and caller-provided portable budgets. It never
//! inspects the operating system, device brand, model family or accelerator API.

use turbojev_backend::{
    BackendCapabilities, BackendResourceHints, CandidateProjection, DecisionBatch, ExecutionPlan,
    PrefixReuse, TaskScheduling,
};

/// Policy controlling how aggressively TurboJev uses backend optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanningPolicy {
    /// Choose the fastest declared portable path while preserving semantics.
    #[default]
    Auto,
    /// Recompute every task independently; useful as a correctness reference.
    Reference,
    /// Reuse the shared prefix when available but keep task execution serial.
    DeterministicReuse,
}

/// Portable resource budget applied to automatic scheduling.
///
/// The budget describes quantities only. It contains no CUDA/Metal/WebGPU/OS
/// assumptions, so the same API can be used by desktop, mobile and browser
/// bindings. Unknown values are represented by `None` and never guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionBudget {
    /// Caller-requested maximum tasks in one parallel suffix batch.
    pub max_parallel_tasks: Option<usize>,
    /// Memory budget available to parallel task branches, in bytes.
    pub memory_budget_bytes: Option<u64>,
}

impl ExecutionBudget {
    /// Construct an unconstrained budget.
    pub const fn unlimited() -> Self {
        Self {
            max_parallel_tasks: None,
            memory_budget_bytes: None,
        }
    }

    /// Apply an explicit task-count cap.
    pub const fn with_max_parallel_tasks(mut self, value: usize) -> Self {
        self.max_parallel_tasks = Some(value);
        self
    }

    /// Apply an explicit memory budget in bytes.
    pub const fn with_memory_budget_bytes(mut self, value: u64) -> Self {
        self.memory_budget_bytes = Some(value);
        self
    }
}

/// Stateless execution planner plus immutable planning configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlanner {
    policy: PlanningPolicy,
    budget: ExecutionBudget,
}

impl Default for ExecutionPlanner {
    fn default() -> Self {
        Self {
            policy: PlanningPolicy::Auto,
            budget: ExecutionBudget::unlimited(),
        }
    }
}

impl ExecutionPlanner {
    /// Create a planner with the requested policy and no caller resource cap.
    pub const fn new(policy: PlanningPolicy) -> Self {
        Self {
            policy,
            budget: ExecutionBudget::unlimited(),
        }
    }

    /// Create a planner with an explicit policy and portable resource budget.
    pub const fn with_budget(policy: PlanningPolicy, budget: ExecutionBudget) -> Self {
        Self { policy, budget }
    }

    /// Return the configured policy.
    pub const fn policy(self) -> PlanningPolicy {
        self.policy
    }

    /// Return the configured portable resource budget.
    pub const fn budget(self) -> ExecutionBudget {
        self.budget
    }

    /// Replace the portable resource budget.
    pub const fn set_budget(mut self, budget: ExecutionBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Compatibility helper for backends with no quantitative hints.
    pub fn plan(self, capabilities: BackendCapabilities, batch: &DecisionBatch) -> ExecutionPlan {
        self.plan_with_hints(capabilities, BackendResourceHints::default(), batch)
    }

    /// Create an execution plan from workload shape, backend capabilities,
    /// backend resource hints and caller budget.
    pub fn plan_with_hints(
        self,
        capabilities: BackendCapabilities,
        resource_hints: BackendResourceHints,
        batch: &DecisionBatch,
    ) -> ExecutionPlan {
        let task_count = batch.tasks.len();
        let multiple_tasks = task_count > 1;

        let prefix_reuse = match self.policy {
            PlanningPolicy::Reference => PrefixReuse::None,
            PlanningPolicy::Auto | PlanningPolicy::DeterministicReuse
                if multiple_tasks && capabilities.shared_prefix =>
            {
                PrefixReuse::Shared
            }
            PlanningPolicy::Auto | PlanningPolicy::DeterministicReuse => PrefixReuse::None,
        };

        let requested_parallel_limit =
            parallel_task_limit(task_count, self.budget, resource_hints.normalized());
        let scheduling = match self.policy {
            PlanningPolicy::Auto
                if multiple_tasks
                    && capabilities.parallel_suffixes
                    && requested_parallel_limit >= 2 =>
            {
                TaskScheduling::Parallel
            }
            PlanningPolicy::Auto
            | PlanningPolicy::Reference
            | PlanningPolicy::DeterministicReuse => TaskScheduling::Serial,
        };
        let parallel_task_limit = match scheduling {
            TaskScheduling::Serial => 1,
            TaskScheduling::Parallel => requested_parallel_limit,
        };

        let projection = match self.policy {
            PlanningPolicy::Reference => CandidateProjection::FullVocabulary,
            PlanningPolicy::Auto | PlanningPolicy::DeterministicReuse
                if capabilities.selective_output_head =>
            {
                CandidateProjection::CandidatesOnly
            }
            PlanningPolicy::Auto | PlanningPolicy::DeterministicReuse => {
                CandidateProjection::FullVocabulary
            }
        };

        ExecutionPlan {
            prefix_reuse,
            scheduling,
            parallel_task_limit,
            projection,
        }
    }
}

fn parallel_task_limit(
    task_count: usize,
    budget: ExecutionBudget,
    hints: BackendResourceHints,
) -> usize {
    if task_count <= 1 {
        return 1;
    }

    let mut limit = task_count;
    if let Some(value) = budget.max_parallel_tasks {
        limit = limit.min(value.max(1));
    }
    if let Some(value) = hints.max_parallel_tasks {
        limit = limit.min(value.max(1));
    }
    if let (Some(memory_budget), Some(bytes_per_task)) = (
        budget.memory_budget_bytes,
        hints.estimated_parallel_task_bytes,
    ) {
        if bytes_per_task > 0 {
            let by_memory = memory_budget / bytes_per_task;
            let by_memory = usize::try_from(by_memory).unwrap_or(usize::MAX).max(1);
            limit = limit.min(by_memory);
        }
    }
    limit.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_backend::{Candidate, DecisionIntent, DecisionTask};

    fn batch(tasks: usize) -> DecisionBatch {
        DecisionBatch {
            state: serde_json::json!({"shared": true}),
            tasks: (0..tasks)
                .map(|index| DecisionTask {
                    id: format!("q{index}"),
                    intent: DecisionIntent::Criterion,
                    instruction: Some("judge".into()),
                    candidates: vec![
                        Candidate {
                            key: "no".into(),
                            description: "no".into(),
                        },
                        Candidate {
                            key: "yes".into(),
                            description: "yes".into(),
                        },
                    ],
                })
                .collect(),
        }
    }

    fn all_capabilities() -> BackendCapabilities {
        BackendCapabilities {
            shared_prefix: true,
            parallel_suffixes: true,
            selective_output_head: true,
            multi_token_candidates: true,
            local_inference: true,
        }
    }

    #[test]
    fn auto_uses_all_semantics_preserving_optimizations() {
        let plan = ExecutionPlanner::default().plan(all_capabilities(), &batch(4));
        assert_eq!(plan.prefix_reuse, PrefixReuse::Shared);
        assert_eq!(plan.scheduling, TaskScheduling::Parallel);
        assert_eq!(plan.parallel_task_limit, 4);
        assert_eq!(plan.projection, CandidateProjection::CandidatesOnly);
    }

    #[test]
    fn reference_mode_disables_reuse_and_parallelism() {
        let plan =
            ExecutionPlanner::new(PlanningPolicy::Reference).plan(all_capabilities(), &batch(4));
        assert_eq!(plan.prefix_reuse, PrefixReuse::None);
        assert_eq!(plan.scheduling, TaskScheduling::Serial);
        assert_eq!(plan.parallel_task_limit, 1);
        assert_eq!(plan.projection, CandidateProjection::FullVocabulary);
    }

    #[test]
    fn deterministic_reuse_uses_shared_prefix_but_stays_serial() {
        let capabilities = BackendCapabilities {
            shared_prefix: true,
            parallel_suffixes: true,
            selective_output_head: false,
            multi_token_candidates: false,
            local_inference: true,
        };
        let plan =
            ExecutionPlanner::new(PlanningPolicy::DeterministicReuse).plan(capabilities, &batch(4));
        assert_eq!(plan.prefix_reuse, PrefixReuse::Shared);
        assert_eq!(plan.scheduling, TaskScheduling::Serial);
        assert_eq!(plan.parallel_task_limit, 1);
        assert_eq!(plan.projection, CandidateProjection::FullVocabulary);
    }

    #[test]
    fn backend_parallel_limit_bounds_auto_batching() {
        let hints = BackendResourceHints {
            max_parallel_tasks: Some(3),
            estimated_parallel_task_bytes: None,
        };
        let plan =
            ExecutionPlanner::default().plan_with_hints(all_capabilities(), hints, &batch(9));
        assert_eq!(plan.scheduling, TaskScheduling::Parallel);
        assert_eq!(plan.parallel_task_limit, 3);
    }

    #[test]
    fn caller_memory_budget_bounds_auto_batching() {
        let budget = ExecutionBudget::unlimited().with_memory_budget_bytes(16 * 1024 * 1024);
        let hints = BackendResourceHints {
            max_parallel_tasks: Some(32),
            estimated_parallel_task_bytes: Some(4 * 1024 * 1024),
        };
        let plan = ExecutionPlanner::with_budget(PlanningPolicy::Auto, budget).plan_with_hints(
            all_capabilities(),
            hints,
            &batch(20),
        );
        assert_eq!(plan.scheduling, TaskScheduling::Parallel);
        assert_eq!(plan.parallel_task_limit, 4);
    }

    #[test]
    fn budget_that_allows_one_task_falls_back_to_serial() {
        let budget = ExecutionBudget::unlimited().with_memory_budget_bytes(1024);
        let hints = BackendResourceHints {
            max_parallel_tasks: Some(32),
            estimated_parallel_task_bytes: Some(4096),
        };
        let plan = ExecutionPlanner::with_budget(PlanningPolicy::Auto, budget).plan_with_hints(
            all_capabilities(),
            hints,
            &batch(20),
        );
        assert_eq!(plan.scheduling, TaskScheduling::Serial);
        assert_eq!(plan.parallel_task_limit, 1);
        assert_eq!(plan.prefix_reuse, PrefixReuse::Shared);
    }

    #[test]
    fn explicit_task_budget_and_backend_limit_take_the_tighter_value() {
        let budget = ExecutionBudget::unlimited().with_max_parallel_tasks(6);
        let hints = BackendResourceHints {
            max_parallel_tasks: Some(4),
            estimated_parallel_task_bytes: None,
        };
        let plan = ExecutionPlanner::with_budget(PlanningPolicy::Auto, budget).plan_with_hints(
            all_capabilities(),
            hints,
            &batch(20),
        );
        assert_eq!(plan.parallel_task_limit, 4);
    }
}
