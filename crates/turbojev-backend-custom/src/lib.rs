#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Adapter for application-provided model runtimes.
//!
//! This crate is intentionally format- and platform-agnostic. It allows a host
//! application to connect any model executor that can satisfy TurboJev's
//! candidate-scoring contract without waiting for a first-party adapter crate.

use std::future::Future;

use turbojev_backend::{
    AsyncScoringBackend, BackendCapabilities, BackendError, BackendFuture, BackendOutput,
    DecisionBatch, ExecutionPlan, ScoringBackend,
};

/// A backend whose scoring implementation is supplied by the application.
pub struct CustomBackend<F> {
    capabilities: BackendCapabilities,
    scorer: F,
}

impl<F> CustomBackend<F> {
    /// Construct a custom backend from declared capabilities and a scoring callback.
    pub fn new(capabilities: BackendCapabilities, scorer: F) -> Self {
        Self {
            capabilities,
            scorer,
        }
    }
}

impl<F> ScoringBackend for CustomBackend<F>
where
    F: Fn(&DecisionBatch, ExecutionPlan) -> Result<BackendOutput, BackendError>,
{
    fn name(&self) -> &'static str {
        "custom"
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    fn score(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        if !self.capabilities.supports(plan) {
            return Err(BackendError::UnsupportedPlan);
        }
        (self.scorer)(batch, plan)
    }
}

/// A custom backend whose scoring callback is naturally asynchronous.
///
/// This is useful for browser/WebGPU/JavaScript bridges and host runtimes that
/// already expose futures/promises. The callback controls scheduling; TurboJev
/// does not require Tokio or any other executor.
pub struct AsyncCustomBackend<F> {
    capabilities: BackendCapabilities,
    scorer: F,
}

impl<F> AsyncCustomBackend<F> {
    /// Construct an asynchronous custom backend.
    pub fn new(capabilities: BackendCapabilities, scorer: F) -> Self {
        Self {
            capabilities,
            scorer,
        }
    }
}

impl<F, Fut> AsyncScoringBackend for AsyncCustomBackend<F>
where
    F: Fn(DecisionBatch, ExecutionPlan) -> Fut,
    Fut: Future<Output = Result<BackendOutput, BackendError>> + 'static,
{
    fn name(&self) -> &'static str {
        "custom-async"
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>> {
        if !self.capabilities.supports(plan) {
            return Box::pin(std::future::ready(Err(BackendError::UnsupportedPlan)));
        }
        let owned_batch = batch.clone();
        Box::pin((self.scorer)(owned_batch, plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_backend::{BackendMetrics, Candidate, DecisionIntent, DecisionTask, TaskScores};

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
    fn arbitrary_runtime_can_be_adapted_without_core_changes() {
        let capabilities = BackendCapabilities::default();
        let backend = CustomBackend::new(capabilities, |batch: &DecisionBatch, _| {
            Ok(BackendOutput {
                model: Some("host-owned-open-model".into()),
                scores: batch
                    .tasks
                    .iter()
                    .map(|task| TaskScores {
                        id: task.id.clone(),
                        logits: (0..task.candidates.len()).map(|i| i as f32).collect(),
                    })
                    .collect(),
                metrics: BackendMetrics::default(),
            })
        });

        let batch = DecisionBatch {
            state: serde_json::json!("state"),
            tasks: vec![DecisionTask {
                id: "route".into(),
                intent: DecisionIntent::Criterion,
                instruction: Some("route".into()),
                candidates: vec![
                    Candidate {
                        key: "a".into(),
                        description: "a".into(),
                    },
                    Candidate {
                        key: "b".into(),
                        description: "b".into(),
                    },
                ],
            }],
        };

        let output = backend.score(&batch, ExecutionPlan::default()).unwrap();
        assert_eq!(output.model.as_deref(), Some("host-owned-open-model"));
        assert_eq!(output.scores[0].logits, vec![0.0, 1.0]);
    }

    #[test]
    fn arbitrary_async_runtime_can_be_adapted_without_executor_dependency() {
        let backend =
            AsyncCustomBackend::new(BackendCapabilities::default(), |batch: DecisionBatch, _| {
                std::future::ready(Ok(BackendOutput {
                    model: Some("host-owned-async-model".to_owned()),
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
                    metrics: BackendMetrics::default(),
                }))
            });
        let batch = DecisionBatch {
            state: serde_json::json!("state"),
            tasks: vec![DecisionTask {
                id: "route".to_owned(),
                intent: DecisionIntent::Criterion,
                instruction: Some("route".to_owned()),
                candidates: vec![
                    Candidate {
                        key: "a".to_owned(),
                        description: "a".to_owned(),
                    },
                    Candidate {
                        key: "b".to_owned(),
                        description: "b".to_owned(),
                    },
                ],
            }],
        };
        let output = block_on(backend.score_async(&batch, ExecutionPlan::default())).unwrap();
        assert_eq!(output.model.as_deref(), Some("host-owned-async-model"));
    }
}
