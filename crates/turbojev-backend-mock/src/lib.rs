#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Deterministic backend for unit tests, examples, binding development and
//! conformance suites. It does not pretend to be a language model.

use indexmap::IndexMap;
use turbojev_backend::{
    AsyncScoringBackend, BackendCapabilities, BackendError, BackendFuture, BackendMetrics,
    BackendOutput, DecisionBatch, ExecutionPlan, FusedBackendCapabilities, FusedScoringBackend,
    ScoringBackend, TaskScores,
};

/// Builder for a deterministic fixture backend.
#[derive(Debug, Default)]
pub struct FixtureBackendBuilder {
    model: Option<String>,
    fixtures: IndexMap<String, Vec<f32>>,
}

impl FixtureBackendBuilder {
    /// Set the model string reported in results.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Configure exact logits for one question identifier.
    pub fn with_logits(mut self, question_id: impl Into<String>, logits: Vec<f32>) -> Self {
        self.fixtures.insert(question_id.into(), logits);
        self
    }

    /// Build the backend.
    pub fn build(self) -> FixtureBackend {
        FixtureBackend {
            model: self.model.or_else(|| Some("turbojev-fixture".to_owned())),
            fixtures: self.fixtures,
        }
    }
}

/// Exact-score deterministic backend.
#[derive(Debug, Clone)]
pub struct FixtureBackend {
    model: Option<String>,
    fixtures: IndexMap<String, Vec<f32>>,
}

impl FixtureBackend {
    /// Start configuring fixtures.
    pub fn builder() -> FixtureBackendBuilder {
        FixtureBackendBuilder::default()
    }
}

impl ScoringBackend for FixtureBackend {
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            shared_prefix: true,
            parallel_suffixes: true,
            selective_output_head: true,
            multi_token_candidates: true,
            local_inference: true,
        }
    }

    fn score(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        if !ScoringBackend::capabilities(self).supports(plan) {
            return Err(BackendError::UnsupportedPlan);
        }
        let mut scores = Vec::with_capacity(batch.tasks.len());

        for task in &batch.tasks {
            let logits = self.fixtures.get(&task.id).ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "fixture backend has no logits for task {:?}",
                    task.id
                ))
            })?;

            if logits.len() != task.candidates.len() {
                return Err(BackendError::InvalidInput(format!(
                    "fixture {:?} has {} logits but task has {} candidates",
                    task.id,
                    logits.len(),
                    task.candidates.len()
                )));
            }

            scores.push(TaskScores {
                id: task.id.clone(),
                logits: logits.clone(),
            });
        }

        Ok(BackendOutput {
            model: self.model.clone(),
            scores,
            metrics: BackendMetrics::default(),
        })
    }
}


impl FusedScoringBackend for FixtureBackend {
    fn fused_capabilities(&self) -> FusedBackendCapabilities {
        FusedBackendCapabilities {
            fused_batch: true,
            max_fused_items: Some(64),
            estimated_fused_item_bytes: None,
        }
    }

    fn score_fused(
        &self,
        batches: &[DecisionBatch],
        plans: &[ExecutionPlan],
    ) -> Result<Vec<BackendOutput>, BackendError> {
        let capabilities = self.fused_capabilities().validate()?;
        if batches.len() < 2 {
            return Err(BackendError::InvalidInput(
                "fixture fused scoring requires at least two independent batches".to_owned(),
            ));
        }
        if batches.len() != plans.len() {
            return Err(BackendError::InvalidInput(
                "fused batch/plan counts must match".to_owned(),
            ));
        }
        if let Some(maximum) = capabilities.max_fused_items {
            if batches.len() > maximum {
                return Err(BackendError::InvalidInput(format!(
                    "fixture fused scoring received {} batches; maximum is {}",
                    batches.len(), maximum
                )));
            }
        }

        // This is one explicit backend batch operation used only for deterministic
        // conformance. It does not claim to be a language-model runtime.
        batches
            .iter()
            .zip(plans)
            .map(|(batch, plan)| ScoringBackend::score(self, batch, *plan))
            .collect()
    }
}

impl AsyncScoringBackend for FixtureBackend {
    fn name(&self) -> &'static str {
        ScoringBackend::name(self)
    }

    fn capabilities(&self) -> BackendCapabilities {
        ScoringBackend::capabilities(self)
    }

    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>> {
        Box::pin(std::future::ready(ScoringBackend::score(self, batch, plan)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_backend::{Candidate, DecisionIntent, DecisionTask};

    #[test]
    fn returns_exact_configured_logits() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![1.0, 3.0])
            .build();
        let batch = DecisionBatch {
            state: serde_json::json!("state"),
            tasks: vec![DecisionTask {
                id: "route".into(),
                intent: DecisionIntent::Criterion,
                instruction: Some("route".into()),
                candidates: vec![
                    Candidate { key: "a".into(), description: "a".into() },
                    Candidate { key: "b".into(), description: "b".into() },
                ],
            }],
        };

        let output = backend.score(&batch, ExecutionPlan::default()).unwrap();
        assert_eq!(output.scores[0].logits, vec![1.0, 3.0]);
    }

    #[test]
    fn fused_fixture_preserves_request_order() {
        let backend = FixtureBackend::builder()
            .with_logits("route", vec![1.0, 3.0])
            .build();
        let make_batch = |state: &str| DecisionBatch {
            state: serde_json::json!(state),
            tasks: vec![DecisionTask {
                id: "route".into(),
                intent: DecisionIntent::Criterion,
                instruction: Some("route".into()),
                candidates: vec![
                    Candidate { key: "a".into(), description: "a".into() },
                    Candidate { key: "b".into(), description: "b".into() },
                ],
            }],
        };
        let batches = vec![make_batch("one"), make_batch("two")];
        let plans = vec![ExecutionPlan::default(), ExecutionPlan::default()];
        let outputs = backend.score_fused(&batches, &plans).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].scores[0].logits, vec![1.0, 3.0]);
        assert_eq!(outputs[1].scores[0].logits, vec![1.0, 3.0]);
    }

}
