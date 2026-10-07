//! Structural and numerical regressions, not real-model accuracy benchmarks.
use indexmap::IndexMap;
use serde_json::json;
use std::cell::RefCell;
use turbojev_backend::{
    AsyncScoringBackend, BackendCapabilities, BackendError, BackendFuture, BackendOutput,
    DecisionBatch, DecisionTask, ExecutionPlan, ScoringBackend, TaskScores,
};
use turbojev_calibration::TemperatureCalibrator;
use turbojev_core::{OrderRobustness, TurboJevAsyncEngine, TurboJevEngine};
use turbojev_schema::{Answer, EvaluationRequest, Question};

#[derive(Default)]
struct SetBackend {
    seen: RefCell<Vec<DecisionBatch>>,
}
fn logits(task: &DecisionTask) -> Vec<f32> {
    if task.id == "severity" {
        return vec![0.0, 2.0, -1.0];
    }
    if task.id == "urgent" {
        return vec![0.0, 1.0];
    }
    task.candidates
        .iter()
        .enumerate()
        .map(|(position, candidate)| {
            // Depend on the set and on neighbours, not just additive position bias.
            let set_term = if candidate.key == "b" && task.candidates.iter().any(|c| c.key == "d") {
                1.5
            } else {
                0.0
            };
            let neighbour = if position > 0 && task.candidates[position - 1].key == "a" {
                2.0
            } else {
                0.0
            };
            set_term + neighbour + [3.0, -2.0, 0.5, -0.5][position]
        })
        .collect()
}
impl ScoringBackend for SetBackend {
    fn name(&self) -> &'static str {
        "set-interaction-fixture"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
    fn score(
        &self,
        batch: &DecisionBatch,
        _: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        self.seen.borrow_mut().push(batch.clone());
        Ok(BackendOutput {
            model: None,
            scores: batch
                .tasks
                .iter()
                .map(|task| TaskScores {
                    id: task.id.clone(),
                    logits: logits(task),
                })
                .collect(),
            metrics: turbojev_backend::BackendMetrics {
                input_tokens: Some(10),
                generated_tokens: Some(2),
                ..Default::default()
            },
        })
    }
}
impl AsyncScoringBackend for SetBackend {
    fn name(&self) -> &'static str {
        "set-interaction-fixture"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>> {
        Box::pin(std::future::ready(ScoringBackend::score(self, batch, plan)))
    }
}
fn request(keys: &[&str]) -> EvaluationRequest {
    EvaluationRequest {
        model: None,
        state: json!("Select among the complete set."),
        questions: IndexMap::from([(
            "route".into(),
            Question::choice(
                "Which is best?",
                keys.iter()
                    .map(|&k| (k.into(), format!("meaning of {k}")))
                    .collect(),
            ),
        )]),
    }
}
fn distribution(response: &turbojev_schema::EvaluationResponse) -> (&str, &IndexMap<String, f64>) {
    let Answer::Choice {
        choice,
        probabilities,
        ..
    } = &response.answers["route"]
    else {
        panic!("expected choice")
    };
    (choice, probabilities)
}
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct Wake;
    impl std::task::Wake for Wake {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Wake));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

#[test]
fn balanced_is_keyed_identically_for_all_24_input_orders_and_async() {
    let engine =
        TurboJevEngine::new(SetBackend::default()).with_order_robustness(OrderRobustness::Balanced);
    let reference = engine.evaluate(request(&["a", "b", "c", "d"])).unwrap();
    for a in ["a", "b", "c", "d"] {
        for b in ["a", "b", "c", "d"] {
            for c in ["a", "b", "c", "d"] {
                for d in ["a", "b", "c", "d"] {
                    if [a, b, c, d]
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != 4
                    {
                        continue;
                    }
                    let response = engine.evaluate(request(&[a, b, c, d])).unwrap();
                    assert_eq!(distribution(&reference).0, distribution(&response).0);
                    for (key, probability) in distribution(&reference).1 {
                        assert_eq!(*probability, distribution(&response).1[key]);
                    }
                    let asynchronous = block_on(
                        TurboJevAsyncEngine::new(SetBackend::default())
                            .with_order_robustness(OrderRobustness::Balanced)
                            .evaluate(request(&[a, b, c, d])),
                    )
                    .unwrap();
                    assert_eq!(response, asynchronous);
                    assert_eq!(response.metadata.order_evaluations, 4);
                }
            }
        }
    }
}

#[test]
fn balanced_calibrates_each_layout_once_and_leaves_score_and_noul_unchanged() {
    let mut input = request(&["d", "a", "c", "b"]);
    input.questions.insert(
        "severity".into(),
        Question::Score {
            instructions: "Severity?".into(),
            criteria: vec!["Low".into(), "Medium".into(), "High".into()],
        },
    );
    input
        .questions
        .insert("urgent".into(), Question::noul("Urgent?"));
    let baseline = TurboJevEngine::with_calibrator(
        SetBackend::default(),
        TemperatureCalibrator::new(2.0).unwrap(),
    )
    .evaluate(input.clone())
    .unwrap();
    let engine = TurboJevEngine::with_calibrator(
        SetBackend::default(),
        TemperatureCalibrator::new(2.0).unwrap(),
    )
    .with_order_robustness(OrderRobustness::Balanced);
    let result = engine.evaluate(input).unwrap();
    assert_eq!(result.answers["severity"], baseline.answers["severity"]);
    assert_eq!(result.answers["urgent"], baseline.answers["urgent"]);
    assert_eq!(result.metadata.order_evaluations, 5);
    assert_eq!(result.usage.input_tokens, Some(50));
    assert_eq!(result.usage.generated_tokens, 10);
    let seen = engine.backend().seen.borrow();
    assert_eq!(
        seen.iter()
            .flat_map(|b| &b.tasks)
            .filter(|t| t.id == "severity")
            .count(),
        1
    );
    let mut expected = IndexMap::<String, f64>::new();
    for task in seen
        .iter()
        .flat_map(|batch| &batch.tasks)
        .filter(|t| t.id == "route")
    {
        let exponentials: Vec<f64> = logits(task)
            .iter()
            .map(|&l| (f64::from(l) / 2.0).exp())
            .collect();
        let denominator: f64 = exponentials.iter().sum();
        for (candidate, weight) in task.candidates.iter().zip(exponentials) {
            *expected.entry(candidate.key.clone()).or_default() += weight / denominator / 4.0;
        }
    }
    for (key, probability) in distribution(&result).1 {
        assert!((probability - expected[key]).abs() < 1e-12);
    }
}

#[test]
fn odd_choice_balancing_uses_two_cycles_but_none_stays_single_pass() {
    let input = request(&["c", "a", "b"]);
    let engine = TurboJevEngine::new(SetBackend::default());
    assert_eq!(
        engine
            .evaluate(input.clone())
            .unwrap()
            .metadata
            .order_evaluations,
        0
    );
    assert_eq!(engine.backend().seen.borrow().len(), 1);
    let result = engine
        .with_order_robustness(OrderRobustness::Balanced)
        .evaluate(input)
        .unwrap();
    assert_eq!(result.metadata.order_evaluations, 6);
}

#[test]
fn exact_choice_ties_use_stable_keys_and_near_ties_keep_the_higher_probability() {
    for policy in [
        OrderRobustness::None,
        OrderRobustness::Cyclic,
        OrderRobustness::Balanced,
    ] {
        for keys in [["z", "a"], ["a", "z"]] {
            let backend = turbojev_backend_mock::FixtureBackend::builder()
                .with_logits("route", vec![0.0, 0.0])
                .build();
            let response = TurboJevEngine::new(backend)
                .with_order_robustness(policy)
                .evaluate(request(&keys))
                .unwrap();
            assert_eq!(distribution(&response).0, "a");
        }
    }
    let backend = turbojev_backend_mock::FixtureBackend::builder()
        .with_logits("route", vec![1e-7, 0.0])
        .build();
    assert_eq!(
        distribution(
            &TurboJevEngine::new(backend)
                .evaluate(request(&["z", "a"]))
                .unwrap()
        )
        .0,
        "z"
    );
}
