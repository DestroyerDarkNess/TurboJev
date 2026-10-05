//! Independent fixtures exercise core semantics; these are not model benchmarks.
use serde_json::{json, Value};
use std::cell::RefCell;
use turbojev_backend::{
    BackendCapabilities, BackendError, BackendOutput, DecisionBatch, ExecutionPlan, FieldOutput,
    FieldTask, ScoringBackend, TaskScores,
};
use turbojev_core::{fields::orderings, TurboJevEngine};
use turbojev_schema::fields::GenerateRequest;

#[derive(Default)]
struct Model {
    seen: RefCell<Vec<FieldTask>>,
}
impl ScoringBackend for Model {
    fn name(&self) -> &'static str {
        "field-test"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
    fn score(
        &self,
        batch: &DecisionBatch,
        _: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        Ok(BackendOutput {
            model: None,
            metrics: Default::default(),
            scores: batch
                .tasks
                .iter()
                .map(|t| TaskScores {
                    id: t.id.clone(),
                    logits: t
                        .candidates
                        .iter()
                        .enumerate()
                        .map(|(i, _)| if i == 0 { 4.0 } else { 0.0 })
                        .collect(),
                })
                .collect(),
        })
    }
    fn infer_field(&self, task: &FieldTask) -> Result<FieldOutput, BackendError> {
        self.seen.borrow_mut().push(task.clone());
        Ok(FieldOutput {
            value: if task.values.is_none() {
                Some(if task.name == "text" {
                    json!("actual model text")
                } else {
                    json!(7)
                })
            } else {
                None
            },
            logits: task.values.as_ref().map_or(Vec::new(), |values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, _)| if i == 0 { 4.0 } else { 0.0 })
                    .collect()
            }),
            thinking: String::new(),
            input_tokens: Some(3),
            generated_tokens: usize::from(task.values.is_none()),
        })
    }
}
fn request(questions: Value) -> GenerateRequest {
    serde_json::from_value(json!({"context":"x", "questions":questions})).unwrap()
}

#[test]
fn skipped_dependencies_propagate_without_inference_or_fake_null() {
    let engine = TurboJevEngine::new(Model::default());
    let result = engine
        .generate(request(json!({
            "root":{"type":"string", "enum":["feature","bug"]},
            "skip":{"type":"boolean", "thinking":"auto", "when":{"root":"bug"}},
            "descendant":{"type":"string", "depends_on":["skip"]}
        })))
        .unwrap();
    assert_eq!(result.result.len(), 1);
    assert_eq!(result.skipped, vec!["descendant", "skip"]); // stable source-map order
    assert_eq!(engine.backend().seen.borrow().len(), 1);
    assert!(result.thinking_effort.is_empty());
}

#[test]
fn dependent_prompts_receive_plain_values_not_probability_wrappers() {
    let engine = TurboJevEngine::new(Model::default());
    let result = engine
        .generate(request(json!({
            "roll":{"type":"integer", "enum":[2,5], "return_probabilities":true},
            "text":{"type":"string", "depends_on":["roll"]}
        })))
        .unwrap();
    assert_eq!(result.result["roll"]["value"], 2);
    assert_eq!(result.result["text"], "actual model text");
    assert_eq!(
        engine.backend().seen.borrow()[1].state["previous_answers"]["roll"],
        2
    );
    assert_eq!(result.usage.generated_tokens, 1);
}

#[test]
fn exhaustive_and_auto_cancel_pure_position_bias() {
    for budget in [json!("all"), json!("auto")] {
        let result=TurboJevEngine::new(Model::default()).generate(request(json!({"x":{"type":"string", "enum":["b","a","c"], "permutations":budget, "return_probabilities":true}}))).unwrap();
        for p in result.result["x"]["probabilities"]
            .as_object()
            .unwrap()
            .values()
        {
            assert!((p.as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-10);
        }
    }
}

#[test]
fn balanced_design_has_equal_positions_and_neighbours() {
    for n in 2..10 {
        let values: Vec<_> = (0..n).map(|i| json!(i)).collect();
        let rows = orderings(&values, &json!("auto"), 42).unwrap();
        let mut positions = vec![vec![0; n]; n];
        let mut pairs = vec![vec![0; n]; n];
        for row in &rows {
            for (pos, &value) in row.iter().enumerate() {
                positions[pos][value] += 1;
            }
            for pair in row.windows(2) {
                pairs[pair[0]][pair[1]] += 1;
            }
        }
        assert!(positions
            .iter()
            .flatten()
            .all(|&count| count == rows.len() / n));
        let expected = if n % 2 == 0 { 1 } else { 2 };
        for (a, row) in pairs.iter().enumerate() {
            for (b, &count) in row.iter().enumerate() {
                assert_eq!(count, if a == b { 0 } else { expected });
            }
        }
    }
}

#[test]
fn sampling_is_unique_reproducible_and_not_factorial_allocation() {
    let values: Vec<_> = (0..24).map(|i| json!(i)).collect();
    let a = orderings(&values, &json!(8), 42).unwrap();
    assert_eq!(a, orderings(&values, &json!(8), 42).unwrap());
    assert_ne!(a, orderings(&values, &json!(8), 43).unwrap());
    assert_eq!(a.len(), 8);
    for (i, row) in a.iter().enumerate() {
        assert!(!a[..i].contains(row));
    }
}

#[test]
fn invalid_complete_schema_performs_zero_model_calls() {
    let engine = TurboJevEngine::new(Model::default());
    assert!(engine
        .generate(request(
            json!({"good":{"type":"boolean"}, "bad":{"type":"integer", "enum":[true]}})
        ))
        .is_err());
    assert!(engine.backend().seen.borrow().is_empty());
}

#[test]
fn generated_type_violations_surface_as_errors() {
    let engine = TurboJevEngine::new(Model::default());
    assert!(engine
        .generate(request(json!({"oops":{"type":"string"}})))
        .is_err());
}

#[test]
fn auto_effort_is_an_actual_extra_decision() {
    let engine = TurboJevEngine::new(Model::default());
    let result = engine
        .generate(request(json!({"x":{"type":"boolean", "thinking":"auto"}})))
        .unwrap();
    assert_eq!(engine.backend().seen.borrow().len(), 2);
    assert_eq!(result.thinking_effort["x"], "none");
    assert_eq!(result.usage.decisions, 1);
    assert_eq!(result.usage.input_tokens, Some(6));
}

#[test]
fn null_is_present_on_the_wire_but_missing_value_is_not_null() {
    let null: FieldOutput = serde_json::from_value(
        json!({"value":null,"logits":[],"thinking":"","generated_tokens":0}),
    )
    .unwrap();
    assert_eq!(null.value, Some(Value::Null));
    let missing: FieldOutput =
        serde_json::from_value(json!({"logits":[1,0],"thinking":"","generated_tokens":0})).unwrap();
    assert_eq!(missing.value, None);
    assert!(!serde_json::to_value(missing)
        .unwrap()
        .as_object()
        .unwrap()
        .contains_key("value"));
}

#[test]
fn nullable_string_null_and_actual_null_have_distinct_probabilities() {
    let result = TurboJevEngine::new(Model::default())
        .generate(request(json!({
            "x":{"type":["string","null"],"enum":["null",null],"return_probabilities":true}
        })))
        .unwrap();
    let value = &result.result["x"];
    assert_eq!(value["probabilities"].as_object().unwrap().len(), 2);
    assert_eq!(value["distribution"][0]["value"], "null");
    assert!(value["distribution"][1]["value"].is_null());
}

#[test]
fn rotations_average_probabilities_rather_than_logits() {
    let mut workflow = turbojev_core::fields::Generation::new(request(json!({
        "x":{"type":"string","enum":["a","b"],"permutations":"all","return_probabilities":true}
    })))
    .unwrap();
    for logits in [vec![10., 0.], vec![1., 0.]] {
        assert!(workflow.next_task().unwrap().is_some());
        workflow
            .accept(
                FieldOutput {
                    value: None,
                    logits,
                    thinking: String::new(),
                    input_tokens: None,
                    generated_tokens: 0,
                },
                &turbojev_calibration::SoftmaxCalibrator::default(),
            )
            .unwrap();
    }
    let result = workflow.finish();
    let expected = ((1.0 / (1.0 + (-10_f64).exp())) + (1.0 / (1.0 + 1_f64.exp()))) / 2.0;
    assert!((result.result["x"]["probabilities"]["a"].as_f64().unwrap() - expected).abs() < 1e-10);
}

#[test]
fn dependencies_preserve_the_media_envelope() {
    let mut req = request(
        json!({"parent":{"type":"boolean"},"child":{"type":"boolean","depends_on":["parent"]}}),
    );
    req.context = json!({"contract":"turbojev.media.v1","text":"Look at the image.","modality":"image","base64":"aW1hZ2U="});
    let mut generation = turbojev_core::fields::Generation::new(req).unwrap();
    generation.next_task().unwrap();
    generation
        .accept(
            FieldOutput {
                value: None,
                logits: vec![2., 0.],
                thinking: String::new(),
                input_tokens: None,
                generated_tokens: 0,
            },
            &turbojev_calibration::SoftmaxCalibrator::default(),
        )
        .unwrap();
    let task = generation.next_task().unwrap().unwrap();
    assert_eq!(task.state["contract"], "turbojev.media.v1");
    assert!(task.state["text"]
        .as_str()
        .unwrap()
        .contains("Previous answers:"));
    assert_eq!(task.state["base64"], "aW1hZ2U=");
}

#[test]
fn constant_fields_do_not_invoke_even_custom_scorers() {
    let engine = TurboJevEngine::new(Model::default());
    let result = engine
        .generate(request(
            json!({"x":{"type":"integer","enum":[7],"return_probabilities":true}}),
        ))
        .unwrap();
    assert!(engine.backend().seen.borrow().is_empty());
    assert_eq!(result.result["x"]["value"], 7);
    assert_eq!(result.result["x"]["confidence"], 1.0);
    assert_eq!(result.usage.generated_tokens, 0);
}

impl turbojev_backend::AsyncScoringBackend for Model {
    fn name(&self) -> &'static str {
        "field-test"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> turbojev_backend::BackendFuture<'a, Result<BackendOutput, BackendError>> {
        Box::pin(async move { ScoringBackend::score(self, batch, plan) })
    }
    fn infer_field_async<'a>(
        &'a self,
        task: &'a FieldTask,
    ) -> turbojev_backend::BackendFuture<'a, Result<FieldOutput, BackendError>> {
        Box::pin(async move { ScoringBackend::infer_field(self, task) })
    }
}

#[test]
fn non_send_async_host_has_identical_workflow_semantics() {
    struct Noop;
    impl std::task::Wake for Noop {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let req = request(json!({
        "roll":{"type":"integer","enum":[2,5],"permutations":"auto","return_probabilities":true},
        "text":{"type":"string","depends_on":["roll"]},
        "constant":{"type":"boolean","enum":[true]},
        "skipped":{"type":"boolean","when":{"roll":5},"thinking":"auto"}
    }));
    let sync = TurboJevEngine::new(Model::default())
        .generate(req.clone())
        .unwrap();
    let engine = turbojev_core::TurboJevAsyncEngine::new(Model::default());
    let mut future = Box::pin(engine.generate(req));
    let waker = std::task::Waker::from(std::sync::Arc::new(Noop));
    let mut context = std::task::Context::from_waker(&waker);
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => assert_eq!(result.unwrap(), sync),
        std::task::Poll::Pending => panic!("Fixture futures should complete immediately"),
    }
}
