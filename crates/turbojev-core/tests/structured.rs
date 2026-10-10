//! Workflow fixtures test contracts and scheduling, not model intelligence.
use serde_json::{json, Value};
use turbojev_backend::{FieldOutput, FieldTask};
use turbojev_calibration::SoftmaxCalibrator;
use turbojev_core::fields::Generation;
use turbojev_schema::fields::{GenerateRequest, GenerateResponse};

fn request(questions: Value) -> GenerateRequest {
    serde_json::from_value(json!({"context": "workflow evidence", "questions": questions})).unwrap()
}

fn answer(value: Value) -> FieldOutput {
    FieldOutput {
        prompt: None,
        value: Some(value),
        logits: vec![],
        thinking: String::new(),
        input_tokens: Some(4),
        generated_tokens: 2,
    }
}

fn choose(task: &FieldTask, value: Value) -> FieldOutput {
    let values = task.values.as_ref().unwrap();
    assert!(values.contains(&value), "missing {value} in {values:?}");
    FieldOutput {
        prompt: None,
        value: None,
        logits: values
            .iter()
            .map(|option| if *option == value { 5.0 } else { -5.0 })
            .collect(),
        thinking: String::new(),
        input_tokens: Some(4),
        generated_tokens: 0,
    }
}

fn run(
    mut generation: Generation,
    mut infer: impl FnMut(&FieldTask) -> FieldOutput,
) -> GenerateResponse {
    for _ in 0..1000 {
        let Some(task) = generation.next_task().unwrap() else {
            return generation.finish();
        };
        generation.accept(infer(&task), &SoftmaxCalibrator).unwrap();
    }
    panic!("workflow did not terminate");
}

#[test]
fn extreme_seed_does_not_stall_distinct_orderings() {
    let seed = 0u64.wrapping_sub(0x9e3779b97f4a7c15);
    let orders =
        turbojev_core::fields::orderings(&[json!("a"), json!("b"), json!("c")], &json!(3), seed)
            .unwrap();
    assert_eq!(orders.len(), 3);
    assert_ne!(orders[1], orders[2]);
}

#[test]
fn sampled_duplicate_retries_are_bounded_and_do_not_fabricate_items() {
    let mut request = request(
        json!({"rows": {"type": "array", "items": {"type": "string"}, "continue_from": ["old"]}}),
    );
    request.temperature = Some(0.5);
    let generation = Generation::new(request).unwrap();
    let mut calls = 0;
    let response = run(generation, |_| {
        calls += 1;
        answer(json!("old"))
    });
    assert_eq!(calls, 3);
    assert_eq!(response.result["rows"], json!(["old"]));
}

#[test]
fn nested_partial_objects_and_group_dependencies_keep_plain_answers() {
    let generation = Generation::new(request(json!({
        "person": {"type": "object", "properties": {
            "name": {"type": "string", "enum": ["Ada"]},
            "optional": {"type": "string", "when": {"name": "Ada"}, "enum": ["present"]},
            "skipped": {"type": "string", "when": {"name": {"ne": "Ada"}}}
        }},
        "summary": {"type": "string", "depends_on": ["person"]}
    })))
    .unwrap();
    let response = run(generation, |task| {
        if task.name == "summary" {
            assert_eq!(
                task.state["previous_answers"]["person"],
                json!({"name": "Ada", "optional": "present"})
            );
            answer(json!("Ada is present"))
        } else {
            choose(task, task.values.as_ref().unwrap()[0].clone())
        }
    });
    assert_eq!(
        response.result["person"],
        json!({"name": "Ada", "optional": "present"})
    );
    assert_eq!(response.skipped, ["person.skipped"]);
    assert_eq!(response.result["summary"], "Ada is present");
}

#[test]
fn fully_skipped_object_is_reported_once_and_skips_dependents() {
    let generation = Generation::new(request(json!({
        "flag": {"type": "boolean", "enum": [false]},
        "person": {"type": "object", "when": {"flag": true}, "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        "summary": {"type": "string", "depends_on": ["person"]}
    }))).err().unwrap();
    // Impossible equality is rejected rather than silently accepting a dead branch.
    assert!(generation.to_string().contains("impossible"));
    let generation = Generation::new(request(json!({
        "flag": {"type": "boolean"},
        "person": {"type": "object", "when": {"flag": true}, "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}},
        "summary": {"type": "string", "depends_on": ["person"]}
    }))).unwrap();
    let response = run(generation, |task| {
        assert_eq!(task.name, "flag");
        choose(task, json!(false))
    });
    assert_eq!(response.skipped, ["person", "summary"]);
}

#[test]
fn scalar_arrays_use_null_as_end_without_adding_a_null_item() {
    let generation = Generation::new(request(
        json!({"skills": {"type": "array", "items": {"type": "string"}, "minItems": 1}}),
    ))
    .unwrap();
    let mut index = 0;
    let response = run(generation, |task| {
        assert_eq!(task.name, "item");
        let result = [json!("Rust"), json!("Python"), Value::Null][index].clone();
        if index == 0 {
            assert_eq!(task.output_schema["type"], "string");
        }
        if index == 1 {
            assert!(task.state.as_str().unwrap().contains("Rust"));
        }
        index += 1;
        answer(result)
    });
    assert_eq!(response.result["skills"], json!(["Rust", "Python"]));
    assert_eq!(response.usage.generated_tokens, 6);
    assert_eq!(response.usage.decisions, 1);
}

#[test]
fn nullable_array_items_use_a_separate_continue_decision() {
    let generation = Generation::new(request(
        json!({"values": {"type": "array", "items": {"type": ["integer", "null"]}}}),
    ))
    .unwrap();
    let mut continuation = 0;
    let response = run(generation, |task| {
        if task.name == "values.append_item" {
            continuation += 1;
            choose(task, json!(continuation == 1))
        } else {
            answer(Value::Null)
        }
    });
    assert_eq!(response.result["values"], json!([null]));
}

#[test]
fn object_array_items_are_generated_whole_and_media_stays_at_the_root() {
    let mut req = request(
        json!({"rows": {"type": "array", "items": {"type": "object", "properties": {
        "sku": {"type": "string"}, "count": {"type": "integer"}
    }}, "continue_from": [{"sku": "A", "count": 2}], "maxItems": 3}}),
    );
    req.context = json!({"contract": "turbojev.media.v1", "text": "Read the rows.", "modality": "image", "base64": "aW1hZ2U="});
    let generation = Generation::new(req).unwrap();
    let mut items = 0;
    let response = run(generation, |task| {
        assert_eq!(task.state["base64"], "aW1hZ2U=");
        assert!(task.state["text"]
            .as_str()
            .unwrap()
            .contains("\"sku\":\"A\""));
        if task.name == "rows.append_item" {
            choose(task, json!(items == 0))
        } else {
            assert_eq!(task.name, "rows[1]");
            assert_eq!(task.output_schema["required"], json!(["sku", "count"]));
            items += 1;
            answer(json!({"sku": "B", "count": 3}))
        }
    });
    assert_eq!(items, 1);
    assert_eq!(
        response.result["rows"],
        json!([{"sku": "A", "count": 2}, {"sku": "B", "count": 3}])
    );
}

#[test]
fn continuation_limit_counts_existing_items_and_duplicate_stops_generation() {
    let generation = Generation::new(request(json!({"tags": {"type": "array", "items": {"type": "string"}, "continue_from": ["a"], "maxItems": 1}}))).unwrap();
    let response = run(generation, |_| panic!("maxItems was already reached"));
    assert_eq!(response.result["tags"], json!(["a"]));
    let generation = Generation::new(request(
        json!({"tags": {"type": "array", "items": {"type": "string"}, "continue_from": ["a"]}}),
    ))
    .unwrap();
    let response = run(generation, |_| answer(json!("a")));
    assert_eq!(response.result["tags"], json!(["a"]));
}

#[test]
fn ordered_score_uses_probability_weighted_indices_without_permutations() {
    let generation = Generation::new(request(json!({
        "tone": {"type": "number", "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}], "return_probabilities": true},
        "urgent": {"type": "boolean", "when": {"tone": {"gte": 1.5}}}
    }))).unwrap();
    let mut tone_calls = 0;
    let response = run(generation, |task| {
        if task.name == "tone" {
            tone_calls += 1;
            choose(task, json!("furious"))
        } else {
            choose(task, json!(true))
        }
    });
    assert_eq!(tone_calls, 1);
    assert!(response.result["tone"]["value"].as_f64().unwrap() > 1.99);
    assert!(
        response.result["tone"]["probabilities"]["furious"]
            .as_f64()
            .unwrap()
            > 0.99
    );
    assert_eq!(response.result["urgent"], true);
}

#[test]
fn ambiguous_choice_has_zero_confidence_and_skips_confidence_gate() {
    let generation = Generation::new(request(json!({
        "coin": {"type": "string", "enum": ["heads", "tails"], "return_probabilities": true, "permutations":1},
        "certain": {"type": "boolean", "when": {"coin": {"confidence": {"gte": 0.8}}}}
    })))
    .unwrap();
    let response = run(generation, |task| {
        assert_eq!(task.name, "coin");
        FieldOutput {
            prompt: None,
            value: None,
            logits: vec![0.0, 0.0],
            thinking: String::new(),
            input_tokens: Some(3),
            generated_tokens: 0,
        }
    });
    assert_eq!(response.result["coin"]["confidence"], 0.0);
    assert_eq!(response.result["coin"]["probability"], 0.5);
    assert_eq!(response.skipped, ["certain"]);
}

#[test]
fn finite_sampling_reports_the_tempered_distribution_and_repeats_its_seed() {
    let mut request = request(json!({"coin": {
        "type": "string", "enum": ["heads", "tails"],
        "return_probabilities": true, "permutations": 1
    }}));
    request.temperature = Some(2.0);
    request.seed = 17;
    let infer = |_: &FieldTask| FieldOutput {
        prompt: None,
        value: None,
        logits: vec![4.0, 0.0],
        thinking: String::new(),
        input_tokens: Some(3),
        generated_tokens: 0,
    };
    let first = run(Generation::new(request.clone()).unwrap(), infer);
    let second = run(Generation::new(request).unwrap(), infer);
    assert_eq!(first.result, second.result);
    let expected = 1.0 / (1.0 + (-2.0_f64).exp());
    let actual = first.result["coin"]["probabilities"]["heads"]
        .as_f64()
        .unwrap();
    assert!((actual - expected).abs() < 1e-12);
    let winner = first.result["coin"]["value"].as_str().unwrap();
    assert_eq!(
        first.result["coin"]["probability"],
        first.result["coin"]["probabilities"][winner]
    );
}

#[test]
fn omitted_maximum_limits_new_items_without_truncating_a_large_prefix() {
    let prefix: Vec<_> = (0..55).map(|number| json!(number)).collect();
    let generation = Generation::new(request(json!({"numbers": {
        "type": "array", "items": {"type": "integer"}, "continue_from": prefix
    }})))
    .unwrap();
    let mut generated = 0;
    let response = run(generation, |_| {
        let value = 55 + generated;
        generated += 1;
        answer(json!(value))
    });
    assert_eq!(generated, 50);
    let numbers = response.result["numbers"].as_array().unwrap();
    assert_eq!(numbers.len(), 105);
    assert_eq!(numbers[0], 0);
    assert_eq!(numbers[104], 104);
    assert_eq!(response.usage.decisions, 1);
    assert_eq!(response.usage.input_tokens, Some(200));
}

#[test]
fn generated_object_items_must_include_every_declared_property() {
    let mut generation = Generation::new(request(json!({"rows": {
        "type": "array", "minItems": 1,
        "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "count": {"type": "integer"}
        }}
    }})))
    .unwrap();
    let task = generation.next_task().unwrap().unwrap();
    assert_eq!(task.name, "rows[0]");
    assert!(generation
        .accept(answer(json!({"sku": "A"})), &SoftmaxCalibrator)
        .is_err());
}

#[test]
fn object_sampling_retries_advance_the_seed_instead_of_replaying_a_duplicate() {
    let mut request = request(json!({"rows": {
        "type": "array", "minItems": 2, "maxItems": 2,
        "continue_from": [{"sku": "A"}],
        "items": {"type": "object", "properties": {"sku": {"type": "string"}}}
    }}));
    request.temperature = Some(0.5);
    request.seed = 7;
    let mut seeds = Vec::new();
    let response = run(Generation::new(request).unwrap(), |task| {
        seeds.push(task.seed);
        answer(json!({"sku": if task.seed == 8 {"A"} else {"B"}}))
    });
    assert_eq!(seeds, [8, 9]);
    assert_eq!(response.result["rows"], json!([{"sku": "A"}, {"sku": "B"}]));
}

#[test]
fn sampling_has_two_duplicate_retries_for_the_whole_array() {
    let mut request = request(json!({"tags": {
        "type": "array", "items": {"type": "string"}, "continue_from": ["A"]
    }}));
    request.temperature = Some(0.5);
    let mut calls = 0;
    let response = run(Generation::new(request).unwrap(), |_| {
        let value = ["A", "B", "B", "C", "C"][calls];
        calls += 1;
        answer(json!(value))
    });
    assert_eq!(calls, 5);
    assert_eq!(response.result["tags"], json!(["A", "B", "C"]));
}

#[test]
fn dependency_ancestors_are_visible_but_siblings_are_not() {
    let generation = Generation::new(request(json!({
        "a": {"type":"integer"}, "unrelated": {"type":"integer"},
        "b": {"type":"integer", "depends_on":["a"]},
        "c": {"type":"integer", "depends_on":["b"]}
    })))
    .unwrap();
    run(generation, |task| {
        if task.name == "c" {
            assert_eq!(task.state["previous_answers"], json!({"a":7,"b":7}));
        }
        answer(json!(7))
    });
}

#[test]
fn integer_conditions_accept_equivalent_float_operands() {
    let generation = Generation::new(request(json!({
        "a": {"type":"integer", "enum":[1,2]},
        "b": {"type":"boolean", "when":{"a":1.0}}
    })))
    .unwrap();
    let result = run(generation, |task| {
        choose(
            task,
            if task.name == "a" {
                json!(1)
            } else {
                json!(true)
            },
        )
    });
    assert_eq!(result.result["b"], true);
    assert!(Generation::new(request(
        json!({"a":{"type":"integer"},"b":{"type":"boolean","when":{"a":1.5}}})
    ))
    .is_err());
}

#[test]
fn explicit_argmax_with_positive_temperature_never_samples() {
    for seed in 0..20 {
        let mut request = request(
            json!({"coin":{"type":"string","enum":["heads","tails"],"return_probabilities":true,"permutations":1}}),
        );
        request.mode = Some(turbojev_schema::fields::DecodingMode::Argmax);
        request.temperature = Some(2.0);
        request.seed = seed;
        let result = run(Generation::new(request).unwrap(), |task| {
            assert_eq!(task.temperature, 0.0); // Open generation uses greedy decoding.
            let mut output = choose(task, json!("tails"));
            output.logits = vec![0.0, 0.1];
            output
        });
        assert_eq!(result.result["coin"]["value"], "tails");
        assert!(
            result.result["coin"]["probabilities"]["tails"]
                .as_f64()
                .unwrap()
                < 0.52
        );
    }
}

#[test]
fn dice_layout_averaging_remaps_scores_to_faces_before_sampling() {
    let labels = ["one", "two", "three", "four", "five", "six"];
    let mut selections = std::collections::HashSet::new();
    for budget in [json!(1), json!("auto"), json!(8), json!("all")] {
        for seed in 0..8 {
            let mut request = request(
                json!({"roll":{"type":"string","enum":labels,"permutations":budget,"return_probabilities":true}}),
            );
            request.mode = Some(turbojev_schema::fields::DecodingMode::Sample);
            request.seed = seed;
            let mut calls = 0;
            let response = run(Generation::new(request).unwrap(), |task| {
                calls += 1;
                assert_eq!(task.temperature, 1.0);
                let mut output = choose(task, json!("one"));
                output.logits.fill(0.0); // Known uniform fixture; not a model accuracy assertion.
                output
            });
            let roll = &response.result["roll"];
            assert!(labels.iter().any(|face| roll["value"] == *face));
            let probabilities = roll["probabilities"].as_object().unwrap();
            assert!(probabilities
                .values()
                .all(|p| (p.as_f64().unwrap() - 1.0 / 6.0).abs() < 1e-12));
            assert_eq!(
                calls,
                match budget {
                    Value::String(ref b) if b == "all" => 720,
                    Value::String(_) => 6,
                    Value::Number(ref n) => n.as_u64().unwrap() as usize,
                    _ => unreachable!(),
                }
            );
            selections.insert(roll["value"].as_str().unwrap().to_owned());
        }
    }
    assert!(selections.len() > 1);
}

#[test]
fn invalid_modes_and_digit_limits_fail_before_inference() {
    for options in [
        json!({"mode":"sample","temperature":0}),
        json!({"numeric_max_digits":0}),
        json!({"temperature":-1}),
    ] {
        let mut payload = json!({"context":"x","questions":{"n":{"type":"integer"}}});
        payload
            .as_object_mut()
            .unwrap()
            .extend(options.as_object().unwrap().clone());
        assert!(Generation::new(serde_json::from_value(payload).unwrap()).is_err());
    }
    assert!(serde_json::from_value::<GenerateRequest>(json!({"mode":"wrong"})).is_err());
}

#[test]
fn numeric_limit_rejects_invalid_values_and_preserves_usage() {
    let mut request = request(json!({"n":{"type":"number"}}));
    request.numeric_max_digits = 2;
    let mut generation = Generation::new(request).unwrap();
    let task = generation.next_task().unwrap().unwrap();
    assert!(task.grammar.is_some());
    assert!(generation
        .accept(answer(json!(123)), &SoftmaxCalibrator)
        .is_err());
    assert_eq!(generation.usage().generated_tokens, 2);
    assert_eq!(generation.usage().input_tokens, Some(4));
}
