//! Independent contract cases covering TypeLLM 0.6.9's structured features.
use serde_json::{json, Value};
use turbojev_schema::fields::{GenerateRequest, MAX_ENUM_VALUES};

fn compile(questions: Value) -> Result<Vec<turbojev_schema::fields::Field>, String> {
    let request: GenerateRequest =
        serde_json::from_value(json!({"context": "schema test", "questions": questions})).unwrap();
    request.compile().map_err(|error| error.to_string())
}

#[test]
fn nested_objects_link_local_dependencies_and_object_groups() {
    let fields = compile(json!({
        "enabled": {"type": "boolean"},
        "person": {"type": "object", "instructions": "Extract the person.", "when": {"enabled": true}, "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "depends_on": ["name"]},
            "address": {"type": "object", "properties": {"city": {"type": "string"}}}
        }},
        "summary": {"type": "string", "depends_on": ["person"]}
    })).unwrap();
    let age = fields
        .iter()
        .find(|field| field.name == "person.age")
        .unwrap();
    assert_eq!(age.path, ["person", "age"]);
    assert_eq!(age.depends_on, ["enabled", "person.name"]);
    assert!(age.instructions.contains("Extract the person."));
    let summary = fields.last().unwrap();
    assert_eq!(
        summary.dependency_groups,
        [vec!["person.name", "person.age", "person.address.city"]]
    );
}

#[test]
fn invalid_object_scopes_paths_cycles_and_container_keywords_fail() {
    for questions in [
        json!({"a": {"type": "object", "properties": {}}}),
        json!({"a": {"type": "object", "enum": [1], "properties": {"b": {"type": "integer"}}}}),
        json!({"a": {"type": "object", "properties": {"b": {"type": "integer"}}}, "a.b": {"type": "integer"}}),
        json!({"a": {"type": "object", "properties": {"b": {"type": "integer", "depends_on": ["missing"]}}}}),
        json!({"a": {"type": "object", "depends_on": ["a"], "properties": {"b": {"type": "integer"}}}}),
        json!({"a": {"type": "object", "properties": {"b": {"type": "integer"}}}, "c": {"type": "boolean", "when": {"a": 1}}}),
    ] {
        assert!(compile(questions.clone()).is_err(), "accepted {questions}");
    }
}

#[test]
fn described_choices_and_ordered_levels_keep_their_meanings() {
    let fields = compile(json!({
        "team": {"type": "string", "choices": [{"value": "billing", "description": "Payments and refunds"}, {"value": "outage"}]},
        "severity": {"type": "number", "levels": [{"label": "calm"}, {"label": "annoyed", "description": "Some frustration"}, {"label": "furious"}], "return_probabilities": true},
        "check": {"type": "boolean", "when": {"severity": {"gte": 1.5}}}
    })).unwrap();
    assert_eq!(
        fields[0].descriptions[0],
        (json!("billing"), "Payments and refunds".into())
    );
    assert_eq!(fields[1].levels, ["calm", "annoyed", "furious"]);
    assert_eq!(fields[1].permutations, 1);
    assert_eq!(fields[1].output_schema()["type"], "string");
    for field in [
        json!({"type": "number", "levels": [{"label": "a"}]}),
        json!({"type": "number", "levels": [{"label": "a"}, {"label": "a"}]}),
        json!({"type": "number", "levels": [{"label": "a"}, {"label": "b"}], "permutations": 1}),
        json!({"type": "string", "choices": [{"value": "a", "description": 3}]}),
        json!({"type": "string", "choices": [{"value": "a"}], "enum": ["b"]}),
    ] {
        assert!(compile(json!({"x": field})).is_err());
    }
}

#[test]
fn confidence_conditions_are_checked_before_inference() {
    let fields = compile(json!({
        "team": {"type": "string", "enum": ["a", "b"], "return_probabilities": true},
        "check": {"type": "boolean", "when": {"team": {"confidence": {"gte": 0.8}, "ne": "b"}}}
    }))
    .unwrap();
    let values = indexmap::IndexMap::from([("team".into(), json!("a"))]);
    assert!(fields[1]
        .conditions_with_confidence(&values, &indexmap::IndexMap::from([("team".into(), 0.9)])));
    assert!(!fields[1]
        .conditions_with_confidence(&values, &indexmap::IndexMap::from([("team".into(), 0.5)])));
    assert!(!fields[1].conditions_met(&values));
    for bounds in [
        json!({}),
        json!({"eq": 0.9}),
        json!({"gte": 1.1}),
        json!({"lt": "0.5"}),
    ] {
        assert!(compile(json!({"x": {"type": "boolean", "return_probabilities": true}, "y": {"type": "boolean", "when": {"x": {"confidence": bounds}}}})).is_err());
    }
}

#[test]
fn large_enums_use_bounded_value_scoring_without_factorial_work() {
    let values: Vec<_> = (0..512).map(|index| format!("zone_{index}")).collect();
    let fields =
        compile(json!({"zone": {"type": "string", "enum": values, "return_probabilities": true}}))
            .unwrap();
    assert_eq!(fields[0].values.as_ref().unwrap().len(), MAX_ENUM_VALUES);
    assert_eq!(fields[0].permutations, 1);
    assert!(
        compile(json!({"zone": {"type": "integer", "enum": (0..513).collect::<Vec<_>>()}}))
            .is_err()
    );
}

#[test]
fn arrays_validate_items_bounds_and_continuation() {
    let fields = compile(json!({"rows": {"type": "array", "items": {"type": "object", "properties": {
        "sku": {"type": "string"}, "count": {"type": "integer"},
        "location": {"type": "object", "properties": {"shelf": {"type": "string"}}}
    }}, "continue_from": [{"sku": "a", "count": 3, "location": {"shelf": "A1"}}], "minItems": 1, "maxItems": 10}})).unwrap();
    let array = fields[0].array.as_ref().unwrap();
    assert_eq!(array.continue_from.len(), 1);
    assert_eq!(
        array.item_schema()["properties"]["location"]["required"],
        json!(["shelf"])
    );
    assert!(!array.accepts_item(&json!({"sku": "a", "count": true})));
    for field in [
        json!({"type": "array"}),
        json!({"type": "array", "items": {"type": "array", "items": {"type": "string"}}}),
        json!({"type": "array", "items": {"type": "string"}, "minItems": -1}),
        json!({"type": "array", "items": {"type": "string"}, "maxItems": 51}),
        json!({"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 2}),
        json!({"type": "array", "items": {"type": "integer"}, "continue_from": [true]}),
        json!({"type": "array", "items": {"type": "string", "return_probabilities": true}}),
        json!({"type": "array", "items": {"type": "object", "properties": {"a": {"type": "string", "when": {"b": true}}}}}),
    ] {
        assert!(compile(json!({"x": field})).is_err());
    }
}
#[test]
fn multiple_images_normalize_before_validation_and_reject_mixed_sources() {
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "images": ["data:image/png;base64,YQ==", "Yg=="],
        "questions": {"scene": {"type":"boolean"}}
    }))
    .unwrap();
    let context = request.effective_context().unwrap();
    assert_eq!(context["images"], serde_json::json!(["YQ==", "Yg=="]));
    assert!(request.compile().is_ok());
    assert!(
        turbojev_schema::media::MediaEvidence::from_state(&serde_json::json!({
            "contract":"turbojev.media.v1","modality":"image","images":["YQ=="],"base64":"Yg=="
        }))
        .is_err()
    );
}
