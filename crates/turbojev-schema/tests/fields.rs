//! Regression cases inspired by TypeLLM's schema/dependency/condition tests.
use serde_json::{json, Value};
use turbojev_schema::fields::{dependency_layers, GenerateRequest, Thinking};

fn compile(fields: Value) -> Result<Vec<turbojev_schema::fields::Field>, String> {
    let request: GenerateRequest =
        serde_json::from_value(json!({"context": "test", "questions": fields})).unwrap();
    request.compile().map_err(|e| e.to_string())
}

#[test]
fn scalar_types_nullable_and_singleton_enums_preserve_values() {
    let fields = compile(json!({
        "a": {"type": "string", "enum": ["one"]},
        "b": {"type": "integer", "enum": [-1, 2]},
        "c": {"type": "number", "enum": [1.25, 2]},
        "d": {"type": ["boolean", "null"]},
        "e": {"type": ["string", "null"]},
        "f": {"type": "number"}
    }))
    .unwrap();
    assert_eq!(fields[0].values.as_ref().unwrap(), &vec![json!("one")]);
    assert!(fields[3].values.as_ref().unwrap().contains(&Value::Null));
    assert!(fields[4].accepts(&Value::Null));
    assert!(!fields[1].accepts(&json!(true)));
    assert!(!fields[1].accepts(&json!(1.25)));
}

#[test]
fn invalid_values_and_options_fail_before_inference() {
    for bad in [
        json!({"type":"integer", "enum":[true, 2]}),
        json!({"type":"number", "enum":[1, 1.0]}),
        json!({"type":"string", "enum":[null, "a"]}),
        json!({"type":["string", "integer", "null"]}),
        json!({"type":"string", "enum":[]}),
        json!({"type":"string", "enum":["a", "a"]}),
        json!({"type":"number", "minimum":0}),
        json!({"type":"string", "return_probabilities":true}),
        json!({"type":"boolean", "return_probabilities":1}),
        json!({"type":"boolean", "permutations":1}),
        json!({"type":"string", "enum":["a","b"], "permutations":0}),
        json!({"type":"string", "enum":["a","b"], "permutations":true}),
        json!({"type":"string", "enum":["a","b"], "permutations":2.0}),
        json!({"type":"string", "enum":["a","b"], "permutations":"ALL"}),
        json!({"type":"integer", "enum":[0,1,2,3,4,5,6], "permutations":"all"}),
        json!({"type":"boolean", "thinking":"yes"}),
        json!({"type":"boolean", "thinking":"auto", "thinking_budget":4}),
        json!({"type":"boolean", "thinking":false, "thinking_effort":"low"}),
        json!({"type":"boolean", "thinking_budget":0}),
        json!({"type":"boolean", "thinking_effort":"low", "thinking_budget":4}),
    ] {
        assert!(compile(json!({"x":bad})).is_err(), "accepted {bad}");
    }
}

#[test]
fn dependency_graph_accepts_forward_references_rejects_cycles() {
    let fields = compile(
        json!({"child":{"type":"boolean", "depends_on":["parent"]}, "parent":{"type":"boolean"}}),
    )
    .unwrap();
    let layers = dependency_layers(&fields).unwrap();
    assert_eq!(fields[layers[0][0]].name, "parent");
    for bad in [
        json!({"a":{"type":"boolean", "depends_on":["missing"]}}),
        json!({"a":{"type":"boolean", "depends_on":["a"]}}),
        json!({"a":{"type":"boolean", "depends_on":["b","b"]}, "b":{"type":"boolean"}}),
        json!({"a":{"type":"boolean", "depends_on":["b"]}, "b":{"type":"boolean", "depends_on":["a"]}}),
    ] {
        assert!(compile(bad).is_err());
    }
}

#[test]
fn conditions_add_dependencies_and_validate_operands() {
    let fields = compile(json!({"count":{"type":"integer"}, "check":{"type":"boolean", "when":{"count":{"gte":3,"lt":10}}}})).unwrap();
    let check = fields.iter().find(|f| f.name == "check").unwrap();
    assert_eq!(check.depends_on, ["count"]);
    for (value, expected) in [(2, false), (3, true), (9, true), (10, false)] {
        assert_eq!(
            check.conditions_met(&indexmap::IndexMap::from([("count".into(), json!(value))])),
            expected
        );
    }
    for bad in [
        json!({}),
        json!({"missing":true}),
        json!({"count":{"gte":"3"}}),
        json!({"count":{"in":[]}}),
        json!({"count":{"unknown":3}}),
    ] {
        assert!(compile(
            json!({"count":{"type":"integer"}, "check":{"type":"boolean", "when":bad}})
        )
        .is_err());
    }
    assert!(compile(
        json!({"flag":{"type":"boolean"}, "check":{"type":"boolean", "when":{"flag":1}}})
    )
    .is_err());
}

#[test]
fn json_schema_root_and_thinking_efforts() {
    let request: GenerateRequest = serde_json::from_value(json!({"context":"x", "schema":{
        "type":"object", "properties":{"x":{"type":"boolean", "thinking_effort":"medium"}}, "required":["x"], "additionalProperties":false
    }})).unwrap();
    assert_eq!(
        request.compile().unwrap()[0].thinking,
        Thinking::Enabled(2048)
    );
    let mut request = request;
    request.schema.as_mut().unwrap()["required"] = json!(["missing"]);
    assert!(request.compile().is_err());
}

#[test]
fn integer_equality_does_not_round_distinct_large_values() {
    use turbojev_schema::fields::same_value;
    assert!(!same_value(
        &json!(9007199254740992_u64),
        &json!(9007199254740993_u64)
    ));
    assert!(!same_value(&json!(true), &json!(1)));
    assert!(same_value(&json!(7), &json!(7.0)));
    assert!(!same_value(
        &json!(9007199254740993_u64),
        &json!(9007199254740992.0)
    ));
}

#[test]
fn field_reasoning_overrides_inherit_the_request_budget() {
    let req: GenerateRequest=serde_json::from_value(json!({"context":"x","thinking":true,"thinking_budget":2048,
        "questions":{"a":{"type":"boolean","thinking":true},"b":{"type":"boolean","thinking_budget":32},"c":{"type":"boolean","thinking":false}}})).unwrap();
    let fields = req.compile().unwrap();
    assert_eq!(fields[0].thinking, Thinking::Enabled(2048));
    assert_eq!(fields[1].thinking, Thinking::Enabled(32));
    assert_eq!(fields[2].thinking, Thinking::Disabled);
}

#[test]
fn numeric_conditions_compare_large_integers_without_float_rounding() {
    let fields=compile(json!({"count":{"type":"integer"},"check":{"type":"boolean","when":{"count":{"gt":9007199254740992_u64}}}})).unwrap();
    let check = fields.iter().find(|f| f.name == "check").unwrap();
    assert!(check.conditions_met(&indexmap::IndexMap::from([(
        "count".into(),
        json!(9007199254740993_u64)
    )])));
    assert!(!check.conditions_met(&indexmap::IndexMap::from([(
        "count".into(),
        json!(9007199254740992_u64)
    )])));
}
