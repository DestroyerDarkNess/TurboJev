use std::time::Duration;
use turbojev_host_native::{GenerationControl, HostError, NativeSession};

#[test]
fn compiler_and_normalizer_need_no_model() {
    let compiled = NativeSession::compile_schema_json(r#"{"type":"object","properties":{"person":{"type":"object","properties":{"age":{"type":"integer"}}}}}"#).unwrap();
    let fields: serde_json::Value = serde_json::from_str(&compiled).unwrap();
    assert_eq!(fields[0]["path"], serde_json::json!(["person", "age"]));
    let probabilities: serde_json::Value = serde_json::from_str(
        &NativeSession::candidate_softmax_json(
            r#"{"logits":{"a":10000,"b":10000},"temperature":0.0001}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(probabilities, serde_json::json!({"a":0.5,"b":0.5}));
    assert!(
        NativeSession::candidate_softmax_json(r#"{"logits":{"a":0},"temperature":0}"#).is_err()
    );
}

#[test]
fn cancelled_and_expired_calls_do_not_start_inference() {
    let session = NativeSession::open_json(
        r#"{"model":"fixture://controls","fixtureLogits":{"flag":[-5,5]}}"#,
    )
    .unwrap();
    let input = r#"{"context":"facts","questions":{"flag":{"type":"boolean"}}}"#;
    let control = GenerationControl::new(None).unwrap();
    control.cancel();
    assert!(matches!(
        session.generate_json_controlled(input, &control),
        Err(HostError::GenerationCancelled)
    ));
    let control = GenerationControl::new(Some(Duration::ZERO)).unwrap();
    assert!(matches!(
        session.generate_json_controlled(input, &control),
        Err(HostError::GenerationTimeout)
    ));
    assert!(session.generate_json(input).is_ok());
}
