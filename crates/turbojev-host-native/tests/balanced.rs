use turbojev_host_native::{NativeLoadOptions, NativeSession};

#[test]
fn balanced_load_option_reaches_the_core_through_the_native_json_contract() {
    let options: NativeLoadOptions = serde_json::from_str(r#"{"model":"fixture://order-gate","execution":{"orderRobustness":"balanced"},"fixtureLogits":{"route":[0,0,0]}}"#).unwrap();
    let session = NativeSession::open(options).unwrap();
    let response: serde_json::Value = serde_json::from_str(&session.evaluate_json(r#"{"state":"facts","questions":{"route":{"type":"choice","instructions":"Choose","criteria":{"z":"Z","a":"A","b":"B"}}}}"#).unwrap()).unwrap();
    assert_eq!(response["answers"]["route"]["choice"], "a");
    assert_eq!(response["_turbojev"]["order_robustness"], "balanced");
    assert_eq!(response["_turbojev"]["order_evaluations"], 6);
}
