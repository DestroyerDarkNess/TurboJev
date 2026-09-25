use serde_json::{json, Value};
use turbojev_host_native::NativeSession;

fn open(mode: &str, timeout: u64) -> NativeSession {
    NativeSession::open_json(
        &json!({
            "model": "protocol-fixture",
            "execution": {"orderRobustness": "none"},
            "worker": {
                "program": std::env::var("PYTHON").unwrap_or_else(|_| "python".into()),
                "args": [format!("{}/tests/scorer_fixture.py", env!("CARGO_MANIFEST_DIR")), mode],
                "timeoutSeconds": timeout,
            }
        })
        .to_string(),
    )
    .unwrap()
}

fn request(state: Value) -> String {
    json!({"state": state, "questions": {
        "choice": {"type":"choice", "instructions":"Choose", "criteria":{"a":"A", "b":"B"}},
        "boolean": {"type":"noul", "instructions":"True?"},
        "score": {"type":"score", "instructions":"Rate", "criteria":["low", "high"]}
    }})
    .to_string()
}

#[test]
fn worker_preserves_core_semantics_for_each_modality_and_reuses_process() {
    let session = open("normal", 10);
    assert!(!session.execution_capabilities().fused_batch);
    let expected: Value =
        serde_json::from_str(&session.evaluate_json(&request(json!("text"))).unwrap()).unwrap();
    for modality in ["image", "audio", "video"] {
        let state = json!({"contract":"turbojev.media.v1", "modality":modality, "base64":"AA=="});
        let output: Value =
            serde_json::from_str(&session.evaluate_json(&request(state)).unwrap()).unwrap();
        assert_eq!(output, expected);
        assert_eq!(output["answers"]["choice"]["choice"], "b");
        assert_eq!(output["usage"]["generated_tokens"], 0);
    }
}

#[test]
fn worker_errors_propagate_and_timeouts_invalidate_the_process() {
    let session = open("error", 10);
    assert!(session
        .evaluate_json(&request(json!("text")))
        .unwrap_err()
        .to_string()
        .contains("unsupported codec"));
    let session = open("timeout", 1);
    assert!(session
        .evaluate_json(&request(json!("text")))
        .unwrap_err()
        .to_string()
        .contains("timed out"));
}

#[test]
fn unresponsive_worker_cannot_block_large_media_writes() {
    let session = open("no_read", 1);
    let state = json!({"contract":"turbojev.media.v1", "modality":"video", "base64":"A".repeat(1024 * 1024)});
    assert!(session
        .evaluate_json(&request(state))
        .unwrap_err()
        .to_string()
        .contains("timed out"));
}
