use indexmap::indexmap;
use turbojev::{
    backend_mock::FixtureBackend,
    conformance::{compare_responses, validate_response, Tolerance},
    engine::PlanningPolicy,
    schema::{EvaluationRequest, Question},
    TurboJev,
};

fn request() -> EvaluationRequest {
    EvaluationRequest {
        model: None,
        state: serde_json::json!({"state": "same evidence"}),
        questions: indexmap! {
            "route".into() => Question::choice("route", indexmap! {
                "a".into() => "A".into(),
                "b".into() => "B".into(),
            }),
            "ok".into() => Question::noul("ok?"),
        },
    }
}

fn backend() -> FixtureBackend {
    FixtureBackend::builder()
        .with_logits("route", vec![0.1, 1.2])
        .with_logits("ok", vec![-0.5, 0.9])
        .build()
}

#[test]
fn optimized_and_reference_paths_can_be_compared_without_runtime_knowledge() {
    let reference = TurboJev::new(backend())
        .with_planning_policy(PlanningPolicy::Reference)
        .evaluate(request())
        .unwrap();
    let optimized = TurboJev::new(backend()).evaluate(request()).unwrap();

    validate_response(&reference).unwrap();
    validate_response(&optimized).unwrap();

    let report = compare_responses(&reference, &optimized, Tolerance::default()).unwrap();
    assert!(report.passed);
    assert_eq!(report.decision_agreements, 2);
}
