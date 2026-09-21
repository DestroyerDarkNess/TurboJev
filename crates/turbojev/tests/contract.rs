use indexmap::indexmap;
use turbojev::{
    backend_mock::FixtureBackend,
    schema::{Answer, EvaluationRequest, Question},
    TurboJev,
};

#[test]
fn public_contract_preserves_question_order_and_types() {
    let backend = FixtureBackend::builder()
        .with_logits("route", vec![0.0, 5.0])
        .with_logits("ok", vec![-1.0, 2.0])
        .with_logits("severity", vec![0.0, 0.5, 3.0])
        .build();

    let engine = TurboJev::new(backend);
    let response = engine
        .evaluate(EvaluationRequest {
            model: Some("requested-model".to_owned()),
            state: serde_json::json!({"ticket": "late delivery"}),
            questions: indexmap! {
                "route".into() => Question::choice(
                    "Which queue?",
                    indexmap! {
                        "billing".into() => "Billing".into(),
                        "shipping".into() => "Shipping".into(),
                    },
                ),
                "ok".into() => Question::noul("Is the condition true?"),
                "severity".into() => Question::score(
                    "How severe?",
                    vec!["low".into(), "medium".into(), "high".into()],
                ),
            },
        })
        .unwrap();

    assert_eq!(
        response.answers.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["route", "ok", "severity"]
    );
    assert!(matches!(response.answers["route"], Answer::Choice { .. }));
    assert!(matches!(response.answers["ok"], Answer::Noul { .. }));
    assert!(matches!(response.answers["severity"], Answer::Score { .. }));
    assert_eq!(response.usage.generated_tokens, 0);
    assert!(!response.metadata.calibrated);
}
