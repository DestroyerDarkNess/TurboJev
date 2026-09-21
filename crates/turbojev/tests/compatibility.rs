use indexmap::indexmap;
use serde_json::Value;
use turbojev::{
    backend_mock::FixtureBackend,
    compat::{encode_response, WireProfile},
    schema::{EvaluationRequest, Question},
    TurboJev,
};

#[test]
fn one_canonical_result_projects_to_typesafe_and_ai_sdk_shapes() {
    let backend = FixtureBackend::builder()
        .with_logits("route", vec![0.0, 4.0])
        .with_logits("review", vec![-1.0, 2.0])
        .with_logits("severity", vec![0.0, 1.0, 3.0])
        .build();
    let engine = TurboJev::new(backend);
    let request = EvaluationRequest {
        model: Some("open-model".into()),
        state: serde_json::json!({"message": "pedido perdido"}),
        questions: indexmap! {
            "route".into() => Question::choice("¿Qué equipo?", indexmap! {
                "billing".into() => "Facturación".into(),
                "shipping".into() => "Envíos".into(),
            }),
            "review".into() => Question::noul("¿Requiere revisión humana?"),
            "severity".into() => Question::score(
                "¿Qué gravedad tiene?",
                vec!["baja".into(), "media".into(), "alta".into()],
            ),
        },
    };

    let response = engine.evaluate(request.clone()).unwrap();

    let typesafe: Value = serde_json::from_str(
        &encode_response(WireProfile::TypeSafeSystemOne, &request, &response).unwrap(),
    )
    .unwrap();
    assert_eq!(typesafe["answers"]["review"]["type"], "noul");
    assert!(typesafe["answers"]["review"]["noul"].as_f64().is_some());

    let ai_sdk: Value = serde_json::from_str(
        &encode_response(WireProfile::AiSdkEvaluationV4, &request, &response).unwrap(),
    )
    .unwrap();
    assert!(ai_sdk["answers"]["review"]["probability"].as_f64().is_some());
    assert!(ai_sdk["providerMetadata"]["typesafe"]["confidence"]["route"]
        .as_f64()
        .is_some());
    assert_eq!(
        ai_sdk["providerMetadata"]["turbojev"]["confidenceSource"],
        "selected-candidate-probability"
    );
}
