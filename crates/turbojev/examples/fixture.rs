use indexmap::indexmap;
use turbojev::{
    backend_mock::FixtureBackend,
    schema::{EvaluationRequest, Question},
    TurboJev,
};

fn main() -> Result<(), turbojev::Error> {
    let backend = FixtureBackend::builder()
        .model("example-fixture")
        .with_logits("route", vec![0.1, 3.8, -0.4])
        .with_logits("urgent", vec![2.0, -1.0])
        .build();

    let jev = TurboJev::new(backend);
    let result = jev.evaluate(EvaluationRequest {
        model: None,
        state: serde_json::json!({
            "message": "My package never arrived",
            "days_since_shipment": 12
        }),
        questions: indexmap! {
            "route".into() => Question::choice(
                "Which queue should handle this request?",
                indexmap! {
                    "billing".into() => "Billing or payment issue".into(),
                    "shipping".into() => "Delivery or logistics issue".into(),
                    "other".into() => "None of the above".into(),
                },
            ),
            "urgent".into() => Question::noul(
                "Does this require urgent human intervention?"
            ),
        },
    })?;

    println!("{result:#?}");
    Ok(())
}
