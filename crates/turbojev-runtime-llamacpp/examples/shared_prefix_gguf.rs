use std::env;

use turbojev::{
    engine::PlanningPolicy,
    schema::{EvaluationRequest, Question},
    TurboJev,
};
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig},
    llamacpp_backend, ReferencePromptConfig, ReferencePromptRenderer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::args().nth(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: shared_prefix_gguf /path/to/model.gguf",
        )
    })?;

    let runtime = FfiLlamaCpp::open(&model_path, NativeConfig::default())?;
    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "reference-en-v1",
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option and respond with only its code, with no explanation or reasoning.",
        "Which option best matches the supplied evidence?",
    ));
    let backend = llamacpp_backend(renderer, runtime);
    let jev = TurboJev::new(backend).with_planning_policy(PlanningPolicy::DeterministicReuse);

    let request = EvaluationRequest {
        model: None,
        state: serde_json::json!({
            "message": "My package says delivered but it never arrived",
            "days_since_delivery": 3
        }),
        questions: [
            (
                "route".to_owned(),
                Question::choice(
                    "Which team should handle this request?",
                    [
                        ("billing".to_owned(), "Billing or payment issue".to_owned()),
                        (
                            "shipping".to_owned(),
                            "Delivery or logistics issue".to_owned(),
                        ),
                        ("technical".to_owned(), "Technical product issue".to_owned()),
                    ]
                    .into_iter()
                    .collect(),
                ),
            ),
            (
                "urgent".to_owned(),
                Question::noul("Does this require urgent human review?"),
            ),
        ]
        .into_iter()
        .collect(),
    };

    let result = jev.evaluate(request)?;
    println!("execution={:?}", result.metadata.execution);
    println!("usage={:?}", result.usage);
    println!("answers={:#?}", result.answers);
    Ok(())
}
