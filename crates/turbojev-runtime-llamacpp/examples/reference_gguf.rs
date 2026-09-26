use std::env;

use indexmap::IndexMap;
use turbojev::{
    schema::{EvaluationRequest, Question},
    TurboJev,
};
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig},
    reference_backend, ReferencePromptConfig, ReferencePromptRenderer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::args().nth(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: reference_gguf /path/to/model.gguf [state]",
        )
    })?;
    let state = env::args()
        .nth(2)
        .unwrap_or_else(|| "My package never arrived".to_owned());

    let runtime = FfiLlamaCpp::open(&model_path, NativeConfig::default())?;
    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "reference-en-v1",
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option and respond with only its code, with no explanation or reasoning.",
        "Which option best matches the supplied evidence?",
    ));
    let backend = reference_backend(renderer, runtime);
    let jev = TurboJev::new(backend);
    let criteria = [
        ("billing", "Payments and refunds"),
        ("shipping", "Delivery"),
        ("technical", "Product issue"),
        ("other", "Other"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect::<IndexMap<_, _>>();
    let request = EvaluationRequest {
        model: None,
        state: serde_json::json!({"message": state}),
        questions: [(
            "department".to_owned(),
            Question::choice("Which department should handle this message?", criteria),
        )]
        .into_iter()
        .collect(),
    };
    let response = jev.evaluate(request)?;
    let turbojev::schema::Answer::Choice {
        choice,
        probabilities,
        ..
    } = response.answers.get("department").expect("answer")
    else {
        unreachable!()
    };

    println!("choice={choice}");
    for (key, probability) in probabilities {
        println!("{key}\t{probability:.6}");
    }
    Ok(())
}
