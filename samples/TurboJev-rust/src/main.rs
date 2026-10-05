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
    let model = std::env::args().nth(1).ok_or("Pass a GGUF model path")?;
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "My package is broken".into());
    let runtime = FfiLlamaCpp::open(
        &model,
        NativeConfig {
            n_ctx: 8192,
            n_seq_max: 1,
            n_gpu_layers: 0,
            n_threads: 4,
            n_threads_batch: 4,
            ..Default::default()
        },
    )?;
    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "sample-en-v1",
        "Choose exactly one allowed code. Do not explain.",
        "Which department should handle this message?",
    ));
    let engine = TurboJev::new(reference_backend(renderer, runtime));
    if text == "--generate" {
        let request = serde_json::from_str(include_str!("../../typed-fields/request.json"))?;
        println!(
            "{}",
            serde_json::to_string_pretty(&engine.generate(request)?)?
        );
        return Ok(());
    }
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
        state: serde_json::json!({"message": text}),
        questions: [(
            "department".to_owned(),
            Question::choice("Which department should handle this message?", criteria),
        )]
        .into_iter()
        .collect(),
    };
    let response = engine.evaluate(request)?;
    let answer = response
        .answers
        .get("department")
        .expect("department answer");
    if let turbojev::schema::Answer::Choice {
        choice,
        probabilities,
        ..
    } = answer
    {
        println!("Department: {choice}");
        for (label, probability) in probabilities {
            println!("  {label:12} {:5.1}%", probability * 100.0);
        }
    }
    Ok(())
}
