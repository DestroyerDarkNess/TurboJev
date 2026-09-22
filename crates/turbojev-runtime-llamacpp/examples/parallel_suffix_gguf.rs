use std::env;

use turbojev::{
    engine::ExecutionBudget,
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
            "usage: parallel_suffix_gguf /path/to/model.gguf",
        )
    })?;

    // Sequence 0 holds the immutable prefix. The remaining sequences are
    // available as isolated parallel suffix branches.
    let runtime = FfiLlamaCpp::open(
        &model_path,
        NativeConfig {
            n_seq_max: 8,
            ..NativeConfig::default()
        },
    )?
    // This estimate is deliberately supplied by the runtime/application, not
    // guessed by TurboJev. It only matters when a memory budget is also set.
    .with_estimated_parallel_task_bytes(32 * 1024 * 1024);

    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "reference-en-v1",
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option and respond with only its code, with no explanation or reasoning.",
        "Which option best matches the supplied evidence?",
    ));
    let backend = llamacpp_backend(renderer, runtime);

    // Auto planning may choose bounded parallel suffixes because this runtime
    // advertises the capability. The budget is portable data, not a device API.
    let jev = TurboJev::new(backend).with_execution_budget(
        ExecutionBudget::unlimited()
            .with_max_parallel_tasks(4)
            .with_memory_budget_bytes(128 * 1024 * 1024),
    );

    let request = EvaluationRequest {
        model: None,
        state: serde_json::json!({
            "message": "My package says delivered but it never arrived",
            "days_since_delivery": 3,
            "customer_tier": "business"
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
            (
                "refund".to_owned(),
                Question::noul("Is a refund an appropriate next action?"),
            ),
            (
                "investigate".to_owned(),
                Question::noul("Should the carrier delivery record be investigated?"),
            ),
            (
                "severity".to_owned(),
                Question::score(
                    "How severe is the customer impact?",
                    vec![
                        "No meaningful impact".to_owned(),
                        "Minor impact".to_owned(),
                        "Significant impact".to_owned(),
                        "Critical impact".to_owned(),
                    ],
                ),
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
