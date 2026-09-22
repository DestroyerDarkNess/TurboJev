use turbojev::TurboJev;
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
    let answer = engine.classify(text, ["billing", "shipping", "technical", "other"])?;
    println!("Department: {}", answer.choice);
    for (label, probability) in answer.probabilities {
        println!("  {label:12} {:5.1}%", probability * 100.0);
    }
    Ok(())
}
