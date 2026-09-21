use std::env;

use turbojev::TurboJev;
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
    let result = jev.classify(state, ["billing", "shipping", "technical", "other"])?;

    println!("choice={}", result.choice);
    for (key, probability) in result.probabilities {
        println!("{key}\t{probability:.6}");
    }
    Ok(())
}
