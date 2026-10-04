use std::{env, fs::File, path::PathBuf, time::Instant};

use turbojev::{engine::ExecutionBudget, schema::EvaluationRequest, OrderRobustness, TurboJev};
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig},
    llamacpp_backend, LlamaCppApi, ReferencePromptConfig, ReferencePromptRenderer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or(
        "usage: multimodal_gguf MODEL.gguf MMPROJ.gguf REQUEST.json [REQUEST.json ...] [--cyclic]",
    )?;
    let mmproj = args.next().ok_or("missing MMPROJ.gguf")?;
    let remaining: Vec<_> = args.collect();
    let cyclic = remaining.iter().any(|value| value == "--cyclic");
    let request_paths: Vec<_> = remaining
        .into_iter()
        .filter(|value| value != "--cyclic")
        .map(PathBuf::from)
        .collect();
    if request_paths.is_empty() {
        return Err("provide at least one request JSON".into());
    }
    let requests: Vec<EvaluationRequest> = request_paths
        .iter()
        .map(|path| {
            serde_json::from_reader(File::open(path)?)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
        })
        .collect::<Result<_, _>>()?;

    let runtime = FfiLlamaCpp::open_multimodal(
        &model,
        &mmproj,
        NativeConfig {
            n_ctx: 4096,
            n_batch: 512,
            n_gpu_layers: 0,
            n_threads: 4,
            ..NativeConfig::default()
        },
    )?;
    eprintln!(
        "model={} image={} audio={} video={}",
        runtime.model_id(),
        runtime.image_input_supported(),
        runtime.audio_input_supported(),
        runtime.video_input_supported()
    );

    let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
        "multimodal-smoke-v1",
        "Use the supplied media and context. Choose one listed answer and respond with only its code.",
        "Which description best matches the supplied media?",
    ));
    let engine = TurboJev::new(llamacpp_backend(renderer, runtime))
        .with_execution_budget(ExecutionBudget::unlimited())
        .with_order_robustness(if cyclic {
            OrderRobustness::Cyclic
        } else {
            OrderRobustness::None
        });
    for (index, request) in requests.into_iter().enumerate() {
        let started = Instant::now();
        let response = engine.evaluate(request)?;
        eprintln!(
            "evaluation={} elapsed_ms={}",
            index + 1,
            started.elapsed().as_millis()
        );
        println!("{}", serde_json::to_string(&response)?);
    }
    Ok(())
}
