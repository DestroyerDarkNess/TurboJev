use std::{error::Error, fs, path::PathBuf, sync::Arc};

use serde_json::json;
use turbojev_backend::{CandidateProjection, ExecutionPlan, PrefixReuse, TaskScheduling};
use turbojev_runtime::{
    FusedRuntimeExecutor, RuntimeBatch, RuntimeCandidate, RuntimeExecutor, RuntimeTask,
};
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig},
    LlamaCppApi, LlamaCppExecutor,
};
use turbojev_tokenizer::{candidate_symbol_sets, CodebookKind};

const SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Reply with exactly one allowed option code and no explanation.";

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(
        args.next()
            .ok_or("usage: real_fused_qualification <model.gguf> <evidence.json> [rounds]")?,
    );
    let output = PathBuf::from(args.next().ok_or("missing evidence output path")?);
    let rounds: usize = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(12);
    if args.next().is_some() || rounds == 0 {
        return Err("invalid qualification arguments".into());
    }

    let config = NativeConfig {
        n_ctx: 2048,
        n_batch: 1024,
        n_seq_max: 8,
        n_gpu_layers: 0,
        n_threads: 4,
        n_threads_batch: 4,
    };
    let api = Arc::new(
        FfiLlamaCpp::open(model.to_str().ok_or("model path is not UTF-8")?, config)?
            .with_multi_root_enabled(true),
    );
    if !api.multi_root_supported() {
        return Err("loaded llama.cpp context does not expose multi-root execution".into());
    }
    let executor = LlamaCppExecutor::new(Arc::clone(&api));
    let plan = ExecutionPlan {
        prefix_reuse: PrefixReuse::None,
        scheduling: TaskScheduling::Serial,
        parallel_task_limit: 1,
        projection: CandidateProjection::FullVocabulary,
    };

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut response_mismatches = 0_u64;
    let mut isolation_failures = 0_u64;
    let mut non_finite_values = 0_u64;
    let mut requests = 0_u64;

    for round in 0..rounds {
        let mut batches = Vec::new();
        for request_index in 0..4 {
            batches.push(build_batch(api.as_ref(), round, request_index)?);
        }
        let plans = vec![plan; batches.len()];
        if !executor.can_score_fused(&batches, &plans)? {
            return Err(format!(
                "round {round}: runtime rejected an eligible single-token multi-root probe"
            )
            .into());
        }
        let reference = batches
            .iter()
            .map(|batch| executor.score(batch, plan))
            .collect::<Result<Vec<_>, _>>()?;
        let fused = executor.score_fused(&batches, &plans)?;
        compare_outputs(
            &reference,
            &fused,
            &mut max_abs,
            &mut max_rel,
            &mut response_mismatches,
            &mut non_finite_values,
        );
        requests += batches.len() as u64;

        // A fused call must not leak sequence state into the next independent call.
        let after = batches
            .iter()
            .map(|batch| executor.score(batch, plan))
            .collect::<Result<Vec<_>, _>>()?;
        let mut isolation_abs = 0.0_f64;
        let mut isolation_rel = 0.0_f64;
        let mut isolation_argmax = 0_u64;
        let mut isolation_non_finite = 0_u64;
        compare_outputs(
            &reference,
            &after,
            &mut isolation_abs,
            &mut isolation_rel,
            &mut isolation_argmax,
            &mut isolation_non_finite,
        );
        if isolation_abs > 1e-5
            || isolation_rel > 1e-5
            || isolation_argmax != 0
            || isolation_non_finite != 0
        {
            isolation_failures += 1;
        }
    }

    let evidence = json!({
        "rounds": rounds,
        "requests": requests,
        "maxAbsLogitDelta": max_abs,
        "maxRelativeLogitDelta": max_rel,
        "responseMismatches": response_mismatches,
        "isolationFailures": isolation_failures,
        "nonFiniteValues": non_finite_values
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&evidence)?)?;
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}

fn build_batch(
    api: &FfiLlamaCpp,
    round: usize,
    request_index: usize,
) -> Result<RuntimeBatch, Box<dyn Error>> {
    let count = 2 + (request_index % 3);
    for symbols in candidate_symbol_sets(count, 64)? {
        let options = symbols
            .iter()
            .enumerate()
            .map(
                |(index, symbol)| json!({"code": symbol, "description": format!("option-{index}")}),
            )
            .collect::<Vec<_>>();
        let user = serde_json::to_string(&json!({
            "evidence": {"round": round, "request": request_index, "text": format!("qualification-{round}-{request_index}")},
            "criterion": "Choose the option that best matches the evidence.",
            "options": options
        }))?;
        let prompt = api.encode_chat(Some(SYSTEM), &user, None)?;
        let mut candidates = Vec::with_capacity(symbols.len());
        let mut valid = true;
        for symbol in &symbols {
            let full = api.encode_chat(Some(SYSTEM), &user, Some(symbol))?;
            if full.len() != prompt.len() + 1 || !full.starts_with(&prompt) {
                valid = false;
                break;
            }
            let token_ids = full[prompt.len()..].to_vec();
            if api.decode_tokens(&token_ids)? != *symbol {
                valid = false;
                break;
            }
            candidates.push(RuntimeCandidate {
                symbol: symbol.clone(),
                token_ids,
            });
        }
        if valid {
            return Ok(RuntimeBatch {
                common_prefix: Vec::new(),
                tasks: vec![RuntimeTask {
                    id: format!("r{round}-q{request_index}"),
                    prompt_suffix: prompt,
                    candidates,
                    codebook_kind: CodebookKind::SingleToken,
                }],
            });
        }
    }
    Err(format!("could not discover {count} stable single-token candidate codes").into())
}

fn compare_outputs(
    expected: &[turbojev_runtime::RuntimeOutput],
    actual: &[turbojev_runtime::RuntimeOutput],
    max_abs: &mut f64,
    max_rel: &mut f64,
    argmax_mismatches: &mut u64,
    non_finite_values: &mut u64,
) {
    if expected.len() != actual.len() {
        *argmax_mismatches += expected.len().abs_diff(actual.len()) as u64 + 1;
        return;
    }
    for (left, right) in expected.iter().zip(actual) {
        if left.scores.len() != right.scores.len() {
            *argmax_mismatches += left.scores.len().abs_diff(right.scores.len()) as u64 + 1;
            continue;
        }
        for (left_row, right_row) in left.scores.iter().zip(&right.scores) {
            if left_row.id != right_row.id || left_row.logits.len() != right_row.logits.len() {
                *argmax_mismatches += 1;
                continue;
            }
            let mut left_best = None::<(usize, f32)>;
            let mut right_best = None::<(usize, f32)>;
            for (index, (&a, &b)) in left_row.logits.iter().zip(&right_row.logits).enumerate() {
                if !(a.is_finite() && b.is_finite()) {
                    *non_finite_values += 1;
                    continue;
                }
                let abs = f64::from((a - b).abs());
                let denom = f64::from(a.abs()).max(f64::from(b.abs())).max(1e-12);
                *max_abs = (*max_abs).max(abs);
                *max_rel = (*max_rel).max(abs / denom);
                if left_best.map_or(true, |(_, value)| a > value) {
                    left_best = Some((index, a));
                }
                if right_best.map_or(true, |(_, value)| b > value) {
                    right_best = Some((index, b));
                }
            }
            if left_best.map(|x| x.0) != right_best.map(|x| x.0) {
                *argmax_mismatches += 1;
            }
        }
    }
}
