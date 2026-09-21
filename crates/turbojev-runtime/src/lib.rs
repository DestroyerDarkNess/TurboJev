#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Universal token-level bridge between TurboJev semantics and model runtimes.
//!
//! Runtime adapters implement three narrow concerns:
//! - [`PromptCodec`]: apply the checkpoint's chat/instruction template and tokenizer;
//! - [`RuntimeExecutor`]: score already-tokenized candidate sequences;
//! - [`PromptRenderer`]: supplied by `turbojev-prompt`, render semantic prompt IR.
//!
//! [`RuntimeBackend`] composes those concerns into TurboJev's high-level
//! [`ScoringBackend`] contract. This keeps GGUF, safetensors, WebGPU, MLX,
//! ONNX, Candle, llama.cpp, mobile NPUs and future runtimes outside the core.

use std::{collections::HashSet, future::Future, pin::Pin};

use thiserror::Error;
use turbojev_backend::{
    AsyncScoringBackend, BackendCapabilities, BackendError, BackendFuture, BackendMetrics,
    BackendOutput, BackendResourceHints, CandidateProjection, DecisionBatch, ExecutionPlan,
    FusedBackendCapabilities, FusedScoringBackend, PrefixReuse, ScoringBackend, TaskScheduling,
    TaskScores,
};
use turbojev_prompt::{compile_document, PromptRenderer, RenderedMessages};
use turbojev_tokenizer::{candidate_symbol_sets, CodebookKind};

/// Runtime-neutral token identifier.
pub type TokenId = u32;

/// Capabilities of one loaded runtime/model combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeCapabilities {
    /// Exact scores for the next token are available.
    pub next_token_logits: bool,
    /// Exact log-scores can be accumulated for candidate sequences longer than one token.
    pub sequence_scores: bool,
    /// A common token prefix can be prefetched once and reused.
    pub shared_prefix: bool,
    /// Independent suffixes can be scored as a batch.
    pub parallel_suffixes: bool,
    /// Only the candidate output rows can be projected instead of the full vocabulary.
    pub selective_output_head: bool,
    /// Execution occurs in the host application/device rather than a required service.
    pub local_inference: bool,
}

impl RuntimeCapabilities {
    /// Convert low-level runtime capabilities into the stable backend SPI.
    pub const fn backend_capabilities(self) -> BackendCapabilities {
        BackendCapabilities {
            shared_prefix: self.shared_prefix,
            parallel_suffixes: self.parallel_suffixes,
            selective_output_head: self.selective_output_head,
            multi_token_candidates: self.sequence_scores,
            local_inference: self.local_inference,
        }
    }
}

/// Encoded candidate attached to one decision task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCandidate {
    /// Opaque textual code rendered in the model-facing prompt.
    pub symbol: String,
    /// Exact token sequence scored at the assistant/answer boundary.
    pub token_ids: Vec<TokenId>,
}

/// One fully prepared decision suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTask {
    /// Stable task identifier.
    pub id: String,
    /// Prompt tokens after [`RuntimeBatch::common_prefix`].
    pub prompt_suffix: Vec<TokenId>,
    /// Candidate codes in the same order as the semantic task candidates.
    pub candidates: Vec<RuntimeCandidate>,
    /// Whether every candidate is a single token.
    pub codebook_kind: CodebookKind,
}

/// Token-level batch passed to a runtime executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeBatch {
    /// Exact common prefix factored from all task prompts when requested.
    pub common_prefix: Vec<TokenId>,
    /// Independent decision suffixes.
    pub tasks: Vec<RuntimeTask>,
}

/// Runtime score row. Values are unnormalized candidate logits/log-scores.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeTaskScores {
    /// Stable task identifier.
    pub id: String,
    /// Candidate values in request order.
    pub logits: Vec<f32>,
}

/// Result produced by a low-level runtime executor.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeOutput {
    /// Resolved model/checkpoint identity, when known.
    pub model: Option<String>,
    /// One row per runtime task.
    pub scores: Vec<RuntimeTaskScores>,
    /// Portable execution counters.
    pub metrics: BackendMetrics,
}

/// Applies the checkpoint's exact chat template and tokenizer.
///
/// `encode_prompt_with_answer` is intentionally explicit. TurboJev must verify
/// candidate codes at the *real* assistant answer boundary; concatenating text
/// outside a chat template is not sufficient for all tokenizers.
pub trait PromptCodec {
    /// Stable codec/tokenizer profile identifier.
    fn profile(&self) -> &str;

    /// Encode the rendered conversation up to the exact answer boundary.
    fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError>;

    /// Encode the same prompt with one candidate code as the complete answer.
    fn encode_prompt_with_answer(
        &self,
        messages: &RenderedMessages,
        answer: &str,
    ) -> Result<Vec<TokenId>, RuntimeError>;

    /// Decode an exact token sequence for round-trip verification.
    fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError>;
}

/// Executes already prepared token-level decision batches.
pub trait RuntimeExecutor {
    /// Stable runtime implementation identifier.
    fn name(&self) -> &'static str;

    /// Capabilities of the loaded runtime/model combination.
    fn capabilities(&self) -> RuntimeCapabilities;

    /// Quantitative resource hints for portable execution planning.
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints::default()
    }

    /// Score candidate sequences for every task.
    fn score(
        &self,
        batch: &RuntimeBatch,
        plan: ExecutionPlan,
    ) -> Result<RuntimeOutput, RuntimeError>;
}

/// Optional runtime SPI for true cross-request model batching.
///
/// Implementations receive multiple independently prepared runtime batches in a
/// single call. An adapter must not implement this trait by calling `score` in a
/// loop and advertising the result as fused execution.
pub trait FusedRuntimeExecutor: RuntimeExecutor {
    /// Truthful cross-request limits for the loaded model/runtime.
    fn fused_capabilities(&self) -> FusedBackendCapabilities;

    /// Return whether the current prepared workload can use the fused path.
    /// This keeps globally supported runtimes truthful when some request shapes
    /// require the independent reference path.
    fn can_score_fused(
        &self,
        _batches: &[RuntimeBatch],
        _plans: &[ExecutionPlan],
    ) -> Result<bool, RuntimeError> {
        Ok(self.fused_capabilities().fused_batch)
    }

    /// Score multiple independent runtime batches in one underlying runtime call.
    fn score_fused(
        &self,
        batches: &[RuntimeBatch],
        plans: &[ExecutionPlan],
    ) -> Result<Vec<RuntimeOutput>, RuntimeError>;
}

/// Runtime-neutral future returned by asynchronous model executors.
///
/// No executor framework is required and the future is intentionally not
/// `Send`; WebGPU/JS and mobile runtimes may be thread-affine.
pub type RuntimeFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Asynchronous low-level runtime executor.
///
/// Adapters for browser WebGPU, JavaScript-hosted engines, mobile runtimes or
/// any naturally asynchronous accelerator can implement this trait directly
/// without introducing Tokio, async-std or a platform-specific scheduler.
pub trait AsyncRuntimeExecutor {
    /// Stable runtime implementation identifier.
    fn name(&self) -> &'static str;

    /// Capabilities of the loaded runtime/model combination.
    fn capabilities(&self) -> RuntimeCapabilities;

    /// Quantitative resource hints for portable execution planning.
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints::default()
    }

    /// Score candidate sequences asynchronously.
    fn score_async<'a>(
        &'a self,
        batch: &'a RuntimeBatch,
        plan: ExecutionPlan,
    ) -> RuntimeFuture<'a, Result<RuntimeOutput, RuntimeError>>;
}

/// Portable composition of semantic prompt rendering, tokenization and runtime execution.
pub struct RuntimeBackend<R, C, E> {
    renderer: R,
    codec: C,
    executor: E,
    max_tokens_per_code: usize,
    candidate_set_attempts: usize,
}

impl<R, C, E> RuntimeBackend<R, C, E> {
    /// Compose a model-facing renderer, codec/tokenizer and runtime executor.
    pub fn new(renderer: R, codec: C, executor: E) -> Self {
        Self {
            renderer,
            codec,
            executor,
            max_tokens_per_code: 4,
            candidate_set_attempts: 96,
        }
    }

    /// Limit candidate code sequence width. Values above one require executor
    /// sequence scoring when a single-token codebook cannot be found.
    pub fn with_max_tokens_per_code(mut self, max_tokens_per_code: usize) -> Self {
        self.max_tokens_per_code = max_tokens_per_code;
        self
    }

    /// Limit deterministic candidate-code search attempts.
    pub fn with_candidate_set_attempts(mut self, attempts: usize) -> Self {
        self.candidate_set_attempts = attempts;
        self
    }

    /// Borrow the renderer.
    pub fn renderer(&self) -> &R {
        &self.renderer
    }

    /// Borrow the prompt codec/tokenizer.
    pub fn codec(&self) -> &C {
        &self.codec
    }

    /// Borrow the runtime executor.
    pub fn executor(&self) -> &E {
        &self.executor
    }
}

impl<R, C, E> ScoringBackend for RuntimeBackend<R, C, E>
where
    R: PromptRenderer,
    C: PromptCodec,
    E: RuntimeExecutor,
{
    fn name(&self) -> &'static str {
        self.executor.name()
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.executor.capabilities().backend_capabilities()
    }

    fn resource_hints(&self) -> BackendResourceHints {
        self.executor.resource_hints()
    }

    fn score(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
    ) -> Result<BackendOutput, BackendError> {
        let capabilities = validate_runtime_execution(
            self.executor.capabilities(),
            plan,
            self.max_tokens_per_code,
            self.candidate_set_attempts,
        )?;
        let prepared = self
            .prepare_batch(batch, plan, capabilities)
            .map_err(map_runtime_error)?;
        let effective_plan = effective_plan_for_codebooks(plan, &prepared);
        if !capabilities.backend_capabilities().supports(effective_plan) {
            return Err(BackendError::UnsupportedPlan);
        }
        let mut output = self
            .executor
            .score(&prepared, effective_plan)
            .map_err(map_runtime_error)?;
        if effective_plan != plan {
            output.metrics.effective_plan = Some(effective_plan);
        }
        Ok(map_runtime_output(output))
    }
}

impl<R, C, E> FusedScoringBackend for RuntimeBackend<R, C, E>
where
    R: PromptRenderer,
    C: PromptCodec,
    E: FusedRuntimeExecutor,
{
    fn fused_capabilities(&self) -> FusedBackendCapabilities {
        self.executor.fused_capabilities()
    }

    fn can_score_fused(
        &self,
        batches: &[DecisionBatch],
        plans: &[ExecutionPlan],
    ) -> Result<bool, BackendError> {
        let fused = self.executor.fused_capabilities().validate()?;
        if !fused.fused_batch || batches.len() < 2 || batches.len() != plans.len() {
            return Ok(false);
        }
        if let Some(maximum) = fused.max_fused_items {
            if batches.len() > maximum {
                return Ok(false);
            }
        }
        let runtime_capabilities = self.executor.capabilities();
        let mut prepared = Vec::with_capacity(batches.len());
        let mut effective_plans = Vec::with_capacity(plans.len());
        for (batch, plan) in batches.iter().zip(plans) {
            let capabilities = validate_runtime_execution(
                runtime_capabilities,
                *plan,
                self.max_tokens_per_code,
                self.candidate_set_attempts,
            )
            .map_err(map_runtime_error)?;
            let runtime_batch = self
                .prepare_batch(batch, *plan, capabilities)
                .map_err(map_runtime_error)?;
            let effective = effective_plan_for_codebooks(*plan, &runtime_batch);
            if !capabilities.backend_capabilities().supports(effective) {
                return Ok(false);
            }
            prepared.push(runtime_batch);
            effective_plans.push(effective);
        }
        self.executor
            .can_score_fused(&prepared, &effective_plans)
            .map_err(map_runtime_error)
    }

    fn score_fused(
        &self,
        batches: &[DecisionBatch],
        plans: &[ExecutionPlan],
    ) -> Result<Vec<BackendOutput>, BackendError> {
        let fused = self
            .executor
            .fused_capabilities()
            .validate()?;
        if !fused.fused_batch {
            return Err(BackendError::Unsupported(
                "runtime does not advertise fused cross-request execution".to_owned(),
            ));
        }
        if batches.len() < 2 || batches.len() != plans.len() {
            return Err(BackendError::InvalidInput(
                "fused runtime execution requires matching batch/plan arrays with at least two items".to_owned(),
            ));
        }
        if let Some(maximum) = fused.max_fused_items {
            if batches.len() > maximum {
                return Err(BackendError::InvalidInput(format!(
                    "fused runtime received {} requests; maximum is {}",
                    batches.len(), maximum
                )));
            }
        }

        let runtime_capabilities = self.executor.capabilities();
        let mut prepared = Vec::with_capacity(batches.len());
        let mut effective_plans = Vec::with_capacity(batches.len());
        for (batch, plan) in batches.iter().zip(plans) {
            let capabilities = validate_runtime_execution(
                runtime_capabilities,
                *plan,
                self.max_tokens_per_code,
                self.candidate_set_attempts,
            )
            .map_err(map_runtime_error)?;
            let runtime_batch = self
                .prepare_batch(batch, *plan, capabilities)
                .map_err(map_runtime_error)?;
            let effective = effective_plan_for_codebooks(*plan, &runtime_batch);
            if !capabilities.backend_capabilities().supports(effective) {
                return Err(BackendError::UnsupportedPlan);
            }
            prepared.push(runtime_batch);
            effective_plans.push(effective);
        }

        if !self
            .executor
            .can_score_fused(&prepared, &effective_plans)
            .map_err(map_runtime_error)?
        {
            return Err(BackendError::Unsupported(
                "prepared runtime workload is not eligible for fused execution".to_owned(),
            ));
        }

        let outputs = self
            .executor
            .score_fused(&prepared, &effective_plans)
            .map_err(map_runtime_error)?;
        if outputs.len() != batches.len() {
            return Err(BackendError::Inference(format!(
                "fused runtime returned {} outputs for {} requests",
                outputs.len(), batches.len()
            )));
        }

        Ok(outputs
            .into_iter()
            .zip(plans.iter().zip(effective_plans))
            .map(|(mut output, (requested, effective))| {
                if *requested != effective {
                    output.metrics.effective_plan = Some(effective);
                }
                map_runtime_output(output)
            })
            .collect())
    }
}

impl<R, C, E> AsyncScoringBackend for RuntimeBackend<R, C, E>
where
    R: PromptRenderer,
    C: PromptCodec,
    E: AsyncRuntimeExecutor,
{
    fn name(&self) -> &'static str {
        self.executor.name()
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.executor.capabilities().backend_capabilities()
    }

    fn resource_hints(&self) -> BackendResourceHints {
        self.executor.resource_hints()
    }

    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>> {
        Box::pin(async move {
            let capabilities = validate_runtime_execution(
                self.executor.capabilities(),
                plan,
                self.max_tokens_per_code,
                self.candidate_set_attempts,
            )?;
            let prepared = self
                .prepare_batch(batch, plan, capabilities)
                .map_err(map_runtime_error)?;
            let effective_plan = effective_plan_for_codebooks(plan, &prepared);
            if !capabilities.backend_capabilities().supports(effective_plan) {
                return Err(BackendError::UnsupportedPlan);
            }
            let mut output = self
                .executor
                .score_async(&prepared, effective_plan)
                .await
                .map_err(map_runtime_error)?;
            if effective_plan != plan {
                output.metrics.effective_plan = Some(effective_plan);
            }
            Ok(map_runtime_output(output))
        })
    }
}

impl<R, C, E> RuntimeBackend<R, C, E>
where
    R: PromptRenderer,
    C: PromptCodec,
{
    fn prepare_batch(
        &self,
        batch: &DecisionBatch,
        plan: ExecutionPlan,
        capabilities: RuntimeCapabilities,
    ) -> Result<RuntimeBatch, RuntimeError> {
        let mut prepared = Vec::with_capacity(batch.tasks.len());
        for task in &batch.tasks {
            let prepared_task = self.prepare_task(&batch.state, task, capabilities)?;
            prepared.push(prepared_task);
        }

        let prompts: Vec<&[TokenId]> = prepared
            .iter()
            .map(|task| task.prompt_suffix.as_slice())
            .collect();
        let prefix_len = if plan.prefix_reuse == PrefixReuse::Shared && prompts.len() > 1 {
            longest_common_prefix_len(&prompts)
        } else {
            0
        };
        let common_prefix = prepared
            .first()
            .map(|task| task.prompt_suffix[..prefix_len].to_vec())
            .unwrap_or_default();
        if prefix_len > 0 {
            for task in &mut prepared {
                drop(task.prompt_suffix.drain(..prefix_len));
            }
        }

        Ok(RuntimeBatch {
            common_prefix,
            tasks: prepared,
        })
    }

    fn prepare_task(
        &self,
        state: &serde_json::Value,
        task: &turbojev_backend::DecisionTask,
        capabilities: RuntimeCapabilities,
    ) -> Result<RuntimeTask, RuntimeError> {
        let candidate_sets = candidate_symbol_sets(
            task.candidates.len(),
            self.candidate_set_attempts,
        )
        .map_err(|error| RuntimeError::Preparation(error.to_string()))?;

        let mut best_multi: Option<(Vec<TokenId>, Vec<RuntimeCandidate>, CodebookKind)> = None;
        for symbols in candidate_sets {
            let messages = self.render_with_codes(state, task, &symbols)?;
            let base = self.codec.encode_prompt(&messages)?;
            let codes = match self.verify_codes(&messages, &base, &symbols) {
                Ok(codes) => codes,
                Err(RuntimeError::CandidateCodeRejected(_)) => continue,
                Err(error) => return Err(error),
            };
            if codes.iter().all(|code| code.token_ids.len() == 1) {
                return Ok(RuntimeTask {
                    id: task.id.clone(),
                    prompt_suffix: base,
                    candidates: codes,
                    codebook_kind: CodebookKind::SingleToken,
                });
            }
            if capabilities.sequence_scores {
                let max_width = codes
                    .iter()
                    .map(|code| code.token_ids.len())
                    .max()
                    .unwrap_or(0);
                if max_width <= self.max_tokens_per_code && is_prefix_free(&codes) {
                    let first_width = codes[0].token_ids.len();
                    let kind = if codes
                        .iter()
                        .all(|code| code.token_ids.len() == first_width)
                    {
                        CodebookKind::FixedLengthSequence {
                            tokens_per_code: max_width,
                        }
                    } else {
                        // Prefer equal-width sequences when possible, but preserve a
                        // verified prefix-free variable-width fallback.
                        if best_multi.is_none() {
                            best_multi = Some((
                                base,
                                codes,
                                CodebookKind::VariableLengthSequence {
                                    max_tokens_per_code: max_width,
                                },
                            ));
                        }
                        continue;
                    };
                    return Ok(RuntimeTask {
                        id: task.id.clone(),
                        prompt_suffix: base,
                        candidates: codes,
                        codebook_kind: kind,
                    });
                }
            }
        }

        if let Some((base, codes, kind)) = best_multi {
            return Ok(RuntimeTask {
                id: task.id.clone(),
                prompt_suffix: base,
                candidates: codes,
                codebook_kind: kind,
            });
        }

        Err(RuntimeError::NoCompatibleCodebook {
            task: task.id.clone(),
            candidates: task.candidates.len(),
        })
    }

    fn render_with_codes(
        &self,
        state: &serde_json::Value,
        task: &turbojev_backend::DecisionTask,
        symbols: &[String],
    ) -> Result<RenderedMessages, RuntimeError> {
        let document = compile_document(state, task, symbols)
            .map_err(|error| RuntimeError::Preparation(error.to_string()))?;
        let messages = self
            .renderer
            .render(&document)
            .map_err(|error| RuntimeError::Preparation(error.to_string()))?;
        Ok(messages)
    }

    fn verify_codes(
        &self,
        messages: &RenderedMessages,
        base: &[TokenId],
        symbols: &[String],
    ) -> Result<Vec<RuntimeCandidate>, RuntimeError> {
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            let full = self.codec.encode_prompt_with_answer(messages, symbol)?;
            if full.len() <= base.len() || !full.starts_with(base) {
                return Err(RuntimeError::CandidateCodeRejected(symbol.clone()));
            }
            let suffix = full[base.len()..].to_vec();
            if suffix.is_empty() || suffix.len() > self.max_tokens_per_code {
                return Err(RuntimeError::CandidateCodeRejected(symbol.clone()));
            }
            if self.codec.decode(&suffix)? != *symbol {
                return Err(RuntimeError::CandidateCodeRejected(symbol.clone()));
            }
            if !seen.insert(suffix.clone()) {
                return Err(RuntimeError::CandidateCodeRejected(symbol.clone()));
            }
            result.push(RuntimeCandidate {
                symbol: symbol.clone(),
                token_ids: suffix,
            });
        }
        Ok(result)
    }
}


fn effective_plan_for_codebooks(plan: ExecutionPlan, batch: &RuntimeBatch) -> ExecutionPlan {
    let requires_sequence_scoring = batch.tasks.iter().any(|task| {
        !matches!(task.codebook_kind, CodebookKind::SingleToken)
            || task.candidates.iter().any(|candidate| candidate.token_ids.len() != 1)
    });
    if !requires_sequence_scoring {
        return plan;
    }

    // Exact sequence probabilities require the full-vocabulary normalizer at
    // every continuation state. Until a runtime advertises an explicit exact
    // selective-sequence primitive, candidate-only projection is unsafe.
    // Likewise, v0.15 keeps task-level sequence traversal deterministic/serial;
    // candidate prefixes are still shared inside each trie.
    ExecutionPlan {
        prefix_reuse: plan.prefix_reuse,
        scheduling: TaskScheduling::Serial,
        parallel_task_limit: 1,
        projection: CandidateProjection::FullVocabulary,
    }
}

fn validate_runtime_execution(
    capabilities: RuntimeCapabilities,
    plan: ExecutionPlan,
    max_tokens_per_code: usize,
    candidate_set_attempts: usize,
) -> Result<RuntimeCapabilities, BackendError> {
    if max_tokens_per_code == 0 {
        return Err(BackendError::InvalidInput(
            "max_tokens_per_code must be at least 1".to_owned(),
        ));
    }
    if candidate_set_attempts == 0 {
        return Err(BackendError::InvalidInput(
            "candidate_set_attempts must be at least 1".to_owned(),
        ));
    }
    if !capabilities.backend_capabilities().supports(plan) {
        return Err(BackendError::UnsupportedPlan);
    }
    if !capabilities.next_token_logits {
        return Err(BackendError::Unsupported(
            "runtime does not expose next-token candidate scores".to_owned(),
        ));
    }
    Ok(capabilities)
}

fn map_runtime_output(output: RuntimeOutput) -> BackendOutput {
    BackendOutput {
        model: output.model,
        scores: output
            .scores
            .into_iter()
            .map(|row| TaskScores {
                id: row.id,
                logits: row.logits,
            })
            .collect(),
        metrics: output.metrics,
    }
}

fn is_prefix_free(codes: &[RuntimeCandidate]) -> bool {
    for (left_index, left) in codes.iter().enumerate() {
        for (right_index, right) in codes.iter().enumerate() {
            if left_index != right_index && right.token_ids.starts_with(&left.token_ids) {
                return false;
            }
        }
    }
    true
}

fn longest_common_prefix_len(rows: &[&[TokenId]]) -> usize {
    let Some(first) = rows.first() else {
        return 0;
    };
    let max = rows.iter().map(|row| row.len()).min().unwrap_or(0);
    for index in 0..max {
        let expected = first[index];
        if rows.iter().any(|row| row[index] != expected) {
            return index;
        }
    }
    max
}

fn map_runtime_error(error: RuntimeError) -> BackendError {
    match error {
        RuntimeError::Unsupported(message) => BackendError::Unsupported(message),
        RuntimeError::Inference(message) => BackendError::Inference(message),
        other => BackendError::InvalidInput(other.to_string()),
    }
}

/// Universal runtime bridge error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// Prompt preparation failed.
    #[error("runtime preparation failed: {0}")]
    Preparation(String),
    /// Chat template/tokenizer failed.
    #[error("runtime prompt codec failed: {0}")]
    Codec(String),
    /// Candidate code did not round-trip at the exact answer boundary.
    #[error("candidate code rejected at answer boundary: {0}")]
    CandidateCodeRejected(String),
    /// No safe candidate codebook was found.
    #[error("no compatible codebook for task {task:?} with {candidates} candidates")]
    NoCompatibleCodebook { task: String, candidates: usize },
    /// The loaded runtime/model cannot implement an operation.
    #[error("runtime operation unsupported: {0}")]
    Unsupported(String),
    /// Model execution failed.
    #[error("runtime inference failed: {0}")]
    Inference(String),
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use turbojev_backend::{Candidate, DecisionIntent, DecisionTask, TaskScheduling};
    use turbojev_prompt::{PromptCriterion, PromptDocument, PromptError};

    use super::*;

    struct JsonRenderer;

    impl PromptRenderer for JsonRenderer {
        fn profile(&self) -> &str {
            "test-json"
        }

        fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError> {
            let criterion = match &document.criterion {
                PromptCriterion::UserText(value) => value.clone(),
                PromptCriterion::BestMatch => "best-match".to_owned(),
            };
            let mut user = format!("state={}\ncriterion={criterion}\n", document.evidence);
            for option in &document.options {
                user.push_str(&format!("{}={}\n", option.code, option.description));
            }
            user.push_str("answer=");
            Ok(RenderedMessages {
                system: None,
                user,
                profile: "test-json".to_owned(),
            })
        }
    }

    struct CharacterCodec;

    impl CharacterCodec {
        fn encode_text(text: &str) -> Vec<TokenId> {
            text.chars().map(u32::from).collect()
        }
    }

    impl PromptCodec for CharacterCodec {
        fn profile(&self) -> &str {
            "unicode-scalar-test"
        }

        fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError> {
            Ok(Self::encode_text(&messages.user))
        }

        fn encode_prompt_with_answer(
            &self,
            messages: &RenderedMessages,
            answer: &str,
        ) -> Result<Vec<TokenId>, RuntimeError> {
            Ok(Self::encode_text(&format!("{}{answer}", messages.user)))
        }

        fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError> {
            token_ids
                .iter()
                .map(|value| char::from_u32(*value).ok_or_else(|| RuntimeError::Codec("invalid scalar".into())))
                .collect()
        }
    }

    struct RecordingExecutor {
        last_batch: Mutex<Option<RuntimeBatch>>,
    }

    impl RecordingExecutor {
        fn new() -> Self {
            Self {
                last_batch: Mutex::new(None),
            }
        }
    }

    impl RuntimeExecutor for RecordingExecutor {
        fn name(&self) -> &'static str {
            "recording-runtime"
        }

        fn capabilities(&self) -> RuntimeCapabilities {
            RuntimeCapabilities {
                next_token_logits: true,
                sequence_scores: true,
                shared_prefix: true,
                parallel_suffixes: true,
                selective_output_head: true,
                local_inference: true,
            }
        }

        fn score(
            &self,
            batch: &RuntimeBatch,
            plan: ExecutionPlan,
        ) -> Result<RuntimeOutput, RuntimeError> {
            assert_eq!(plan.scheduling, TaskScheduling::Parallel);
            *self.last_batch.lock().unwrap() = Some(batch.clone());
            Ok(RuntimeOutput {
                model: Some("open-model-test".into()),
                scores: batch
                    .tasks
                    .iter()
                    .map(|task| RuntimeTaskScores {
                        id: task.id.clone(),
                        logits: (0..task.candidates.len()).map(|index| index as f32).collect(),
                    })
                    .collect(),
                metrics: BackendMetrics {
                    input_tokens: Some(
                        batch.common_prefix.len()
                            + batch.tasks.iter().map(|task| task.prompt_suffix.len()).sum::<usize>(),
                    ),
                    ..BackendMetrics::default()
                },
            })
        }
    }

    struct AsyncRecordingExecutor;

    impl AsyncRuntimeExecutor for AsyncRecordingExecutor {
        fn name(&self) -> &'static str {
            "async-recording-runtime"
        }

        fn capabilities(&self) -> RuntimeCapabilities {
            RuntimeCapabilities {
                next_token_logits: true,
                sequence_scores: true,
                shared_prefix: true,
                parallel_suffixes: true,
                selective_output_head: true,
                local_inference: true,
            }
        }

        fn score_async<'a>(
            &'a self,
            batch: &'a RuntimeBatch,
            _plan: ExecutionPlan,
        ) -> RuntimeFuture<'a, Result<RuntimeOutput, RuntimeError>> {
            Box::pin(std::future::ready(Ok(RuntimeOutput {
                    model: Some("async-open-model-test".to_owned()),
                    scores: batch
                        .tasks
                        .iter()
                        .map(|task| RuntimeTaskScores {
                            id: task.id.clone(),
                            logits: (0..task.candidates.len())
                                .map(|index| index as f32)
                                .collect(),
                        })
                        .collect(),
                    metrics: BackendMetrics::default(),
                })))
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }

        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(output) => return output,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn task(id: &str, instruction: &str) -> DecisionTask {
        DecisionTask {
            id: id.into(),
            intent: DecisionIntent::Criterion,
            instruction: Some(instruction.into()),
            candidates: vec![
                Candidate {
                    key: "no".into(),
                    description: "no".into(),
                },
                Candidate {
                    key: "yes".into(),
                    description: "yes".into(),
                },
            ],
        }
    }

    #[test]
    fn bridge_prepares_codes_and_factors_exact_common_prefix() {
        let backend = RuntimeBackend::new(JsonRenderer, CharacterCodec, RecordingExecutor::new());
        let batch = DecisionBatch {
            state: serde_json::json!({"mensaje":"hola"}),
            tasks: vec![task("a", "¿Uno?"), task("b", "¿Dos?")],
        };
        let plan = ExecutionPlan {
            prefix_reuse: PrefixReuse::Shared,
            scheduling: TaskScheduling::Parallel,
            parallel_task_limit: 2,
            projection: turbojev_backend::CandidateProjection::CandidatesOnly,
        };
        let output = backend.score(&batch, plan).unwrap();
        assert_eq!(output.scores.len(), 2);
        let prepared = backend.executor().last_batch.lock().unwrap().clone().unwrap();
        assert!(!prepared.common_prefix.is_empty());
        assert!(prepared.tasks.iter().all(|task| task.codebook_kind == CodebookKind::SingleToken));
    }

    #[test]
    fn runtime_backend_is_model_family_agnostic() {
        let backend = RuntimeBackend::new(JsonRenderer, CharacterCodec, RecordingExecutor::new());
        assert_eq!(backend.name(), "recording-runtime");
        assert!(backend.capabilities().multi_token_candidates);
    }

    #[test]
    fn async_runtime_bridge_uses_the_same_preparation_contract() {
        let backend = RuntimeBackend::new(JsonRenderer, CharacterCodec, AsyncRecordingExecutor);
        let batch = DecisionBatch {
            state: serde_json::json!({"mensaje":"hola"}),
            tasks: vec![task("a", "¿Uno?"), task("b", "¿Dos?")],
        };
        let plan = ExecutionPlan {
            prefix_reuse: PrefixReuse::Shared,
            scheduling: TaskScheduling::Parallel,
            parallel_task_limit: 2,
            projection: turbojev_backend::CandidateProjection::CandidatesOnly,
        };
        let output = block_on(AsyncScoringBackend::score_async(&backend, &batch, plan)).unwrap();
        assert_eq!(output.model.as_deref(), Some("async-open-model-test"));
        assert_eq!(output.scores.len(), 2);
    }

}
