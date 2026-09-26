#![deny(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Optional llama.cpp/GGUF reference adapter.
//!
//! This crate is deliberately outside TurboJev's portable core. It implements
//! four correctness-preserving scheduling paths: independent reference
//! scoring, deterministic shared-prefix reuse, bounded parallel suffix batches,
//! and true isolated multi-root batching for eligible cross-request workloads. v0.9 adds a truthful candidate-only projection contract: a low-level
//! API may advertise it only when it bypasses the ordinary full-vocabulary head
//! and passes parity against the reference path. The stock FFI shim deliberately
//! keeps this capability disabled until llama.cpp exposes such a path cleanly.
//!
//! The high-level adapter is independent of llama.cpp's C ABI. [`LlamaCppApi`]
//! is the narrow stable boundary. With the `ffi` feature, [`ffi::FfiLlamaCpp`]
//! talks to the versioned `turbojev_llama_shim` shipped in `native/`.

use std::{sync::Arc, time::Instant};

use thiserror::Error;
use turbojev_backend::{
    BackendMetrics, BackendResourceHints, CandidateProjection, ExecutionPlan,
    FusedBackendCapabilities, PrefixReuse, TaskScheduling,
};
use turbojev_prompt::{
    PromptCriterion, PromptDocument, PromptError, PromptRenderer, RenderedMessages,
};
use turbojev_runtime::{
    FusedRuntimeExecutor, PromptCodec, RuntimeBackend, RuntimeCapabilities, RuntimeError,
    RuntimeExecutor, RuntimeOutput, RuntimeTaskScores, TokenId,
};
use turbojev_sequence::{CandidateTrie, SequenceWork};

#[cfg(feature = "ffi")]
pub mod ffi;

/// Caller-configured prompt profile for the reference GGUF adapter.
///
/// Human-language policy lives here, outside TurboJev core. Applications may
/// supply Spanish, Japanese, Arabic, English, or model-specific instructions
/// without changing any semantic or inference code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencePromptConfig {
    /// Stable profile identifier recorded by benchmark/conformance tooling.
    pub profile: String,
    /// System instruction telling the checkpoint to emit exactly one option code.
    pub system_instruction: String,
    /// Instruction used by the internal best-match prompt criterion.
    pub best_match_instruction: String,
}

impl ReferencePromptConfig {
    /// Build an explicit model/language prompt profile.
    pub fn new(
        profile: impl Into<String>,
        system_instruction: impl Into<String>,
        best_match_instruction: impl Into<String>,
    ) -> Self {
        Self {
            profile: profile.into(),
            system_instruction: system_instruction.into(),
            best_match_instruction: best_match_instruction.into(),
        }
    }
}

/// Deterministic JSON renderer used by the llama.cpp reference/reuse paths.
///
/// It preserves caller evidence and criterion text without translation. Only
/// the adapter-owned control instructions come from [`ReferencePromptConfig`].
#[derive(Debug, Clone)]
pub struct ReferencePromptRenderer {
    config: ReferencePromptConfig,
}

impl ReferencePromptRenderer {
    /// Create a renderer from an explicit prompt profile.
    pub fn new(config: ReferencePromptConfig) -> Self {
        Self { config }
    }

    /// Borrow the configured profile.
    pub fn config(&self) -> &ReferencePromptConfig {
        &self.config
    }
}

impl PromptRenderer for ReferencePromptRenderer {
    fn profile(&self) -> &str {
        &self.config.profile
    }

    fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError> {
        if self.config.profile.trim().is_empty()
            || self.config.system_instruction.trim().is_empty()
            || self.config.best_match_instruction.trim().is_empty()
        {
            return Err(PromptError::Renderer(
                "llama.cpp reference prompt profile contains empty control text".to_owned(),
            ));
        }

        let criterion = match &document.criterion {
            PromptCriterion::UserText(text) => text.clone(),
            PromptCriterion::BestMatch => self.config.best_match_instruction.clone(),
        };
        let options: Vec<serde_json::Value> = document
            .options
            .iter()
            .map(|option| {
                serde_json::json!({
                    "code": &option.code,
                    "description": &option.description,
                })
            })
            .collect();
        let payload = serde_json::json!({
            "evidence": &document.evidence,
            "criterion": criterion,
            "options": options,
        });
        let user = serde_json::to_string(&payload)
            .map_err(|error| PromptError::Renderer(error.to_string()))?;
        Ok(RenderedMessages {
            system: Some(self.config.system_instruction.clone()),
            user,
            profile: self.config.profile.clone(),
        })
    }
}

/// Stable independent-reference capabilities.
pub const REFERENCE_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    next_token_logits: true,
    sequence_scores: false,
    shared_prefix: false,
    parallel_suffixes: false,
    selective_output_head: false,
    local_inference: true,
};

/// Capabilities when the loaded llama.cpp context exposes reusable sequence
/// memory but only serial branch execution.
pub const SHARED_PREFIX_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    next_token_logits: true,
    sequence_scores: false,
    shared_prefix: true,
    parallel_suffixes: false,
    selective_output_head: false,
    local_inference: true,
};

/// Capabilities when reusable sequence memory and batched independent
/// suffix branches are both available.
pub const PARALLEL_SUFFIX_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    next_token_logits: true,
    sequence_scores: false,
    shared_prefix: true,
    parallel_suffixes: true,
    selective_output_head: false,
    local_inference: true,
};

/// Borrowed low-level branch request for llama.cpp batch scoring.
#[derive(Debug, Clone, Copy)]
pub struct LlamaCppSuffixTask<'a> {
    /// Prompt suffix evaluated from the currently prefetched shared prefix.
    pub suffix_tokens: &'a [TokenId],
    /// Single-token candidates whose next-token logits are gathered.
    pub candidate_token_ids: &'a [TokenId],
}

/// Borrowed independent-root request for true cross-request llama.cpp batching.
#[derive(Debug, Clone, Copy)]
pub struct LlamaCppRootTask<'a> {
    /// Complete prompt tokens for this independent root.
    pub prompt_tokens: &'a [TokenId],
    /// Single-token candidates whose next-token logits are gathered.
    pub candidate_token_ids: &'a [TokenId],
}

/// Low-level llama.cpp operations required by TurboJev.
///
/// Implementations own the loaded model and tokenizer. This boundary makes the
/// Rust adapter testable without linking native code and insulates TurboJev from
/// llama.cpp API churn.
pub trait LlamaCppApi {
    /// Human-readable loaded model/checkpoint identity.
    fn model_id(&self) -> &str;

    /// Stable tokenizer/template profile identifier for diagnostics.
    fn profile(&self) -> &str;

    /// Apply the checkpoint's actual chat template and tokenize up to the
    /// assistant answer boundary. `answer` is appended at that exact boundary
    /// when supplied.
    fn encode_chat(
        &self,
        system: Option<&str>,
        user: &str,
        answer: Option<&str>,
    ) -> Result<Vec<TokenId>, LlamaCppError>;

    /// Decode an exact token sequence without adding or removing content.
    fn decode_tokens(&self, tokens: &[TokenId]) -> Result<String, LlamaCppError>;

    /// Run an independent full prompt and gather the next-token logits for the
    /// supplied single-token candidate IDs in the same order.
    fn score_next_token(
        &self,
        prompt_tokens: &[TokenId],
        candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError>;

    /// Whether the runtime can compute exact conditional log-likelihoods for
    /// prefix-free candidate token sequences using full-vocabulary normalizers.
    fn sequence_scores_supported(&self) -> bool {
        false
    }

    /// Exact independent candidate-sequence scores in request order.
    fn score_sequences(
        &self,
        _prompt_tokens: &[TokenId],
        _candidate_sequences: &[Vec<TokenId>],
    ) -> Result<Vec<f32>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "exact multi-token sequence scoring is not implemented by this llama.cpp API"
                .to_owned(),
        ))
    }

    /// Whether this runtime can bypass the ordinary full-vocabulary LM head and
    /// project only the requested candidate rows. Implementations must advertise
    /// this only when the selective path is semantically equivalent to the
    /// ordinary full-vocabulary logits for the loaded checkpoint.
    fn selective_output_head_supported(&self) -> bool {
        false
    }

    /// Score one independent prompt using a true candidate-only output projection.
    fn score_next_token_selected(
        &self,
        _prompt_tokens: &[TokenId],
        _candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "candidate-only output projection is not implemented by this llama.cpp API".to_owned(),
        ))
    }

    /// Whether this loaded runtime can evaluate independent full prompt roots
    /// in one native batch while keeping sequence/KV state isolated.
    fn multi_root_supported(&self) -> bool {
        false
    }

    /// Hard maximum independent roots accepted by one native submission.
    fn max_multi_root_tasks(&self) -> Option<usize> {
        None
    }

    /// Hard maximum total prompt tokens accepted by one native multi-root call.
    fn max_multi_root_tokens(&self) -> Option<usize> {
        None
    }

    /// Conservative incremental memory estimate per independent request, when
    /// the concrete adapter can provide one without guessing.
    fn estimated_multi_root_item_bytes(&self) -> Option<u64> {
        None
    }

    /// Score multiple independent full prompts in one native model operation.
    /// Returned rows must preserve input root order.
    fn score_multi_root_next_token(
        &self,
        _tasks: &[LlamaCppRootTask<'_>],
    ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "multi-root scoring is not implemented by this llama.cpp API".to_owned(),
        ))
    }

    /// Whether this loaded runtime can preserve one immutable prefix sequence
    /// while evaluating ephemeral suffix branches.
    fn shared_prefix_supported(&self) -> bool {
        false
    }

    /// Clear prior sequence state and prefill one reusable token prefix.
    fn prefill_shared_prefix(&self, _prefix_tokens: &[TokenId]) -> Result<(), LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "shared-prefix prefill is not implemented by this llama.cpp API".to_owned(),
        ))
    }

    /// Score one suffix from the previously prefetched prefix. Implementations
    /// must leave the base prefix unchanged for subsequent calls.
    fn score_suffix_next_token(
        &self,
        _suffix_tokens: &[TokenId],
        _candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "shared-prefix suffix scoring is not implemented by this llama.cpp API".to_owned(),
        ))
    }

    /// Exact sequence-scoring counterpart of [`LlamaCppApi::score_suffix_next_token`].
    fn score_suffix_sequences(
        &self,
        _suffix_tokens: &[TokenId],
        _candidate_sequences: &[Vec<TokenId>],
    ) -> Result<Vec<f32>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "exact shared-prefix sequence scoring is not implemented by this llama.cpp API"
                .to_owned(),
        ))
    }

    /// Candidate-only counterpart of [`LlamaCppApi::score_suffix_next_token`].
    fn score_suffix_next_token_selected(
        &self,
        _suffix_tokens: &[TokenId],
        _candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "candidate-only shared-prefix scoring is not implemented by this llama.cpp API"
                .to_owned(),
        ))
    }

    /// Whether this runtime can evaluate multiple isolated suffix branches in
    /// one model call after a shared-prefix prefill.
    fn parallel_suffix_supported(&self) -> bool {
        false
    }

    /// Hard maximum number of suffix branches accepted in one parallel batch.
    fn max_parallel_tasks(&self) -> Option<usize> {
        None
    }

    /// Conservative incremental memory estimate per parallel branch, when the
    /// host/runtime can provide one.
    fn estimated_parallel_task_bytes(&self) -> Option<u64> {
        None
    }

    /// Maximum total suffix tokens accepted by one native parallel submission
    /// after the current shared prefix has been prefetched. `None` means the
    /// adapter cannot expose such a bound.
    fn max_parallel_suffix_tokens(&self) -> Option<usize> {
        None
    }

    /// Evaluate multiple isolated suffix branches from the current prefetched
    /// prefix. Returned rows must preserve input task order.
    fn score_suffix_batch_next_token(
        &self,
        _tasks: &[LlamaCppSuffixTask<'_>],
    ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "parallel shared-suffix scoring is not implemented by this llama.cpp API".to_owned(),
        ))
    }

    /// Candidate-only counterpart of [`LlamaCppApi::score_suffix_batch_next_token`].
    fn score_suffix_batch_next_token_selected(
        &self,
        _tasks: &[LlamaCppSuffixTask<'_>],
    ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
        Err(LlamaCppError::Unsupported(
            "candidate-only parallel shared-suffix scoring is not implemented by this llama.cpp API".to_owned(),
        ))
    }
}

/// Prompt codec backed by llama.cpp's tokenizer and model chat template.
#[derive(Clone)]
pub struct LlamaCppCodec<A> {
    api: Arc<A>,
}

impl<A> LlamaCppCodec<A> {
    /// Create a codec around a loaded llama.cpp API implementation.
    pub fn new(api: Arc<A>) -> Self {
        Self { api }
    }

    /// Borrow the low-level API.
    pub fn api(&self) -> &A {
        &self.api
    }
}

impl<A: LlamaCppApi> PromptCodec for LlamaCppCodec<A> {
    fn profile(&self) -> &str {
        self.api.profile()
    }

    fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError> {
        self.api
            .encode_chat(messages.system.as_deref(), &messages.user, None)
            .map_err(map_api_error)
    }

    fn encode_prompt_with_answer(
        &self,
        messages: &RenderedMessages,
        answer: &str,
    ) -> Result<Vec<TokenId>, RuntimeError> {
        self.api
            .encode_chat(messages.system.as_deref(), &messages.user, Some(answer))
            .map_err(map_api_error)
    }

    fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError> {
        self.api.decode_tokens(token_ids).map_err(map_api_error)
    }
}

/// Reference llama.cpp executor.
#[derive(Clone)]
pub struct LlamaCppExecutor<A> {
    api: Arc<A>,
}

impl<A> LlamaCppExecutor<A> {
    /// Create an executor around a loaded llama.cpp API implementation.
    pub fn new(api: Arc<A>) -> Self {
        Self { api }
    }
}

impl<A: LlamaCppApi> RuntimeExecutor for LlamaCppExecutor<A> {
    fn name(&self) -> &'static str {
        "llamacpp"
    }

    fn capabilities(&self) -> RuntimeCapabilities {
        let mut capabilities = if self.api.parallel_suffix_supported() {
            PARALLEL_SUFFIX_CAPABILITIES
        } else if self.api.shared_prefix_supported() {
            SHARED_PREFIX_CAPABILITIES
        } else {
            REFERENCE_CAPABILITIES
        };
        capabilities.selective_output_head = self.api.selective_output_head_supported();
        capabilities.sequence_scores = self.api.sequence_scores_supported();
        capabilities
    }

    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints {
            max_parallel_tasks: self.api.max_parallel_tasks(),
            estimated_parallel_task_bytes: self.api.estimated_parallel_task_bytes(),
        }
    }

    fn score(
        &self,
        batch: &turbojev_runtime::RuntimeBatch,
        plan: ExecutionPlan,
    ) -> Result<RuntimeOutput, RuntimeError> {
        validate_llamacpp_plan(
            plan,
            self.api.shared_prefix_supported(),
            self.api.parallel_suffix_supported(),
            self.api.selective_output_head_supported(),
        )?;
        match (plan.prefix_reuse, plan.scheduling) {
            (PrefixReuse::None, TaskScheduling::Serial) => {
                self.score_independent(batch, plan.projection)
            }
            (PrefixReuse::Shared, TaskScheduling::Serial) => {
                self.score_shared(batch, plan.projection)
            }
            (PrefixReuse::Shared, TaskScheduling::Parallel) => {
                self.score_shared_parallel(batch, plan.parallel_task_limit, plan.projection)
            }
            (PrefixReuse::None, TaskScheduling::Parallel) => Err(RuntimeError::Unsupported(
                "llama.cpp parallel suffix execution requires a shared prefix".to_owned(),
            )),
        }
    }
}

impl<A: LlamaCppApi> FusedRuntimeExecutor for LlamaCppExecutor<A> {
    fn fused_capabilities(&self) -> FusedBackendCapabilities {
        if !self.api.multi_root_supported() {
            return FusedBackendCapabilities::disabled();
        }
        FusedBackendCapabilities {
            fused_batch: true,
            max_fused_items: self.api.max_multi_root_tasks(),
            estimated_fused_item_bytes: self.api.estimated_multi_root_item_bytes(),
        }
    }

    fn can_score_fused(
        &self,
        batches: &[turbojev_runtime::RuntimeBatch],
        plans: &[ExecutionPlan],
    ) -> Result<bool, RuntimeError> {
        if !self.api.multi_root_supported() || batches.len() < 2 || batches.len() != plans.len() {
            return Ok(false);
        }
        let mut roots = 0usize;
        let mut prompt_tokens = 0usize;
        for (batch, plan) in batches.iter().zip(plans) {
            if plan.projection != CandidateProjection::FullVocabulary {
                return Ok(false);
            }
            for task in &batch.tasks {
                if !is_single_token_task(task) {
                    return Ok(false);
                }
                roots = roots.saturating_add(1);
                prompt_tokens = prompt_tokens
                    .saturating_add(batch.common_prefix.len())
                    .saturating_add(task.prompt_suffix.len());
            }
        }
        if roots < 2 {
            return Ok(false);
        }
        if let Some(maximum) = self.api.max_multi_root_tasks() {
            if roots > maximum {
                return Ok(false);
            }
        }
        if let Some(maximum) = self.api.max_multi_root_tokens() {
            if prompt_tokens > maximum {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn score_fused(
        &self,
        batches: &[turbojev_runtime::RuntimeBatch],
        plans: &[ExecutionPlan],
    ) -> Result<Vec<RuntimeOutput>, RuntimeError> {
        if !self.can_score_fused(batches, plans)? {
            return Err(RuntimeError::Unsupported(
                "llama.cpp fused execution requires single-token candidates, full-vocabulary projection, and a multi-root batch that fits native sequence/token limits".to_owned(),
            ));
        }

        let started = Instant::now();
        let mut prompts = Vec::<Vec<TokenId>>::new();
        let mut candidates = Vec::<Vec<TokenId>>::new();
        let mut request_root_ranges = Vec::<(usize, usize)>::with_capacity(batches.len());
        let mut request_input_tokens = Vec::<usize>::with_capacity(batches.len());

        for batch in batches {
            let begin = prompts.len();
            let mut input_tokens = 0usize;
            for task in &batch.tasks {
                let mut full_prompt = Vec::with_capacity(
                    batch
                        .common_prefix
                        .len()
                        .saturating_add(task.prompt_suffix.len()),
                );
                full_prompt.extend_from_slice(&batch.common_prefix);
                full_prompt.extend_from_slice(&task.prompt_suffix);
                input_tokens = input_tokens.saturating_add(full_prompt.len());
                prompts.push(full_prompt);
                candidates.push(single_token_candidate_ids(task)?);
            }
            request_root_ranges.push((begin, prompts.len()));
            request_input_tokens.push(input_tokens);
        }

        let roots: Vec<LlamaCppRootTask<'_>> = prompts
            .iter()
            .zip(&candidates)
            .map(|(prompt, candidate_ids)| LlamaCppRootTask {
                prompt_tokens: prompt,
                candidate_token_ids: candidate_ids,
            })
            .collect();
        let fused_logits = self
            .api
            .score_multi_root_next_token(&roots)
            .map_err(map_api_error)?;
        if fused_logits.len() != roots.len() {
            return Err(RuntimeError::Inference(format!(
                "llama.cpp returned {} multi-root rows for {} roots",
                fused_logits.len(),
                roots.len()
            )));
        }

        let elapsed = started.elapsed().as_secs_f64();
        let total_roots = roots.len();
        let mut outputs = Vec::with_capacity(batches.len());
        for (request_index, batch) in batches.iter().enumerate() {
            let (begin, end) = request_root_ranges[request_index];
            let rows = batch
                .tasks
                .iter()
                .zip(&candidates[begin..end])
                .zip(&fused_logits[begin..end])
                .map(|((task, candidate_ids), logits)| {
                    validate_logits(&task.id, candidate_ids, logits)?;
                    Ok(RuntimeTaskScores {
                        id: task.id.clone(),
                        logits: logits.clone(),
                    })
                })
                .collect::<Result<Vec<_>, RuntimeError>>()?;
            outputs.push(RuntimeOutput {
                model: Some(self.api.model_id().to_owned()),
                scores: rows,
                metrics: BackendMetrics {
                    scoring_seconds: Some(elapsed),
                    total_seconds: Some(elapsed),
                    input_tokens: Some(request_input_tokens[request_index]),
                    fused_request_count: Some(batches.len()),
                    fused_root_count: Some(total_roots),
                    effective_plan: Some(ExecutionPlan::default()),
                    ..BackendMetrics::default()
                },
            });
        }
        Ok(outputs)
    }
}

impl<A: LlamaCppApi> LlamaCppExecutor<A> {
    fn score_independent(
        &self,
        batch: &turbojev_runtime::RuntimeBatch,
        projection: CandidateProjection,
    ) -> Result<RuntimeOutput, RuntimeError> {
        if !batch.common_prefix.is_empty() {
            return Err(RuntimeError::Preparation(
                "independent llama.cpp plan received a factored common prefix".to_owned(),
            ));
        }
        let started = Instant::now();
        let mut input_tokens = 0usize;
        let mut sequence_candidate_tokens = 0usize;
        let mut sequence_prefix_evaluations = 0usize;
        let mut sequence_token_steps_saved = 0usize;
        let mut rows = Vec::with_capacity(batch.tasks.len());
        for task in &batch.tasks {
            input_tokens = input_tokens.saturating_add(task.prompt_suffix.len());
            let (logits, work) = if is_single_token_task(task) {
                let candidate_ids = single_token_candidate_ids(task)?;
                let logits = self.score_prompt(&task.prompt_suffix, &candidate_ids, projection)?;
                validate_logits(&task.id, &candidate_ids, &logits)?;
                (logits, None)
            } else {
                if projection != CandidateProjection::FullVocabulary {
                    return Err(RuntimeError::Unsupported(
                        "exact multi-token scoring requires full-vocabulary normalization"
                            .to_owned(),
                    ));
                }
                if !self.api.sequence_scores_supported() {
                    return Err(RuntimeError::Unsupported(
                        "loaded llama.cpp runtime does not expose exact sequence scoring"
                            .to_owned(),
                    ));
                }
                let sequences = candidate_sequences(task);
                let work = CandidateTrie::build(&sequences)
                    .map_err(|error| RuntimeError::Preparation(error.to_string()))?
                    .work();
                let logits = self
                    .api
                    .score_sequences(&task.prompt_suffix, &sequences)
                    .map_err(map_api_error)?;
                validate_sequence_scores(&task.id, task.candidates.len(), &logits)?;
                (logits, Some(work))
            };
            if let Some(work) = work {
                accumulate_sequence_work(
                    work,
                    &mut sequence_candidate_tokens,
                    &mut sequence_prefix_evaluations,
                    &mut sequence_token_steps_saved,
                );
            }
            rows.push(RuntimeTaskScores {
                id: task.id.clone(),
                logits,
            });
        }
        let total = started.elapsed().as_secs_f64();
        Ok(RuntimeOutput {
            model: Some(self.api.model_id().to_owned()),
            scores: rows,
            metrics: BackendMetrics {
                prefill_seconds: None,
                scoring_seconds: Some(total),
                total_seconds: Some(total),
                input_tokens: Some(input_tokens),
                candidate_rows_projected: (projection == CandidateProjection::CandidatesOnly)
                    .then(|| batch.tasks.iter().map(|task| task.candidates.len()).sum()),
                sequence_candidate_tokens: (sequence_candidate_tokens > 0)
                    .then_some(sequence_candidate_tokens),
                sequence_prefix_evaluations: (sequence_prefix_evaluations > 0)
                    .then_some(sequence_prefix_evaluations),
                sequence_token_steps_saved: (sequence_candidate_tokens > 0)
                    .then_some(sequence_token_steps_saved),
                ..BackendMetrics::default()
            },
        })
    }

    fn score_shared(
        &self,
        batch: &turbojev_runtime::RuntimeBatch,
        projection: CandidateProjection,
    ) -> Result<RuntimeOutput, RuntimeError> {
        if batch.common_prefix.is_empty() {
            return self.score_independent(batch, projection);
        }

        let total_started = Instant::now();
        let prefill_started = Instant::now();
        self.api
            .prefill_shared_prefix(&batch.common_prefix)
            .map_err(map_api_error)?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();

        let scoring_started = Instant::now();
        let mut rows = Vec::with_capacity(batch.tasks.len());
        let mut input_tokens = batch.common_prefix.len();
        let mut sequence_candidate_tokens = 0usize;
        let mut sequence_prefix_evaluations = 0usize;
        let mut sequence_token_steps_saved = 0usize;
        for task in &batch.tasks {
            input_tokens = input_tokens.saturating_add(task.prompt_suffix.len());
            let (logits, work) = if is_single_token_task(task) {
                let candidate_ids = single_token_candidate_ids(task)?;
                let logits = self.score_suffix(&task.prompt_suffix, &candidate_ids, projection)?;
                validate_logits(&task.id, &candidate_ids, &logits)?;
                (logits, None)
            } else {
                if projection != CandidateProjection::FullVocabulary {
                    return Err(RuntimeError::Unsupported(
                        "exact multi-token scoring requires full-vocabulary normalization"
                            .to_owned(),
                    ));
                }
                if !self.api.sequence_scores_supported() {
                    return Err(RuntimeError::Unsupported(
                        "loaded llama.cpp runtime does not expose exact sequence scoring"
                            .to_owned(),
                    ));
                }
                let sequences = candidate_sequences(task);
                let work = CandidateTrie::build(&sequences)
                    .map_err(|error| RuntimeError::Preparation(error.to_string()))?
                    .work();
                let logits = self
                    .api
                    .score_suffix_sequences(&task.prompt_suffix, &sequences)
                    .map_err(map_api_error)?;
                validate_sequence_scores(&task.id, task.candidates.len(), &logits)?;
                (logits, Some(work))
            };
            if let Some(work) = work {
                accumulate_sequence_work(
                    work,
                    &mut sequence_candidate_tokens,
                    &mut sequence_prefix_evaluations,
                    &mut sequence_token_steps_saved,
                );
            }
            rows.push(RuntimeTaskScores {
                id: task.id.clone(),
                logits,
            });
        }
        let scoring_seconds = scoring_started.elapsed().as_secs_f64();
        Ok(RuntimeOutput {
            model: Some(self.api.model_id().to_owned()),
            scores: rows,
            metrics: BackendMetrics {
                prefill_seconds: Some(prefill_seconds),
                scoring_seconds: Some(scoring_seconds),
                total_seconds: Some(total_started.elapsed().as_secs_f64()),
                input_tokens: Some(input_tokens),
                candidate_rows_projected: (projection == CandidateProjection::CandidatesOnly)
                    .then(|| batch.tasks.iter().map(|task| task.candidates.len()).sum()),
                sequence_candidate_tokens: (sequence_candidate_tokens > 0)
                    .then_some(sequence_candidate_tokens),
                sequence_prefix_evaluations: (sequence_prefix_evaluations > 0)
                    .then_some(sequence_prefix_evaluations),
                sequence_token_steps_saved: (sequence_candidate_tokens > 0)
                    .then_some(sequence_token_steps_saved),
                ..BackendMetrics::default()
            },
        })
    }

    fn score_shared_parallel(
        &self,
        batch: &turbojev_runtime::RuntimeBatch,
        parallel_task_limit: usize,
        projection: CandidateProjection,
    ) -> Result<RuntimeOutput, RuntimeError> {
        if parallel_task_limit < 2 {
            return Err(RuntimeError::Preparation(
                "parallel llama.cpp execution requires parallel_task_limit >= 2".to_owned(),
            ));
        }
        if batch.common_prefix.is_empty() {
            return self.score_independent(batch, projection);
        }

        let total_started = Instant::now();
        let prefill_started = Instant::now();
        self.api
            .prefill_shared_prefix(&batch.common_prefix)
            .map_err(map_api_error)?;
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();

        let scoring_started = Instant::now();
        let mut rows = Vec::with_capacity(batch.tasks.len());
        let mut input_tokens = batch.common_prefix.len();
        let mut parallel_batches = 0usize;
        let mut peak_parallel_tasks = 0usize;

        let suffix_token_limit = self.api.max_parallel_suffix_tokens();
        let mut start = 0usize;
        while start < batch.tasks.len() {
            let end =
                bounded_parallel_end(&batch.tasks, start, parallel_task_limit, suffix_token_limit);
            let chunk = &batch.tasks[start..end];
            let candidate_ids: Vec<Vec<TokenId>> = chunk
                .iter()
                .map(single_token_candidate_ids)
                .collect::<Result<_, _>>()?;
            for task in chunk {
                input_tokens = input_tokens.saturating_add(task.prompt_suffix.len());
            }

            if chunk.len() == 1 {
                let logits =
                    self.score_suffix(&chunk[0].prompt_suffix, &candidate_ids[0], projection)?;
                validate_logits(&chunk[0].id, &candidate_ids[0], &logits)?;
                rows.push(RuntimeTaskScores {
                    id: chunk[0].id.clone(),
                    logits,
                });
                start = end;
                continue;
            }

            let requests: Vec<LlamaCppSuffixTask<'_>> = chunk
                .iter()
                .zip(&candidate_ids)
                .map(|(task, candidates)| LlamaCppSuffixTask {
                    suffix_tokens: &task.prompt_suffix,
                    candidate_token_ids: candidates,
                })
                .collect();
            let batch_logits = self.score_suffix_batch(&requests, projection)?;
            if batch_logits.len() != chunk.len() {
                return Err(RuntimeError::Inference(format!(
                    "llama.cpp returned {} parallel rows for {} tasks",
                    batch_logits.len(),
                    chunk.len()
                )));
            }
            parallel_batches = parallel_batches.saturating_add(1);
            peak_parallel_tasks = peak_parallel_tasks.max(chunk.len());
            for ((task, candidates), logits) in chunk.iter().zip(&candidate_ids).zip(batch_logits) {
                validate_logits(&task.id, candidates, &logits)?;
                rows.push(RuntimeTaskScores {
                    id: task.id.clone(),
                    logits,
                });
            }
            start = end;
        }

        let scoring_seconds = scoring_started.elapsed().as_secs_f64();
        Ok(RuntimeOutput {
            model: Some(self.api.model_id().to_owned()),
            scores: rows,
            metrics: BackendMetrics {
                prefill_seconds: Some(prefill_seconds),
                scoring_seconds: Some(scoring_seconds),
                total_seconds: Some(total_started.elapsed().as_secs_f64()),
                input_tokens: Some(input_tokens),
                parallel_batches: Some(parallel_batches),
                peak_parallel_tasks: Some(peak_parallel_tasks),
                candidate_rows_projected: (projection == CandidateProjection::CandidatesOnly)
                    .then(|| batch.tasks.iter().map(|task| task.candidates.len()).sum()),
                ..BackendMetrics::default()
            },
        })
    }

    fn score_prompt(
        &self,
        prompt_tokens: &[TokenId],
        candidate_ids: &[TokenId],
        projection: CandidateProjection,
    ) -> Result<Vec<f32>, RuntimeError> {
        let result = match projection {
            CandidateProjection::FullVocabulary => {
                self.api.score_next_token(prompt_tokens, candidate_ids)
            }
            CandidateProjection::CandidatesOnly => self
                .api
                .score_next_token_selected(prompt_tokens, candidate_ids),
        };
        result.map_err(map_api_error)
    }

    fn score_suffix(
        &self,
        suffix_tokens: &[TokenId],
        candidate_ids: &[TokenId],
        projection: CandidateProjection,
    ) -> Result<Vec<f32>, RuntimeError> {
        let result = match projection {
            CandidateProjection::FullVocabulary => self
                .api
                .score_suffix_next_token(suffix_tokens, candidate_ids),
            CandidateProjection::CandidatesOnly => self
                .api
                .score_suffix_next_token_selected(suffix_tokens, candidate_ids),
        };
        result.map_err(map_api_error)
    }

    fn score_suffix_batch(
        &self,
        tasks: &[LlamaCppSuffixTask<'_>],
        projection: CandidateProjection,
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        let result = match projection {
            CandidateProjection::FullVocabulary => self.api.score_suffix_batch_next_token(tasks),
            CandidateProjection::CandidatesOnly => {
                self.api.score_suffix_batch_next_token_selected(tasks)
            }
        };
        result.map_err(map_api_error)
    }
}

fn bounded_parallel_end(
    tasks: &[turbojev_runtime::RuntimeTask],
    start: usize,
    task_limit: usize,
    suffix_token_limit: Option<usize>,
) -> usize {
    debug_assert!(start < tasks.len());
    debug_assert!(task_limit >= 2);

    let hard_end = tasks.len().min(start.saturating_add(task_limit));
    let Some(token_limit) = suffix_token_limit.filter(|limit| *limit > 0) else {
        return hard_end;
    };

    let mut end = start;
    let mut tokens = 0usize;
    while end < hard_end {
        let next = tasks[end].prompt_suffix.len();
        if end > start && tokens.saturating_add(next) > token_limit {
            break;
        }
        tokens = tokens.saturating_add(next);
        end += 1;
        if tokens >= token_limit {
            break;
        }
    }
    end.max(start + 1)
}

fn is_single_token_task(task: &turbojev_runtime::RuntimeTask) -> bool {
    task.candidates
        .iter()
        .all(|candidate| candidate.token_ids.len() == 1)
}

fn candidate_sequences(task: &turbojev_runtime::RuntimeTask) -> Vec<Vec<TokenId>> {
    task.candidates
        .iter()
        .map(|candidate| candidate.token_ids.clone())
        .collect()
}

fn accumulate_sequence_work(
    work: SequenceWork,
    candidate_tokens: &mut usize,
    prefix_evaluations: &mut usize,
    saved_steps: &mut usize,
) {
    *candidate_tokens = candidate_tokens.saturating_add(work.naive_token_steps);
    *prefix_evaluations = prefix_evaluations.saturating_add(work.unique_prefix_evaluations);
    *saved_steps = saved_steps.saturating_add(work.shared_token_steps_saved);
}

fn validate_sequence_scores(
    task_id: &str,
    expected: usize,
    scores: &[f32],
) -> Result<(), RuntimeError> {
    if scores.len() != expected {
        return Err(RuntimeError::Inference(format!(
            "llama.cpp returned {} sequence scores for {} candidates in task {:?}",
            scores.len(),
            expected,
            task_id
        )));
    }
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Inference(format!(
            "llama.cpp returned non-finite sequence scores for task {:?}",
            task_id
        )));
    }
    Ok(())
}

fn single_token_candidate_ids(
    task: &turbojev_runtime::RuntimeTask,
) -> Result<Vec<TokenId>, RuntimeError> {
    if task
        .candidates
        .iter()
        .any(|candidate| candidate.token_ids.len() != 1)
    {
        return Err(RuntimeError::Unsupported(format!(
            "task {:?} requires multi-token candidate scoring; llama.cpp next-token path is single-token only",
            task.id
        )));
    }
    Ok(task
        .candidates
        .iter()
        .map(|candidate| candidate.token_ids[0])
        .collect())
}

fn validate_logits(
    task_id: &str,
    candidate_ids: &[TokenId],
    logits: &[f32],
) -> Result<(), RuntimeError> {
    if logits.len() != candidate_ids.len() {
        return Err(RuntimeError::Inference(format!(
            "llama.cpp returned {} logits for {} candidates in task {:?}",
            logits.len(),
            candidate_ids.len(),
            task_id
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Inference(format!(
            "llama.cpp returned non-finite candidate logits for task {:?}",
            task_id
        )));
    }
    Ok(())
}

/// Compose a model-specific renderer with the llama.cpp codec and executor.
///
/// The returned backend advertises shared-prefix reuse only when the loaded
/// [`LlamaCppApi`] reports that capability. Planning policy, not this factory,
/// decides whether a call uses the independent reference path or cached reuse.
pub fn llamacpp_backend<R, A>(
    renderer: R,
    api: A,
) -> RuntimeBackend<R, LlamaCppCodec<A>, LlamaCppExecutor<A>>
where
    R: PromptRenderer,
    A: LlamaCppApi,
{
    let api = Arc::new(api);
    RuntimeBackend::new(
        renderer,
        LlamaCppCodec::new(Arc::clone(&api)),
        LlamaCppExecutor::new(api),
    )
    // Single-token codes remain the fast path. Exact prefix-free sequence
    // scoring is used only when the loaded runtime advertises it.
    .with_max_tokens_per_code(4)
}

/// Backwards-compatible alias retained for the v0.6 API.
///
/// Use `PlanningPolicy::Reference` at the TurboJev engine layer when an application needs
/// to force independent scoring; the backend itself now supports both paths.
pub fn reference_backend<R, A>(
    renderer: R,
    api: A,
) -> RuntimeBackend<R, LlamaCppCodec<A>, LlamaCppExecutor<A>>
where
    R: PromptRenderer,
    A: LlamaCppApi,
{
    llamacpp_backend(renderer, api)
}

fn validate_llamacpp_plan(
    plan: ExecutionPlan,
    shared_prefix_supported: bool,
    parallel_suffix_supported: bool,
    selective_output_head_supported: bool,
) -> Result<(), RuntimeError> {
    if plan.projection == CandidateProjection::CandidatesOnly && !selective_output_head_supported {
        return Err(RuntimeError::Unsupported(
            "loaded llama.cpp runtime does not expose a parity-verified candidate-only projection"
                .to_owned(),
        ));
    }
    if !plan.scheduling_is_valid() {
        return Err(RuntimeError::Preparation(
            "invalid scheduling/parallel_task_limit combination".to_owned(),
        ));
    }
    if plan.prefix_reuse == PrefixReuse::Shared && !shared_prefix_supported {
        return Err(RuntimeError::Unsupported(
            "loaded llama.cpp runtime does not support shared-prefix reuse".to_owned(),
        ));
    }
    if plan.scheduling == TaskScheduling::Parallel {
        if plan.prefix_reuse != PrefixReuse::Shared {
            return Err(RuntimeError::Unsupported(
                "llama.cpp parallel suffix execution requires shared-prefix reuse".to_owned(),
            ));
        }
        if !parallel_suffix_supported {
            return Err(RuntimeError::Unsupported(
                "loaded llama.cpp runtime does not support parallel suffix batches".to_owned(),
            ));
        }
    }
    Ok(())
}

fn map_api_error(error: LlamaCppError) -> RuntimeError {
    match error {
        LlamaCppError::Unsupported(message) => RuntimeError::Unsupported(message),
        LlamaCppError::Tokenization(message) => RuntimeError::Codec(message),
        LlamaCppError::Inference(message) | LlamaCppError::Native(message) => {
            RuntimeError::Inference(message)
        }
        LlamaCppError::InvalidInput(message) => RuntimeError::Preparation(message),
    }
}

/// Stable error category crossing the llama.cpp adapter boundary.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LlamaCppError {
    /// Input/configuration was invalid before native inference.
    #[error("invalid llama.cpp adapter input: {0}")]
    InvalidInput(String),
    /// Tokenizer or chat-template operation failed.
    #[error("llama.cpp tokenization failed: {0}")]
    Tokenization(String),
    /// Model execution failed.
    #[error("llama.cpp inference failed: {0}")]
    Inference(String),
    /// Runtime/model does not implement a required operation.
    #[error("llama.cpp operation unsupported: {0}")]
    Unsupported(String),
    /// Native shim failure.
    #[error("llama.cpp native bridge failed: {0}")]
    Native(String),
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use turbojev_backend::{Candidate, DecisionBatch, DecisionIntent, DecisionTask};
    use turbojev_prompt::{PromptCriterion, PromptDocument, PromptError};
    use turbojev_runtime_conformance::{validate_runtime_output, verify_candidate_boundary};

    use super::*;

    #[derive(Clone)]
    struct FamilyApi {
        model: String,
        profile: String,
        prefix: Vec<TokenId>,
        answer_marker: Vec<TokenId>,
        symbols: HashMap<String, Vec<TokenId>>,
        shared_prefix: Arc<Mutex<Vec<TokenId>>>,
        parallel_batches: Arc<Mutex<Vec<usize>>>,
        parallel_suffix_token_limit: Option<usize>,
        full_projection_calls: Arc<Mutex<usize>>,
        selective_projection_calls: Arc<Mutex<usize>>,
    }

    impl FamilyApi {
        fn qwen_style() -> Self {
            Self {
                model: "fixture-qwen-family".into(),
                profile: "fixture-qwen-chat".into(),
                prefix: vec![151_644, 8948],
                answer_marker: vec![151_645, 77091],
                symbols: [
                    ("A".into(), vec![32]),
                    ("B".into(), vec![33]),
                    ("C".into(), vec![34]),
                ]
                .into_iter()
                .collect(),
                shared_prefix: Arc::new(Mutex::new(Vec::new())),
                parallel_batches: Arc::new(Mutex::new(Vec::new())),
                parallel_suffix_token_limit: None,
                full_projection_calls: Arc::new(Mutex::new(0)),
                selective_projection_calls: Arc::new(Mutex::new(0)),
            }
        }

        fn llama_style() -> Self {
            Self {
                model: "fixture-llama-family".into(),
                profile: "fixture-llama-chat".into(),
                prefix: vec![128_000, 128_006],
                answer_marker: vec![128_009, 128_006],
                symbols: [
                    ("A".into(), vec![32]),
                    ("B".into(), vec![33]),
                    ("C".into(), vec![34]),
                ]
                .into_iter()
                .collect(),
                shared_prefix: Arc::new(Mutex::new(Vec::new())),
                parallel_batches: Arc::new(Mutex::new(Vec::new())),
                parallel_suffix_token_limit: None,
                full_projection_calls: Arc::new(Mutex::new(0)),
                selective_projection_calls: Arc::new(Mutex::new(0)),
            }
        }

        fn multi_token_style() -> Self {
            Self {
                model: "fixture-multi-token-family".into(),
                profile: "fixture-multi-token-chat".into(),
                prefix: vec![9_001, 9_002],
                answer_marker: vec![9_003],
                symbols: [
                    ("A".into(), vec![32, 32]),
                    ("B".into(), vec![32, 33]),
                    ("C".into(), vec![33, 32]),
                ]
                .into_iter()
                .collect(),
                shared_prefix: Arc::new(Mutex::new(Vec::new())),
                parallel_batches: Arc::new(Mutex::new(Vec::new())),
                parallel_suffix_token_limit: None,
                full_projection_calls: Arc::new(Mutex::new(0)),
                selective_projection_calls: Arc::new(Mutex::new(0)),
            }
        }

        fn with_parallel_suffix_token_limit(mut self, limit: usize) -> Self {
            self.parallel_suffix_token_limit = Some(limit);
            self
        }

        fn fixture_full_logits(
            &self,
            prompt_tokens: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            if prompt_tokens.is_empty() {
                return Err(LlamaCppError::InvalidInput("empty prompt".into()));
            }
            let checksum = prompt_tokens
                .iter()
                .fold(0u64, |sum, token| sum.wrapping_add(u64::from(*token)));
            let bias = (checksum % 101) as f32 / 101.0;
            let mut logits = vec![-9.0 + 0.1 * bias; 256];
            logits[32] = 0.5 + bias;
            logits[33] = 3.0 + 1.75 * bias;
            logits[34] = -1.0 - 0.5 * bias;
            Ok(logits)
        }

        fn fixture_logits(
            &self,
            prompt_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            let full = self.fixture_full_logits(prompt_tokens)?;
            candidate_token_ids
                .iter()
                .map(|token| {
                    full.get(*token as usize).copied().ok_or_else(|| {
                        LlamaCppError::InvalidInput("fixture candidate token out of range".into())
                    })
                })
                .collect()
        }

        fn fixture_batch_logits(
            &self,
            tasks: &[LlamaCppSuffixTask<'_>],
            selective: bool,
        ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
            if tasks.len() < 2 || tasks.len() > 6 {
                return Err(LlamaCppError::InvalidInput(
                    "fixture parallel batch must contain 2..=6 tasks".into(),
                ));
            }
            self.parallel_batches.lock().unwrap().push(tasks.len());
            let prefix = self.shared_prefix.lock().unwrap().clone();
            if prefix.is_empty() {
                return Err(LlamaCppError::InvalidInput(
                    "shared prefix was not prefetched".into(),
                ));
            }
            if selective {
                *self.selective_projection_calls.lock().unwrap() += tasks.len();
            } else {
                *self.full_projection_calls.lock().unwrap() += tasks.len();
            }
            tasks
                .iter()
                .map(|task| {
                    let mut full = prefix.clone();
                    full.extend_from_slice(task.suffix_tokens);
                    self.fixture_logits(&full, task.candidate_token_ids)
                })
                .collect()
        }

        fn encode_text(&self, text: &str) -> Vec<TokenId> {
            text.bytes().map(|byte| 1_000 + u32::from(byte)).collect()
        }
    }

    impl LlamaCppApi for FamilyApi {
        fn model_id(&self) -> &str {
            &self.model
        }

        fn profile(&self) -> &str {
            &self.profile
        }

        fn encode_chat(
            &self,
            system: Option<&str>,
            user: &str,
            answer: Option<&str>,
        ) -> Result<Vec<TokenId>, LlamaCppError> {
            let mut result = self.prefix.clone();
            if let Some(system) = system {
                result.extend(self.encode_text(system));
            }
            result.extend(self.encode_text(user));
            result.extend(&self.answer_marker);
            if let Some(answer) = answer {
                let tokens = self.symbols.get(answer).ok_or_else(|| {
                    LlamaCppError::Tokenization(format!("unknown fixture answer {answer:?}"))
                })?;
                result.extend_from_slice(tokens);
            }
            Ok(result)
        }

        fn decode_tokens(&self, tokens: &[TokenId]) -> Result<String, LlamaCppError> {
            self.symbols
                .iter()
                .find_map(|(symbol, candidate)| {
                    (candidate.as_slice() == tokens).then(|| symbol.clone())
                })
                .ok_or_else(|| LlamaCppError::Tokenization("unknown fixture token sequence".into()))
        }

        fn score_next_token(
            &self,
            prompt_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            *self.full_projection_calls.lock().unwrap() += 1;
            self.fixture_logits(prompt_tokens, candidate_token_ids)
        }

        fn sequence_scores_supported(&self) -> bool {
            true
        }

        fn score_sequences(
            &self,
            prompt_tokens: &[TokenId],
            candidate_sequences: &[Vec<TokenId>],
        ) -> Result<Vec<f32>, LlamaCppError> {
            let trie = CandidateTrie::build(candidate_sequences)
                .map_err(|error| LlamaCppError::InvalidInput(error.to_string()))?;
            trie.score_with(|prefix| {
                let mut full = prompt_tokens.to_vec();
                full.extend_from_slice(prefix);
                self.fixture_full_logits(&full)
            })
            .map_err(|error| LlamaCppError::Inference(error.to_string()))
        }

        fn selective_output_head_supported(&self) -> bool {
            true
        }

        fn score_next_token_selected(
            &self,
            prompt_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            *self.selective_projection_calls.lock().unwrap() += 1;
            self.fixture_logits(prompt_tokens, candidate_token_ids)
        }

        fn multi_root_supported(&self) -> bool {
            true
        }

        fn max_multi_root_tasks(&self) -> Option<usize> {
            Some(8)
        }

        fn max_multi_root_tokens(&self) -> Option<usize> {
            Some(16_384)
        }

        fn score_multi_root_next_token(
            &self,
            tasks: &[LlamaCppRootTask<'_>],
        ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
            if tasks.len() < 2 || tasks.len() > 8 {
                return Err(LlamaCppError::InvalidInput(
                    "fixture multi-root batch must contain 2..=8 roots".into(),
                ));
            }
            tasks
                .iter()
                .map(|task| self.fixture_logits(task.prompt_tokens, task.candidate_token_ids))
                .collect()
        }

        fn shared_prefix_supported(&self) -> bool {
            true
        }

        fn prefill_shared_prefix(&self, prefix_tokens: &[TokenId]) -> Result<(), LlamaCppError> {
            if prefix_tokens.is_empty() {
                return Err(LlamaCppError::InvalidInput("empty shared prefix".into()));
            }
            *self.shared_prefix.lock().unwrap() = prefix_tokens.to_vec();
            Ok(())
        }

        fn score_suffix_next_token(
            &self,
            suffix_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            let prefix = self.shared_prefix.lock().unwrap();
            if prefix.is_empty() {
                return Err(LlamaCppError::InvalidInput(
                    "shared prefix was not prefetched".into(),
                ));
            }
            let mut full = prefix.clone();
            full.extend_from_slice(suffix_tokens);
            drop(prefix);
            *self.full_projection_calls.lock().unwrap() += 1;
            self.fixture_logits(&full, candidate_token_ids)
        }

        fn score_suffix_sequences(
            &self,
            suffix_tokens: &[TokenId],
            candidate_sequences: &[Vec<TokenId>],
        ) -> Result<Vec<f32>, LlamaCppError> {
            let prefix = self.shared_prefix.lock().unwrap();
            if prefix.is_empty() {
                return Err(LlamaCppError::InvalidInput(
                    "shared prefix was not prefetched".into(),
                ));
            }
            let mut full = prefix.clone();
            full.extend_from_slice(suffix_tokens);
            drop(prefix);
            self.score_sequences(&full, candidate_sequences)
        }

        fn score_suffix_next_token_selected(
            &self,
            suffix_tokens: &[TokenId],
            candidate_token_ids: &[TokenId],
        ) -> Result<Vec<f32>, LlamaCppError> {
            let prefix = self.shared_prefix.lock().unwrap();
            if prefix.is_empty() {
                return Err(LlamaCppError::InvalidInput(
                    "shared prefix was not prefetched".into(),
                ));
            }
            let mut full = prefix.clone();
            full.extend_from_slice(suffix_tokens);
            drop(prefix);
            *self.selective_projection_calls.lock().unwrap() += 1;
            self.fixture_logits(&full, candidate_token_ids)
        }

        fn parallel_suffix_supported(&self) -> bool {
            true
        }

        fn max_parallel_tasks(&self) -> Option<usize> {
            Some(6)
        }

        fn estimated_parallel_task_bytes(&self) -> Option<u64> {
            Some(4096)
        }

        fn max_parallel_suffix_tokens(&self) -> Option<usize> {
            self.parallel_suffix_token_limit
        }

        fn score_suffix_batch_next_token(
            &self,
            tasks: &[LlamaCppSuffixTask<'_>],
        ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
            self.fixture_batch_logits(tasks, false)
        }

        fn score_suffix_batch_next_token_selected(
            &self,
            tasks: &[LlamaCppSuffixTask<'_>],
        ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
            self.fixture_batch_logits(tasks, true)
        }
    }

    struct Renderer;

    impl PromptRenderer for Renderer {
        fn profile(&self) -> &str {
            "fixture-renderer"
        }

        fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError> {
            let criterion = match &document.criterion {
                PromptCriterion::UserText(text) => text.clone(),
                PromptCriterion::BestMatch => "best-match".to_owned(),
            };
            let options = document
                .options
                .iter()
                .map(|option| format!("{}={}", option.code, option.description))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(RenderedMessages {
                system: Some("fixture".into()),
                user: format!("{}\n{}\n{}", document.evidence, criterion, options),
                profile: self.profile().to_owned(),
            })
        }
    }

    fn batch() -> DecisionBatch {
        DecisionBatch {
            state: serde_json::json!({"message":"package missing"}),
            tasks: vec![DecisionTask {
                id: "route".into(),
                intent: DecisionIntent::Criterion,
                instruction: Some("route request".into()),
                candidates: vec![
                    Candidate {
                        key: "billing".into(),
                        description: "billing".into(),
                    },
                    Candidate {
                        key: "shipping".into(),
                        description: "shipping".into(),
                    },
                    Candidate {
                        key: "other".into(),
                        description: "other".into(),
                    },
                ],
            }],
        }
    }

    fn reference_plan() -> ExecutionPlan {
        ExecutionPlan {
            prefix_reuse: PrefixReuse::None,
            scheduling: TaskScheduling::Serial,
            parallel_task_limit: 1,
            projection: CandidateProjection::FullVocabulary,
        }
    }

    fn check_family(api: FamilyApi) {
        let api = Arc::new(api);
        let codec = LlamaCppCodec::new(Arc::clone(&api));
        let messages = RenderedMessages {
            system: Some("fixture".into()),
            user: "test".into(),
            profile: "fixture".into(),
        };
        let symbols = vec!["A".to_owned(), "B".to_owned(), "C".to_owned()];
        let verified = verify_candidate_boundary(&codec, &messages, &symbols, 1).unwrap();
        assert_eq!(verified.len(), 3);

        let backend = reference_backend(Renderer, (*api).clone());
        let prepared = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![],
            tasks: vec![turbojev_runtime::RuntimeTask {
                id: "route".into(),
                prompt_suffix: api.encode_chat(Some("fixture"), "test", None).unwrap(),
                candidates: verified
                    .into_iter()
                    .map(|item| turbojev_runtime::RuntimeCandidate {
                        symbol: item.symbol,
                        token_ids: item.token_ids,
                    })
                    .collect(),
                codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
            }],
        };
        let output = backend
            .executor()
            .score(&prepared, reference_plan())
            .unwrap();
        validate_runtime_output(&prepared, &output).unwrap();
        assert_eq!(output.scores[0].logits.len(), 3);
        assert!(output.scores[0].logits[1] > output.scores[0].logits[0]);

        // Also exercise the semantic backend composition end-to-end.
        let semantic =
            turbojev_backend::ScoringBackend::score(&backend, &batch(), reference_plan()).unwrap();
        assert_eq!(semantic.scores[0].logits.len(), 3);
        assert!(semantic.scores[0].logits[1] > semantic.scores[0].logits[0]);
    }

    #[test]
    fn configured_renderer_preserves_caller_language() {
        let renderer = ReferencePromptRenderer::new(ReferencePromptConfig::new(
            "es-reference-v1",
            "Selecciona exactamente un código y responde únicamente con ese código.",
            "¿Qué opción corresponde mejor con la evidencia?",
        ));
        let document = PromptDocument {
            evidence: serde_json::json!({"mensaje":"Mi paquete no llegó"}),
            criterion: PromptCriterion::UserText("¿Qué equipo debe atender este caso?".to_owned()),
            options: vec![
                turbojev_prompt::PromptOption {
                    code: "A".to_owned(),
                    key: "facturacion".to_owned(),
                    description: "Facturación".to_owned(),
                },
                turbojev_prompt::PromptOption {
                    code: "B".to_owned(),
                    key: "envios".to_owned(),
                    description: "Envíos".to_owned(),
                },
            ],
            contract: turbojev_prompt::PROMPT_IR_CONTRACT.to_owned(),
        };
        let rendered = renderer.render(&document).unwrap();
        assert_eq!(rendered.profile, "es-reference-v1");
        assert!(rendered.user.contains("Mi paquete no llegó"));
        assert!(rendered
            .user
            .contains("¿Qué equipo debe atender este caso?"));
        assert!(rendered.user.contains("Envíos"));
    }

    #[test]
    fn qwen_family_contract_passes() {
        check_family(FamilyApi::qwen_style());
    }

    #[test]
    fn llama_family_contract_passes() {
        check_family(FamilyApi::llama_style());
    }

    #[test]
    fn shared_prefix_matches_independent_reference_exactly() {
        use turbojev_runtime_conformance::{compare_runtime_outputs, RuntimeTolerance};

        let executor = LlamaCppExecutor::new(Arc::new(FamilyApi::qwen_style()));
        let candidates = vec![
            turbojev_runtime::RuntimeCandidate {
                symbol: "A".into(),
                token_ids: vec![32],
            },
            turbojev_runtime::RuntimeCandidate {
                symbol: "B".into(),
                token_ids: vec![33],
            },
        ];
        let full_a = vec![1, 2, 3, 10];
        let full_b = vec![1, 2, 3, 20];
        let reference_batch = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![],
            tasks: vec![
                turbojev_runtime::RuntimeTask {
                    id: "a".into(),
                    prompt_suffix: full_a.clone(),
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
                turbojev_runtime::RuntimeTask {
                    id: "b".into(),
                    prompt_suffix: full_b.clone(),
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
            ],
        };
        let shared_batch = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![1, 2, 3],
            tasks: vec![
                turbojev_runtime::RuntimeTask {
                    id: "a".into(),
                    prompt_suffix: vec![10],
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
                turbojev_runtime::RuntimeTask {
                    id: "b".into(),
                    prompt_suffix: vec![20],
                    candidates,
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
            ],
        };
        let reference = executor.score(&reference_batch, reference_plan()).unwrap();
        let shared = executor
            .score(
                &shared_batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Serial,
                    parallel_task_limit: 1,
                    projection: CandidateProjection::FullVocabulary,
                },
            )
            .unwrap();
        let report = compare_runtime_outputs(
            &shared_batch,
            &reference,
            &shared,
            RuntimeTolerance { logit_abs: 0.0 },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(shared.metrics.input_tokens, Some(5));
        assert_eq!(reference.metrics.input_tokens, Some(8));
    }

    #[test]
    fn semantic_answers_match_between_reference_and_deterministic_reuse() {
        use turbojev::engine::PlanningPolicy;
        use turbojev::schema::{EvaluationRequest, Question};

        let request = EvaluationRequest {
            model: None,
            state: serde_json::json!({"message":"package missing","days":12}),
            questions: [
                (
                    "route".to_owned(),
                    Question::choice(
                        "Which queue should handle this request?",
                        [
                            ("billing".to_owned(), "billing".to_owned()),
                            ("shipping".to_owned(), "shipping".to_owned()),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
                (
                    "urgent".to_owned(),
                    Question::noul("Does this require urgent review?"),
                ),
            ]
            .into_iter()
            .collect(),
        };

        let reference =
            turbojev::TurboJev::new(reference_backend(Renderer, FamilyApi::qwen_style()))
                .with_planning_policy(PlanningPolicy::Reference)
                .with_order_robustness(turbojev::OrderRobustness::None)
                .evaluate(request.clone())
                .unwrap();
        let reused = turbojev::TurboJev::new(reference_backend(Renderer, FamilyApi::qwen_style()))
            .with_planning_policy(PlanningPolicy::DeterministicReuse)
            .with_order_robustness(turbojev::OrderRobustness::None)
            .evaluate(request)
            .unwrap();

        assert_eq!(reference.answers, reused.answers);
        assert_eq!(reference.metadata.execution.prefix_reuse, "none");
        assert_eq!(reused.metadata.execution.prefix_reuse, "shared");
        assert_eq!(reference.usage.generated_tokens, 0);
        assert_eq!(reused.usage.generated_tokens, 0);
    }

    #[test]
    fn parallel_suffix_matches_reference_and_cached_serial_exactly() {
        use turbojev_runtime_conformance::{compare_runtime_outputs, RuntimeTolerance};

        let api = FamilyApi::qwen_style();
        let observed_batches = Arc::clone(&api.parallel_batches);
        let executor = LlamaCppExecutor::new(Arc::new(api));
        let candidates = vec![
            turbojev_runtime::RuntimeCandidate {
                symbol: "A".into(),
                token_ids: vec![32],
            },
            turbojev_runtime::RuntimeCandidate {
                symbol: "B".into(),
                token_ids: vec![33],
            },
        ];
        let prefix = vec![7, 11, 13, 17];
        let suffixes = [vec![], vec![19], vec![23, 29], vec![31], vec![37, 41, 43]];
        let reference_batch = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![],
            tasks: suffixes
                .iter()
                .enumerate()
                .map(|(index, suffix)| {
                    let mut full = prefix.clone();
                    full.extend_from_slice(suffix);
                    turbojev_runtime::RuntimeTask {
                        id: format!("q{index}"),
                        prompt_suffix: full,
                        candidates: candidates.clone(),
                        codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                    }
                })
                .collect(),
        };
        let shared_batch = turbojev_runtime::RuntimeBatch {
            common_prefix: prefix,
            tasks: suffixes
                .iter()
                .enumerate()
                .map(|(index, suffix)| turbojev_runtime::RuntimeTask {
                    id: format!("q{index}"),
                    prompt_suffix: suffix.clone(),
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                })
                .collect(),
        };

        let reference = executor.score(&reference_batch, reference_plan()).unwrap();
        let serial = executor
            .score(
                &shared_batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Serial,
                    parallel_task_limit: 1,
                    projection: CandidateProjection::FullVocabulary,
                },
            )
            .unwrap();
        let parallel = executor
            .score(
                &shared_batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Parallel,
                    parallel_task_limit: 3,
                    projection: CandidateProjection::FullVocabulary,
                },
            )
            .unwrap();

        for candidate in [&serial, &parallel] {
            let report = compare_runtime_outputs(
                &shared_batch,
                &reference,
                candidate,
                RuntimeTolerance { logit_abs: 0.0 },
            )
            .unwrap();
            assert!(report.passed);
        }
        assert_eq!(*observed_batches.lock().unwrap(), vec![3, 2]);
        assert_eq!(parallel.metrics.parallel_batches, Some(2));
        assert_eq!(parallel.metrics.peak_parallel_tasks, Some(3));
    }

    #[test]
    fn native_suffix_token_bound_further_splits_parallel_batches() {
        let api = FamilyApi::qwen_style().with_parallel_suffix_token_limit(2);
        let observed_batches = Arc::clone(&api.parallel_batches);
        let executor = LlamaCppExecutor::new(Arc::new(api));
        let candidates = vec![
            turbojev_runtime::RuntimeCandidate {
                symbol: "A".into(),
                token_ids: vec![32],
            },
            turbojev_runtime::RuntimeCandidate {
                symbol: "B".into(),
                token_ids: vec![33],
            },
        ];
        let batch = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![1, 2, 3],
            tasks: (0..5)
                .map(|index| turbojev_runtime::RuntimeTask {
                    id: format!("q{index}"),
                    prompt_suffix: vec![10 + index],
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                })
                .collect(),
        };

        let output = executor
            .score(
                &batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Parallel,
                    parallel_task_limit: 6,
                    projection: CandidateProjection::FullVocabulary,
                },
            )
            .unwrap();

        assert_eq!(output.scores.len(), 5);
        assert_eq!(*observed_batches.lock().unwrap(), vec![2, 2]);
        assert_eq!(output.metrics.parallel_batches, Some(2));
        assert_eq!(output.metrics.peak_parallel_tasks, Some(2));
    }

    #[test]
    fn auto_parallel_respects_portable_memory_budget() {
        use turbojev::engine::ExecutionBudget;
        use turbojev::schema::{EvaluationRequest, Question};

        let api = FamilyApi::qwen_style();
        let observed_batches = Arc::clone(&api.parallel_batches);
        let backend = reference_backend(Renderer, api);
        let request = EvaluationRequest {
            model: None,
            state: serde_json::json!({"message":"package missing","days":12}),
            questions: (0..5)
                .map(|index| {
                    (
                        format!("q{index}"),
                        Question::noul(format!("Does condition {index} hold?")),
                    )
                })
                .collect(),
        };

        let response = turbojev::TurboJev::new(backend)
            .with_execution_budget(ExecutionBudget::unlimited().with_memory_budget_bytes(8 * 1024))
            .evaluate(request)
            .unwrap();

        assert_eq!(response.metadata.execution.prefix_reuse, "shared");
        assert_eq!(response.metadata.execution.scheduling, "parallel");
        assert_eq!(response.metadata.execution.parallel_task_limit, 2);
        assert_eq!(*observed_batches.lock().unwrap(), vec![2, 2]);
    }

    #[test]
    fn candidate_only_projection_matches_full_vocabulary_and_uses_selective_api() {
        use turbojev_runtime_conformance::{verify_selective_projection, RuntimeTolerance};

        let api = FamilyApi::qwen_style();
        let full_calls = Arc::clone(&api.full_projection_calls);
        let selective_calls = Arc::clone(&api.selective_projection_calls);
        let executor = LlamaCppExecutor::new(Arc::new(api));
        let candidates = vec![
            turbojev_runtime::RuntimeCandidate {
                symbol: "A".into(),
                token_ids: vec![32],
            },
            turbojev_runtime::RuntimeCandidate {
                symbol: "B".into(),
                token_ids: vec![33],
            },
        ];
        let batch = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![1, 2, 3],
            tasks: vec![
                turbojev_runtime::RuntimeTask {
                    id: "a".into(),
                    prompt_suffix: vec![10],
                    candidates: candidates.clone(),
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
                turbojev_runtime::RuntimeTask {
                    id: "b".into(),
                    prompt_suffix: vec![20, 30],
                    candidates,
                    codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
                },
            ],
        };

        let full = executor
            .score(
                &batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Parallel,
                    parallel_task_limit: 2,
                    projection: CandidateProjection::FullVocabulary,
                },
            )
            .unwrap();
        let full_after = *full_calls.lock().unwrap();
        assert!(full_after > 0);

        let selected = executor
            .score(
                &batch,
                ExecutionPlan {
                    prefix_reuse: PrefixReuse::Shared,
                    scheduling: TaskScheduling::Parallel,
                    parallel_task_limit: 2,
                    projection: CandidateProjection::CandidatesOnly,
                },
            )
            .unwrap();
        assert_eq!(*full_calls.lock().unwrap(), full_after);
        assert!(*selective_calls.lock().unwrap() > 0);

        let report = verify_selective_projection(
            &batch,
            &full,
            &selected,
            RuntimeTolerance { logit_abs: 0.0 },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.expected_candidate_rows, 4);
        assert_eq!(report.reported_candidate_rows, 4);
    }

    #[test]
    fn multi_token_fallback_is_exact_and_downgrades_unsafe_optimizations() {
        use turbojev_backend::ScoringBackend;

        let backend = reference_backend(Renderer, FamilyApi::multi_token_style());
        let decision_batch = DecisionBatch {
            state: serde_json::json!({"message":"multi token fixture"}),
            tasks: vec![
                DecisionTask {
                    id: "q1".into(),
                    intent: DecisionIntent::Criterion,
                    instruction: Some("first".into()),
                    candidates: vec![
                        Candidate {
                            key: "a".into(),
                            description: "alpha".into(),
                        },
                        Candidate {
                            key: "b".into(),
                            description: "beta".into(),
                        },
                    ],
                },
                DecisionTask {
                    id: "q2".into(),
                    intent: DecisionIntent::Criterion,
                    instruction: Some("second".into()),
                    candidates: vec![
                        Candidate {
                            key: "a".into(),
                            description: "alpha".into(),
                        },
                        Candidate {
                            key: "b".into(),
                            description: "beta".into(),
                        },
                    ],
                },
            ],
        };

        let reference = ScoringBackend::score(&backend, &decision_batch, reference_plan()).unwrap();
        let requested = ExecutionPlan {
            prefix_reuse: PrefixReuse::Shared,
            scheduling: TaskScheduling::Parallel,
            parallel_task_limit: 2,
            projection: CandidateProjection::CandidatesOnly,
        };
        let optimized_request =
            ScoringBackend::score(&backend, &decision_batch, requested).unwrap();

        let effective = optimized_request.metrics.effective_plan.unwrap();
        assert_eq!(effective.prefix_reuse, PrefixReuse::Shared);
        assert_eq!(effective.scheduling, TaskScheduling::Serial);
        assert_eq!(effective.parallel_task_limit, 1);
        assert_eq!(effective.projection, CandidateProjection::FullVocabulary);
        assert!(
            optimized_request
                .metrics
                .sequence_candidate_tokens
                .unwrap_or(0)
                > 0
        );
        assert!(
            optimized_request
                .metrics
                .sequence_prefix_evaluations
                .unwrap_or(0)
                > 0
        );
        assert!(optimized_request
            .metrics
            .sequence_token_steps_saved
            .is_some());

        assert_eq!(reference.scores.len(), optimized_request.scores.len());
        for (left, right) in reference.scores.iter().zip(&optimized_request.scores) {
            assert_eq!(left.id, right.id);
            assert_eq!(left.logits, right.logits);
        }
    }

    #[test]
    fn planner_auto_selects_candidate_only_when_runtime_advertises_parity() {
        use turbojev::engine::PlanningPolicy;
        use turbojev::schema::{EvaluationRequest, Question};

        let response =
            turbojev::TurboJev::new(reference_backend(Renderer, FamilyApi::qwen_style()))
                .with_planning_policy(PlanningPolicy::Auto)
                .evaluate(EvaluationRequest {
                    model: None,
                    state: serde_json::json!({"message":"package missing"}),
                    questions: [
                        (
                            "route".to_owned(),
                            Question::choice(
                                "Which queue?",
                                [
                                    ("billing".to_owned(), "billing".to_owned()),
                                    ("shipping".to_owned(), "shipping".to_owned()),
                                ]
                                .into_iter()
                                .collect(),
                            ),
                        ),
                        (
                            "urgent".to_owned(),
                            Question::noul("Does it require review?"),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                })
                .unwrap();

        assert_eq!(response.metadata.execution.projection, "candidates-only");
        assert_eq!(response.usage.generated_tokens, 0);
    }

    #[test]
    fn multi_root_runtime_matches_independent_roots_and_rejects_multitoken_shape() {
        let executor = LlamaCppExecutor::new(Arc::new(FamilyApi::qwen_style()));
        let candidates = vec![
            turbojev_runtime::RuntimeCandidate {
                symbol: "A".into(),
                token_ids: vec![32],
            },
            turbojev_runtime::RuntimeCandidate {
                symbol: "B".into(),
                token_ids: vec![33],
            },
        ];
        let first = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![1, 2],
            tasks: vec![turbojev_runtime::RuntimeTask {
                id: "a".into(),
                prompt_suffix: vec![10, 11],
                candidates: candidates.clone(),
                codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
            }],
        };
        let second = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![3, 4],
            tasks: vec![turbojev_runtime::RuntimeTask {
                id: "b".into(),
                prompt_suffix: vec![20],
                candidates,
                codebook_kind: turbojev_tokenizer::CodebookKind::SingleToken,
            }],
        };
        let plan = ExecutionPlan::default();
        assert!(executor
            .can_score_fused(&[first.clone(), second.clone()], &[plan, plan])
            .unwrap());
        let fused = executor
            .score_fused(&[first.clone(), second.clone()], &[plan, plan])
            .unwrap();
        let independent = |batch: &turbojev_runtime::RuntimeBatch| {
            let mut full = batch.clone();
            for task in &mut full.tasks {
                task.prompt_suffix
                    .splice(0..0, full.common_prefix.iter().copied());
            }
            full.common_prefix.clear();
            executor.score(&full, plan).unwrap()
        };
        let serial_first = independent(&first);
        let serial_second = independent(&second);
        assert_eq!(fused[0].scores, serial_first.scores);
        assert_eq!(fused[1].scores, serial_second.scores);
        assert_eq!(fused[0].metrics.fused_request_count, Some(2));
        assert_eq!(fused[0].metrics.fused_root_count, Some(2));

        let multi = turbojev_runtime::RuntimeBatch {
            common_prefix: vec![1],
            tasks: vec![turbojev_runtime::RuntimeTask {
                id: "multi".into(),
                prompt_suffix: vec![2],
                candidates: vec![
                    turbojev_runtime::RuntimeCandidate {
                        symbol: "A".into(),
                        token_ids: vec![32, 32],
                    },
                    turbojev_runtime::RuntimeCandidate {
                        symbol: "B".into(),
                        token_ids: vec![32, 33],
                    },
                ],
                codebook_kind: turbojev_tokenizer::CodebookKind::FixedLengthSequence {
                    tokens_per_code: 2,
                },
            }],
        };
        assert!(!executor
            .can_score_fused(&[first, multi], &[plan, plan])
            .unwrap());
    }
}
