#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Model-bound specialization artifacts for TurboJev.
//!
//! This crate is deliberately *not* a trainer. It connects the portable Phase
//! 10 recipe/materializer to the exact prompt renderer and tokenizer/chat
//! template of an open-weight checkpoint. The output is a framework-neutral
//! dataset that an external LoRA/QLoRA implementation can consume without
//! re-inventing TurboJev's candidate semantics.
//!
//! There is no filesystem, network, GPU, process, optimizer or training
//! framework dependency here.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_calibration::CalibrationProfile;
use turbojev_model::CandidateEncodingPolicy;
use turbojev_prompt::{compile_document, PromptRenderer, RenderedMessages};
use turbojev_runtime::{PromptCodec, TokenId};
use turbojev_runtime_conformance::{verify_candidate_boundary, VerifiedCandidateCode};
use turbojev_specialized::{
    BaseModelIdentity, SpecializedModelManifest, TrainingObjective, TrainingRecipe,
    SPECIALIZED_MODEL_CONTRACT,
};
use turbojev_tokenizer::candidate_symbol_sets;
use turbojev_training_data::{
    AugmentationProvenance, CandidateCodeProvider, CandidateCodeRequest,
    MaterializedTrainingDataset,
};

/// Versioned framework-neutral dataset after exact renderer/tokenizer binding.
pub const MODEL_BOUND_DATASET_CONTRACT: &str = "turbojev.model-bound-training-dataset.v1";

/// Configuration for deterministic candidate-code discovery at the *real*
/// assistant answer boundary of a concrete checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodebookDiscoveryConfig {
    /// Stable candidate-code policy/profile persisted by the materializer.
    pub profile: String,
    /// Maximum token width accepted while probing. Candidate-logit CE v1 still
    /// requires the final selected codebook to be single-token.
    pub max_tokens_per_code: usize,
    /// Maximum deterministic symbol sets to probe.
    pub candidate_set_attempts: usize,
}

impl Default for CodebookDiscoveryConfig {
    fn default() -> Self {
        Self {
            profile: "turbojev-dynamic-single-token-v1".to_owned(),
            max_tokens_per_code: 4,
            candidate_set_attempts: 96,
        }
    }
}

impl CodebookDiscoveryConfig {
    pub fn validate(&self) -> Result<(), ArtifactError> {
        require_text("codebook.profile", &self.profile)?;
        if self.max_tokens_per_code == 0 {
            return Err(ArtifactError::InvalidPositive("codebook.max_tokens_per_code"));
        }
        if self.candidate_set_attempts == 0 {
            return Err(ArtifactError::InvalidPositive("codebook.candidate_set_attempts"));
        }
        Ok(())
    }
}

/// Candidate-code provider backed by the exact model renderer + tokenizer/chat
/// template. It can be fed directly to `turbojev_training_data::materialize`.
///
/// The v1 specialization objective is candidate-logit cross entropy, therefore
/// this provider deliberately accepts only codebooks whose candidates are each
/// exactly one token at the answer boundary. Multi-token inference remains
/// supported elsewhere in TurboJev; it is not silently treated as the same
/// training objective.
pub struct BoundaryVerifiedCodeProvider<'a, R, C> {
    renderer: &'a R,
    codec: &'a C,
    config: CodebookDiscoveryConfig,
}

impl<'a, R, C> BoundaryVerifiedCodeProvider<'a, R, C>
where
    R: PromptRenderer,
    C: PromptCodec,
{
    pub fn new(
        renderer: &'a R,
        codec: &'a C,
        config: CodebookDiscoveryConfig,
    ) -> Result<Self, ArtifactError> {
        config.validate()?;
        require_text("renderer.profile", renderer.profile())?;
        require_text("codec.profile", codec.profile())?;
        Ok(Self {
            renderer,
            codec,
            config,
        })
    }

    pub fn renderer(&self) -> &R {
        self.renderer
    }

    pub fn codec(&self) -> &C {
        self.codec
    }

    pub fn config(&self) -> &CodebookDiscoveryConfig {
        &self.config
    }

    fn discover(&self, request: CandidateCodeRequest<'_>) -> Result<Vec<String>, ArtifactError> {
        let symbol_sets = candidate_symbol_sets(
            request.required_codes,
            self.config.candidate_set_attempts,
        )
        .map_err(|error| ArtifactError::Codebook(error.to_string()))?;

        for symbols in symbol_sets {
            let document = compile_document(request.evidence, request.task, &symbols)
                .map_err(|error| ArtifactError::Prompt(error.to_string()))?;
            let messages = self
                .renderer
                .render(&document)
                .map_err(|error| ArtifactError::Prompt(error.to_string()))?;
            if messages.profile != self.renderer.profile() {
                return Err(ArtifactError::RenderedProfileMismatch {
                    expected: self.renderer.profile().to_owned(),
                    actual: messages.profile,
                });
            }
            let verified = match verify_candidate_boundary(
                self.codec,
                &messages,
                &symbols,
                self.config.max_tokens_per_code,
            ) {
                Ok(verified) => verified,
                Err(_) => continue,
            };
            if verified.iter().all(|code| code.token_ids.len() == 1) {
                return Ok(symbols);
            }
        }

        Err(ArtifactError::NoSingleTokenCodebook {
            example: request.source_example_id.to_owned(),
            question: request.question_id.to_owned(),
            candidates: request.required_codes,
        })
    }
}

impl<R, C> CandidateCodeProvider for BoundaryVerifiedCodeProvider<'_, R, C>
where
    R: PromptRenderer,
    C: PromptCodec,
{
    fn profile(&self) -> &str {
        &self.config.profile
    }

    fn codes_for(&self, request: CandidateCodeRequest<'_>) -> Result<Vec<String>, String> {
        self.discover(request).map_err(|error| error.to_string())
    }
}

/// Exact model-facing candidate exported for external training.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundCandidate {
    /// Stable application meaning.
    pub key: String,
    /// Candidate symbol rendered in the prompt.
    pub code: String,
    /// Exact answer-boundary token sequence. v1 requires length == 1.
    pub token_ids: Vec<TokenId>,
}

/// One fully model-bound candidate-logit CE record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelBoundTrainingRecord {
    pub id: String,
    pub source_record_id: String,
    pub source_example_id: String,
    pub question_id: String,
    pub language: String,
    /// Exact messages produced by the recipe-pinned renderer.
    pub messages: RenderedMessages,
    /// Exact chat-template/tokenizer ids up to the assistant answer boundary.
    pub prompt_token_ids: Vec<TokenId>,
    pub candidates: Vec<BoundCandidate>,
    pub target_key: String,
    pub target_index: usize,
    pub target_code: String,
    /// Single answer token selected by candidate-logit CE v1.
    pub target_token_id: TokenId,
    pub weight: f64,
    pub augmentation: AugmentationProvenance,
}

/// Audit counters for one model-bound export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelBoundStats {
    pub source_records: usize,
    pub records: usize,
    pub total_record_weight: f64,
    pub max_prompt_tokens: usize,
    pub max_context_tokens_used: usize,
}

/// Portable export consumed by any external trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelBoundTrainingDataset {
    pub contract: String,
    pub id: String,
    pub source_materialized_dataset: String,
    pub source_semantic_dataset: String,
    pub recipe_id: String,
    pub objective: TrainingObjective,
    pub model: BaseModelIdentity,
    pub prompt_ir: String,
    pub renderer_profile: String,
    pub tokenizer_profile: String,
    pub candidate_code_profile: String,
    pub max_context_tokens: usize,
    pub records: Vec<ModelBoundTrainingRecord>,
    pub stats: ModelBoundStats,
}

impl ModelBoundTrainingDataset {
    pub fn validate(&self) -> Result<(), ArtifactError> {
        if self.contract != MODEL_BOUND_DATASET_CONTRACT {
            return Err(ArtifactError::UnsupportedContract(self.contract.clone()));
        }
        for (field, value) in [
            ("dataset.id", self.id.as_str()),
            (
                "dataset.source_materialized_dataset",
                self.source_materialized_dataset.as_str(),
            ),
            (
                "dataset.source_semantic_dataset",
                self.source_semantic_dataset.as_str(),
            ),
            ("dataset.recipe_id", self.recipe_id.as_str()),
            ("dataset.prompt_ir", self.prompt_ir.as_str()),
            ("dataset.renderer_profile", self.renderer_profile.as_str()),
            ("dataset.tokenizer_profile", self.tokenizer_profile.as_str()),
            (
                "dataset.candidate_code_profile",
                self.candidate_code_profile.as_str(),
            ),
        ] {
            require_text(field, value)?;
        }
        require_base_model(&self.model)?;
        if self.max_context_tokens == 0 {
            return Err(ArtifactError::InvalidPositive("dataset.max_context_tokens"));
        }
        if self.records.is_empty() {
            return Err(ArtifactError::EmptyDataset);
        }
        if self.stats.source_records == 0 || self.stats.records != self.records.len() {
            return Err(ArtifactError::StatsMismatch);
        }

        let mut ids = HashSet::with_capacity(self.records.len());
        let mut total_weight = 0.0_f64;
        let mut max_prompt = 0_usize;
        let mut max_context = 0_usize;
        for record in &self.records {
            if !ids.insert(record.id.as_str()) {
                return Err(ArtifactError::DuplicateRecord(record.id.clone()));
            }
            for (field, value) in [
                ("record.id", record.id.as_str()),
                ("record.source_record_id", record.source_record_id.as_str()),
                ("record.source_example_id", record.source_example_id.as_str()),
                ("record.question_id", record.question_id.as_str()),
                ("record.language", record.language.as_str()),
                ("record.target_key", record.target_key.as_str()),
                ("record.target_code", record.target_code.as_str()),
            ] {
                require_text(field, value)?;
            }
            if record.messages.profile != self.renderer_profile {
                return Err(ArtifactError::RenderedProfileMismatch {
                    expected: self.renderer_profile.clone(),
                    actual: record.messages.profile.clone(),
                });
            }
            if record.prompt_token_ids.is_empty() {
                return Err(ArtifactError::EmptyPrompt(record.id.clone()));
            }
            if record.candidates.len() < 2 {
                return Err(ArtifactError::TooFewCandidates(record.id.clone()));
            }
            if record.target_index >= record.candidates.len() {
                return Err(ArtifactError::TargetOutOfRange(record.id.clone()));
            }
            if !record.weight.is_finite() || record.weight <= 0.0 {
                return Err(ArtifactError::InvalidWeight(record.id.clone()));
            }

            let mut keys = HashSet::with_capacity(record.candidates.len());
            let mut codes = HashSet::with_capacity(record.candidates.len());
            let mut tokens = HashSet::with_capacity(record.candidates.len());
            for candidate in &record.candidates {
                require_text("candidate.key", &candidate.key)?;
                require_text("candidate.code", &candidate.code)?;
                if candidate.token_ids.len() != 1 {
                    return Err(ArtifactError::CandidateLogitRequiresSingleToken {
                        record: record.id.clone(),
                        code: candidate.code.clone(),
                        width: candidate.token_ids.len(),
                    });
                }
                if !keys.insert(candidate.key.as_str())
                    || !codes.insert(candidate.code.as_str())
                    || !tokens.insert(candidate.token_ids[0])
                {
                    return Err(ArtifactError::CandidateCollision(record.id.clone()));
                }
            }
            let target = &record.candidates[record.target_index];
            if target.key != record.target_key
                || target.code != record.target_code
                || target.token_ids[0] != record.target_token_id
            {
                return Err(ArtifactError::TargetMismatch(record.id.clone()));
            }

            let used = record.prompt_token_ids.len() + 1;
            if used > self.max_context_tokens {
                return Err(ArtifactError::ContextLimit {
                    record: record.id.clone(),
                    used,
                    limit: self.max_context_tokens,
                });
            }
            total_weight += record.weight;
            max_prompt = max_prompt.max(record.prompt_token_ids.len());
            max_context = max_context.max(used);
        }

        if (total_weight - self.stats.total_record_weight).abs() > 1e-9
            || max_prompt != self.stats.max_prompt_tokens
            || max_context != self.stats.max_context_tokens_used
        {
            return Err(ArtifactError::StatsMismatch);
        }
        Ok(())
    }
}

/// Bind a deterministic v0.18 materialized dataset to the exact renderer and
/// tokenizer/chat-template of the recipe's open-weight checkpoint.
pub fn bind_materialized_dataset<R, C>(
    output_id: impl Into<String>,
    materialized: &MaterializedTrainingDataset,
    recipe: &TrainingRecipe,
    renderer: &R,
    codec: &C,
) -> Result<ModelBoundTrainingDataset, ArtifactError>
where
    R: PromptRenderer,
    C: PromptCodec,
{
    materialized
        .validate()
        .map_err(|error| ArtifactError::Materialized(error.to_string()))?;
    recipe
        .validate()
        .map_err(|error| ArtifactError::Recipe(error.to_string()))?;
    validate_materialized_provenance(materialized, recipe, renderer, codec)?;

    let mut records = Vec::with_capacity(materialized.records.len());
    let mut max_prompt_tokens = 0_usize;
    let mut max_context_tokens_used = 0_usize;
    let mut total_record_weight = 0.0_f64;

    for source in &materialized.records {
        let messages = renderer
            .render(&source.document)
            .map_err(|error| ArtifactError::Prompt(error.to_string()))?;
        if messages.profile != renderer.profile() {
            return Err(ArtifactError::RenderedProfileMismatch {
                expected: renderer.profile().to_owned(),
                actual: messages.profile,
            });
        }
        let prompt_token_ids = codec
            .encode_prompt(&messages)
            .map_err(|error| ArtifactError::Codec(error.to_string()))?;
        if prompt_token_ids.is_empty() {
            return Err(ArtifactError::EmptyPrompt(source.id.clone()));
        }
        let symbols = source
            .document
            .options
            .iter()
            .map(|option| option.code.clone())
            .collect::<Vec<_>>();
        let verified = verify_candidate_boundary(codec, &messages, &symbols, 1)
            .map_err(|error| ArtifactError::Boundary {
                record: source.id.clone(),
                message: error.to_string(),
            })?;
        ensure_single_token_objective(source.id.as_str(), recipe.objective, &verified)?;

        let candidates = source
            .document
            .options
            .iter()
            .zip(&verified)
            .map(|(option, verified)| BoundCandidate {
                key: option.key.clone(),
                code: option.code.clone(),
                token_ids: verified.token_ids.clone(),
            })
            .collect::<Vec<_>>();
        let target = candidates
            .get(source.target_index)
            .ok_or_else(|| ArtifactError::TargetOutOfRange(source.id.clone()))?;
        if target.key != source.target_key || target.code != source.target_code {
            return Err(ArtifactError::TargetMismatch(source.id.clone()));
        }
        let target_token_id = target.token_ids[0];
        let context_used = prompt_token_ids.len() + 1;
        if context_used > recipe.max_context_tokens {
            return Err(ArtifactError::ContextLimit {
                record: source.id.clone(),
                used: context_used,
                limit: recipe.max_context_tokens,
            });
        }
        max_prompt_tokens = max_prompt_tokens.max(prompt_token_ids.len());
        max_context_tokens_used = max_context_tokens_used.max(context_used);
        total_record_weight += source.weight;

        records.push(ModelBoundTrainingRecord {
            id: format!("{}::bound", source.id),
            source_record_id: source.id.clone(),
            source_example_id: source.source_example_id.clone(),
            question_id: source.question_id.clone(),
            language: source.language.clone(),
            messages,
            prompt_token_ids,
            candidates,
            target_key: source.target_key.clone(),
            target_index: source.target_index,
            target_code: source.target_code.clone(),
            target_token_id,
            weight: source.weight,
            augmentation: source.augmentation.clone(),
        });
    }

    let dataset = ModelBoundTrainingDataset {
        contract: MODEL_BOUND_DATASET_CONTRACT.to_owned(),
        id: output_id.into(),
        source_materialized_dataset: materialized.id.clone(),
        source_semantic_dataset: materialized.source_dataset.clone(),
        recipe_id: recipe.id.clone(),
        objective: recipe.objective,
        model: recipe.base_model.clone(),
        prompt_ir: recipe.prompt_ir.clone(),
        renderer_profile: renderer.profile().to_owned(),
        tokenizer_profile: codec.profile().to_owned(),
        candidate_code_profile: materialized.candidate_code_profile.clone(),
        max_context_tokens: recipe.max_context_tokens,
        stats: ModelBoundStats {
            source_records: materialized.records.len(),
            records: records.len(),
            total_record_weight,
            max_prompt_tokens,
            max_context_tokens_used,
        },
        records,
    };
    dataset.validate()?;
    Ok(dataset)
}

fn validate_materialized_provenance<R, C>(
    materialized: &MaterializedTrainingDataset,
    recipe: &TrainingRecipe,
    renderer: &R,
    codec: &C,
) -> Result<(), ArtifactError>
where
    R: PromptRenderer,
    C: PromptCodec,
{
    if materialized.recipe_id != recipe.id {
        return Err(ArtifactError::ProvenanceMismatch("recipe_id"));
    }
    if materialized.source_dataset != recipe.splits.train {
        return Err(ArtifactError::ProvenanceMismatch("training_dataset"));
    }
    if materialized.prompt_ir != recipe.prompt_ir {
        return Err(ArtifactError::ProvenanceMismatch("prompt_ir"));
    }
    if materialized.renderer_profile != recipe.renderer_profile
        || renderer.profile() != recipe.renderer_profile
    {
        return Err(ArtifactError::ProvenanceMismatch("renderer_profile"));
    }
    require_text("codec.profile", codec.profile())?;
    if let CandidateEncodingPolicy::Profile(expected) = &recipe.candidate_encoding {
        if materialized.candidate_code_profile.as_str() != expected.as_str() {
            return Err(ArtifactError::ProvenanceMismatch("candidate_encoding.profile"));
        }
    }
    Ok(())
}

fn ensure_single_token_objective(
    record: &str,
    objective: TrainingObjective,
    candidates: &[VerifiedCandidateCode],
) -> Result<(), ArtifactError> {
    match objective {
        TrainingObjective::CandidateLogitCrossEntropy => {
            for candidate in candidates {
                if candidate.token_ids.len() != 1 {
                    return Err(ArtifactError::CandidateLogitRequiresSingleToken {
                        record: record.to_owned(),
                        code: candidate.symbol.clone(),
                        width: candidate.token_ids.len(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Construct the manifest that an externally trained adapter/merged checkpoint
/// must ship. The external trainer only supplies the produced artifact id and
/// immutable revision/hash; TurboJev supplies the semantic contract.
pub fn build_specialized_manifest(
    recipe: &TrainingRecipe,
    model_id: impl Into<String>,
    revision: impl Into<String>,
    calibration: Option<&CalibrationProfile>,
) -> Result<SpecializedModelManifest, ArtifactError> {
    recipe
        .validate()
        .map_err(|error| ArtifactError::Recipe(error.to_string()))?;
    let model_id = model_id.into();
    let revision = revision.into();
    require_text("manifest.model_id", &model_id)?;
    require_text("manifest.revision", &revision)?;

    let calibration_id = if let Some(profile) = calibration {
        validate_calibration_provenance(recipe, &model_id, profile)?;
        Some(profile.id.clone())
    } else {
        None
    };

    let manifest = SpecializedModelManifest {
        contract: SPECIALIZED_MODEL_CONTRACT.to_owned(),
        id: model_id,
        revision,
        recipe_id: recipe.id.clone(),
        base_model: recipe.base_model.clone(),
        decision_contract: recipe.decision_contract(calibration_id.clone()),
        training_dataset: recipe.splits.train.clone(),
        calibration_profile: calibration_id,
    };
    manifest
        .validate_against_recipe(recipe)
        .map_err(|error| ArtifactError::Manifest(error.to_string()))?;
    Ok(manifest)
}

/// Verify an independently fitted temperature profile before attaching it to a
/// specialized checkpoint. Missing provenance is rejected rather than guessed.
pub fn validate_calibration_handoff(
    recipe: &TrainingRecipe,
    manifest: &SpecializedModelManifest,
    profile: &CalibrationProfile,
) -> Result<(), ArtifactError> {
    recipe
        .validate()
        .map_err(|error| ArtifactError::Recipe(error.to_string()))?;
    manifest
        .validate_against_recipe(recipe)
        .map_err(|error| ArtifactError::Manifest(error.to_string()))?;
    validate_calibration_provenance(recipe, &manifest.id, profile)?;
    if manifest.calibration_profile.as_deref() != Some(profile.id.as_str()) {
        return Err(ArtifactError::CalibrationMismatch("profile_id"));
    }
    Ok(())
}

fn validate_calibration_provenance(
    recipe: &TrainingRecipe,
    model_id: &str,
    profile: &CalibrationProfile,
) -> Result<(), ArtifactError> {
    profile
        .validate()
        .map_err(|error| ArtifactError::Calibration(error.to_string()))?;
    if profile.holdout_id != recipe.splits.calibration {
        return Err(ArtifactError::CalibrationMismatch("holdout_id"));
    }
    if profile.model_id.as_deref() != Some(model_id) {
        return Err(ArtifactError::CalibrationMismatch("model_id"));
    }
    if profile.tokenizer_revision.as_deref()
        != Some(recipe.base_model.tokenizer_revision.as_str())
    {
        return Err(ArtifactError::CalibrationMismatch("tokenizer_revision"));
    }
    if profile.prompt_profile.as_deref() != Some(recipe.prompt_ir.as_str()) {
        return Err(ArtifactError::CalibrationMismatch("prompt_profile"));
    }
    if profile.renderer_profile.as_deref() != Some(recipe.renderer_profile.as_str()) {
        return Err(ArtifactError::CalibrationMismatch("renderer_profile"));
    }
    Ok(())
}

/// Runtime-side identity supplied by the selected adapter/model combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecializedRuntimeIdentity {
    pub prompt_ir: String,
    pub renderer_profile: String,
    pub tokenizer_revision: String,
    pub candidate_code_profile: Option<String>,
    pub max_context_tokens: usize,
    pub calibration_profile: Option<String>,
}

/// Fail-fast compatibility gate used before specialized inference starts.
pub fn validate_runtime_compatibility(
    manifest: &SpecializedModelManifest,
    runtime: &SpecializedRuntimeIdentity,
) -> Result<(), ArtifactError> {
    manifest
        .validate()
        .map_err(|error| ArtifactError::Manifest(error.to_string()))?;
    let expected = &manifest.decision_contract;
    if runtime.prompt_ir != expected.prompt_ir {
        return Err(ArtifactError::RuntimeMismatch("prompt_ir"));
    }
    if runtime.renderer_profile != expected.renderer_profile {
        return Err(ArtifactError::RuntimeMismatch("renderer_profile"));
    }
    if expected.tokenizer_revision.as_deref() != Some(runtime.tokenizer_revision.as_str()) {
        return Err(ArtifactError::RuntimeMismatch("tokenizer_revision"));
    }
    if runtime.max_context_tokens < expected.max_context_tokens.unwrap_or(0) {
        return Err(ArtifactError::RuntimeMismatch("max_context_tokens"));
    }
    if runtime.calibration_profile.as_deref() != expected.calibration_profile.as_deref() {
        return Err(ArtifactError::RuntimeMismatch("calibration_profile"));
    }
    if let CandidateEncodingPolicy::Profile(profile) = &expected.candidate_encoding {
        if runtime.candidate_code_profile.as_deref() != Some(profile.as_str()) {
            return Err(ArtifactError::RuntimeMismatch("candidate_encoding.profile"));
        }
    }
    Ok(())
}

fn require_base_model(model: &BaseModelIdentity) -> Result<(), ArtifactError> {
    require_text("model.id", &model.id)?;
    require_text("model.revision", &model.revision)?;
    require_text("model.tokenizer_revision", &model.tokenizer_revision)
}

fn require_text(field: &'static str, value: &str) -> Result<(), ArtifactError> {
    if value.trim().is_empty() {
        Err(ArtifactError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ArtifactError {
    #[error("unsupported model-bound training dataset contract {0:?}")]
    UnsupportedContract(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must be greater than zero")]
    InvalidPositive(&'static str),
    #[error("training recipe is invalid: {0}")]
    Recipe(String),
    #[error("materialized training dataset is invalid: {0}")]
    Materialized(String),
    #[error("prompt rendering failed: {0}")]
    Prompt(String),
    #[error("prompt codec failed: {0}")]
    Codec(String),
    #[error("candidate-code discovery failed: {0}")]
    Codebook(String),
    #[error("no exact single-token codebook for {example:?}/{question:?} with {candidates} candidates")]
    NoSingleTokenCodebook {
        example: String,
        question: String,
        candidates: usize,
    },
    #[error("rendered profile mismatch: expected {expected:?}, got {actual:?}")]
    RenderedProfileMismatch { expected: String, actual: String },
    #[error("candidate boundary verification failed for record {record:?}: {message}")]
    Boundary { record: String, message: String },
    #[error("candidate-logit CE requires one-token codes; record {record:?} code {code:?} has width {width}")]
    CandidateLogitRequiresSingleToken {
        record: String,
        code: String,
        width: usize,
    },
    #[error("model-bound dataset contains no records")]
    EmptyDataset,
    #[error("duplicate model-bound record id {0:?}")]
    DuplicateRecord(String),
    #[error("model-bound record {0:?} has an empty prompt token sequence")]
    EmptyPrompt(String),
    #[error("model-bound record {0:?} has fewer than two candidates")]
    TooFewCandidates(String),
    #[error("model-bound record {0:?} has duplicate keys, codes or token ids")]
    CandidateCollision(String),
    #[error("model-bound record {0:?} target index is out of range")]
    TargetOutOfRange(String),
    #[error("model-bound record {0:?} target provenance does not match candidate order")]
    TargetMismatch(String),
    #[error("model-bound record {0:?} weight must be finite and greater than zero")]
    InvalidWeight(String),
    #[error("model-bound record {record:?} uses {used} tokens, exceeding context limit {limit}")]
    ContextLimit {
        record: String,
        used: usize,
        limit: usize,
    },
    #[error("model-bound dataset audit statistics do not match its records")]
    StatsMismatch,
    #[error("materialized/recipe provenance mismatch for {0}")]
    ProvenanceMismatch(&'static str),
    #[error("specialized manifest is invalid: {0}")]
    Manifest(String),
    #[error("calibration profile is invalid: {0}")]
    Calibration(String),
    #[error("calibration handoff mismatch for {0}")]
    CalibrationMismatch(&'static str),
    #[error("specialized runtime compatibility mismatch for {0}")]
    RuntimeMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_model::CandidateEncodingPolicy;
    use turbojev_prompt::{PromptDocument, PromptError};
    use turbojev_runtime::RuntimeError;
    use turbojev_specialized::{
        AdapterConfig, DataPolicy, DatasetSplits, DebiasingPolicy, FineTuneMethod,
        OptimizationConfig, TRAINING_RECIPE_CONTRACT,
    };

    struct FixtureRenderer;

    impl PromptRenderer for FixtureRenderer {
        fn profile(&self) -> &str {
            "fixture-renderer-v1"
        }

        fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError> {
            let options = document
                .options
                .iter()
                .map(|option| format!("{}={}", option.code, option.key))
                .collect::<Vec<_>>()
                .join("|");
            Ok(RenderedMessages {
                system: Some("fixture".into()),
                user: format!("state:{};options:{options}", document.evidence),
                profile: self.profile().to_owned(),
            })
        }
    }

    struct FixtureCodec;

    impl PromptCodec for FixtureCodec {
        fn profile(&self) -> &str {
            "fixture-tokenizer-v1"
        }

        fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError> {
            Ok(messages.user.bytes().map(u32::from).collect())
        }

        fn encode_prompt_with_answer(
            &self,
            messages: &RenderedMessages,
            answer: &str,
        ) -> Result<Vec<TokenId>, RuntimeError> {
            let mut ids = self.encode_prompt(messages)?;
            let token = match answer {
                "A" => 1001,
                "B" => 1002,
                "C" => 1003,
                "D" => 1004,
                _ => return Err(RuntimeError::Codec("fixture symbol rejected".into())),
            };
            ids.push(token);
            Ok(ids)
        }

        fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError> {
            if token_ids.len() != 1 {
                return Err(RuntimeError::Codec("fixture width rejected".into()));
            }
            match token_ids[0] {
                1001 => Ok("A".into()),
                1002 => Ok("B".into()),
                1003 => Ok("C".into()),
                1004 => Ok("D".into()),
                _ => Err(RuntimeError::Codec("fixture token rejected".into())),
            }
        }
    }

    fn recipe() -> TrainingRecipe {
        TrainingRecipe {
            contract: TRAINING_RECIPE_CONTRACT.into(),
            id: "fixture-recipe".into(),
            base_model: BaseModelIdentity {
                id: "open/fixture-0.6b".into(),
                revision: "weights-v1".into(),
                tokenizer_revision: "tokenizer-v1".into(),
            },
            method: FineTuneMethod::Lora,
            objective: TrainingObjective::CandidateLogitCrossEntropy,
            prompt_ir: "turbojev.prompt-ir.v2".into(),
            renderer_profile: "fixture-renderer-v1".into(),
            candidate_encoding: CandidateEncodingPolicy::Dynamic,
            max_context_tokens: 4096,
            adapter: AdapterConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.05,
                target_modules: vec!["q_proj".into()],
            },
            optimization: OptimizationConfig {
                learning_rate: 0.0002,
                epochs: 1,
                micro_batch_size: 1,
                gradient_accumulation_steps: 1,
                warmup_ratio: 0.0,
                weight_decay: 0.0,
                seed: 7,
            },
            debiasing: DebiasingPolicy {
                candidate_order_permutations: 1,
                candidate_code_permutations: 1,
                seed: 9,
            },
            data: DataPolicy {
                languages: vec!["en".into()],
                require_contrastive_pairs: false,
                require_negative_no_match: false,
                require_adversarial_criteria: false,
                require_long_shared_state: false,
            },
            splits: DatasetSplits {
                train: "train-v1".into(),
                validation: "val-v1".into(),
                calibration: "cal-v1".into(),
                benchmark: "bench-v1".into(),
            },
        }
    }

    #[test]
    fn runtime_compatibility_accepts_exact_specialized_contract() {
        let recipe = recipe();
        let manifest = build_specialized_manifest(&recipe, "specialized-v1", "sha256:abc", None)
            .unwrap();
        validate_runtime_compatibility(
            &manifest,
            &SpecializedRuntimeIdentity {
                prompt_ir: recipe.prompt_ir.clone(),
                renderer_profile: recipe.renderer_profile.clone(),
                tokenizer_revision: recipe.base_model.tokenizer_revision.clone(),
                candidate_code_profile: None,
                max_context_tokens: recipe.max_context_tokens,
                calibration_profile: None,
            },
        )
        .unwrap();
    }

    #[test]
    fn boundary_provider_discovers_fixture_single_token_codes() {
        use turbojev_backend::{Candidate, DecisionIntent, DecisionTask};

        let renderer = FixtureRenderer;
        let codec = FixtureCodec;
        let provider = BoundaryVerifiedCodeProvider::new(
            &renderer,
            &codec,
            CodebookDiscoveryConfig {
                candidate_set_attempts: 4,
                ..CodebookDiscoveryConfig::default()
            },
        )
        .unwrap();
        let evidence = serde_json::json!({"message":"fixture"});
        let task = DecisionTask {
            id: "route".into(),
            intent: DecisionIntent::Criterion,
            instruction: Some("route?".into()),
            candidates: vec![
                Candidate { key: "a".into(), description: "A semantic".into() },
                Candidate { key: "b".into(), description: "B semantic".into() },
            ],
        };
        let codes = provider
            .codes_for(CandidateCodeRequest {
                source_example_id: "example",
                question_id: "route",
                evidence: &evidence,
                task: &task,
                required_codes: 2,
            })
            .unwrap();
        assert_eq!(codes, vec!["A".to_owned(), "B".to_owned()]);
        assert_eq!(provider.profile(), "turbojev-dynamic-single-token-v1");
    }
}
