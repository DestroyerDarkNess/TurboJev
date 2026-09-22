#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable contracts for TurboJev decision-specialized checkpoints.
//!
//! This crate deliberately contains no training framework, filesystem, network,
//! GPU, process or OS integration. It describes *what* was trained and the
//! reproducibility/leakage invariants a host-side trainer must obey. Python,
//! Rust-native, MLX, CUDA or another trainer can consume the same recipe.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev_model::{CandidateEncodingPolicy, DecisionModelContract, DECISION_MODEL_CONTRACT};

/// Versioned training-recipe contract.
pub const TRAINING_RECIPE_CONTRACT: &str = "turbojev.specialized-training-recipe.v1";
/// Versioned specialized checkpoint manifest contract.
pub const SPECIALIZED_MODEL_CONTRACT: &str = "turbojev.specialized-model.v1";
/// Versioned generic-vs-specialized evaluation contract.
pub const SPECIALIZATION_EVALUATION_CONTRACT: &str = "turbojev.specialization-evaluation.v1";

/// Exact base-model identity used by specialization training.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseModelIdentity {
    pub id: String,
    pub revision: String,
    pub tokenizer_revision: String,
}

impl BaseModelIdentity {
    fn validate(&self) -> Result<(), SpecializedError> {
        require_text("base_model.id", &self.id)?;
        require_text("base_model.revision", &self.revision)?;
        require_text("base_model.tokenizer_revision", &self.tokenizer_revision)
    }
}

/// TurboJev specializes candidate logits directly instead of teaching JSON generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingObjective {
    CandidateLogitCrossEntropy,
}

/// Parameter-efficient fine-tuning family. Execution belongs to a host trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FineTuneMethod {
    Lora,
    Qlora,
}

/// Framework-neutral LoRA/QLoRA adapter hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub rank: u32,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>,
}

impl AdapterConfig {
    fn validate(&self) -> Result<(), SpecializedError> {
        if self.rank == 0 {
            return Err(SpecializedError::ZeroRank);
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(SpecializedError::InvalidPositive("adapter.alpha"));
        }
        if !self.dropout.is_finite() || !(0.0..1.0).contains(&self.dropout) {
            return Err(SpecializedError::InvalidDropout);
        }
        require_unique_nonempty("adapter.target_modules", &self.target_modules)
    }
}

/// Optimizer-facing scalar hyperparameters kept independent of any framework.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub learning_rate: f64,
    pub epochs: u32,
    pub micro_batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub warmup_ratio: f64,
    pub weight_decay: f64,
    pub seed: u64,
}

impl OptimizationConfig {
    fn validate(&self) -> Result<(), SpecializedError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(SpecializedError::InvalidPositive(
                "optimization.learning_rate",
            ));
        }
        if self.epochs == 0 {
            return Err(SpecializedError::InvalidPositive("optimization.epochs"));
        }
        if self.micro_batch_size == 0 {
            return Err(SpecializedError::InvalidPositive(
                "optimization.micro_batch_size",
            ));
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(SpecializedError::InvalidPositive(
                "optimization.gradient_accumulation_steps",
            ));
        }
        if !self.warmup_ratio.is_finite() || !(0.0..=1.0).contains(&self.warmup_ratio) {
            return Err(SpecializedError::InvalidRatio("optimization.warmup_ratio"));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(SpecializedError::InvalidNonNegative(
                "optimization.weight_decay",
            ));
        }
        Ok(())
    }
}

/// Deterministic anti-position-bias augmentation policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebiasingPolicy {
    /// Number of semantic candidate-order variants requested per source example.
    pub candidate_order_permutations: u32,
    /// Number of candidate-code assignments requested per semantic ordering.
    pub candidate_code_permutations: u32,
    /// Stable seed used by the host-side dataset materializer.
    pub seed: u64,
}

impl DebiasingPolicy {
    fn validate(&self) -> Result<(), SpecializedError> {
        if self.candidate_order_permutations == 0 {
            return Err(SpecializedError::InvalidPositive(
                "debiasing.candidate_order_permutations",
            ));
        }
        if self.candidate_code_permutations == 0 {
            return Err(SpecializedError::InvalidPositive(
                "debiasing.candidate_code_permutations",
            ));
        }
        Ok(())
    }
}

/// Dataset properties required by a specialization experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataPolicy {
    /// BCP-47-like language/profile labels expected in the training corpus.
    pub languages: Vec<String>,
    pub require_contrastive_pairs: bool,
    pub require_negative_no_match: bool,
    pub require_adversarial_criteria: bool,
    pub require_long_shared_state: bool,
}

impl DataPolicy {
    fn validate(&self) -> Result<(), SpecializedError> {
        require_unique_nonempty("data.languages", &self.languages)
    }
}

/// Explicit split identities. These must be pairwise distinct.
///
/// Calibration and benchmark data are never silently borrowed from training or
/// validation; this is enforced as part of recipe validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSplits {
    pub train: String,
    pub validation: String,
    pub calibration: String,
    pub benchmark: String,
}

impl DatasetSplits {
    pub fn validate(&self) -> Result<(), SpecializedError> {
        let fields = [
            ("splits.train", self.train.as_str()),
            ("splits.validation", self.validation.as_str()),
            ("splits.calibration", self.calibration.as_str()),
            ("splits.benchmark", self.benchmark.as_str()),
        ];
        for (name, value) in fields {
            require_text(name, value)?;
        }
        let unique = HashSet::from([
            self.train.as_str(),
            self.validation.as_str(),
            self.calibration.as_str(),
            self.benchmark.as_str(),
        ]);
        if unique.len() != 4 {
            return Err(SpecializedError::SplitLeakage);
        }
        Ok(())
    }
}

/// Reproducible framework-neutral specialization recipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRecipe {
    pub contract: String,
    pub id: String,
    pub base_model: BaseModelIdentity,
    pub method: FineTuneMethod,
    pub objective: TrainingObjective,
    pub prompt_ir: String,
    pub renderer_profile: String,
    pub candidate_encoding: CandidateEncodingPolicy,
    pub max_context_tokens: usize,
    pub adapter: AdapterConfig,
    pub optimization: OptimizationConfig,
    pub debiasing: DebiasingPolicy,
    pub data: DataPolicy,
    pub splits: DatasetSplits,
}

impl TrainingRecipe {
    pub fn validate(&self) -> Result<(), SpecializedError> {
        if self.contract != TRAINING_RECIPE_CONTRACT {
            return Err(SpecializedError::UnsupportedContract(self.contract.clone()));
        }
        require_text("id", &self.id)?;
        self.base_model.validate()?;
        require_text("prompt_ir", &self.prompt_ir)?;
        require_text("renderer_profile", &self.renderer_profile)?;
        validate_candidate_encoding(&self.candidate_encoding)?;
        if self.max_context_tokens == 0 {
            return Err(SpecializedError::ZeroContext);
        }
        self.adapter.validate()?;
        self.optimization.validate()?;
        self.debiasing.validate()?;
        self.data.validate()?;
        self.splits.validate()?;
        Ok(())
    }

    /// Produce the runtime contract that a trained checkpoint must publish.
    pub fn decision_contract(&self, calibration_profile: Option<String>) -> DecisionModelContract {
        DecisionModelContract {
            contract: DECISION_MODEL_CONTRACT.to_owned(),
            prompt_ir: self.prompt_ir.clone(),
            renderer_profile: self.renderer_profile.clone(),
            candidate_encoding: self.candidate_encoding.clone(),
            tokenizer_revision: Some(self.base_model.tokenizer_revision.clone()),
            max_context_tokens: Some(self.max_context_tokens),
            calibration_profile,
        }
    }
}

/// Manifest shipped next to a specialized adapter/merged checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecializedModelManifest {
    pub contract: String,
    pub id: String,
    pub revision: String,
    pub recipe_id: String,
    pub base_model: BaseModelIdentity,
    pub decision_contract: DecisionModelContract,
    pub training_dataset: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_profile: Option<String>,
}

impl SpecializedModelManifest {
    pub fn validate(&self) -> Result<(), SpecializedError> {
        if self.contract != SPECIALIZED_MODEL_CONTRACT {
            return Err(SpecializedError::UnsupportedContract(self.contract.clone()));
        }
        for (field, value) in [
            ("id", self.id.as_str()),
            ("revision", self.revision.as_str()),
            ("recipe_id", self.recipe_id.as_str()),
            ("training_dataset", self.training_dataset.as_str()),
        ] {
            require_text(field, value)?;
        }
        self.base_model.validate()?;
        self.decision_contract
            .validate()
            .map_err(|error| SpecializedError::DecisionContract(error.to_string()))?;
        if self.decision_contract.tokenizer_revision.as_deref()
            != Some(self.base_model.tokenizer_revision.as_str())
        {
            return Err(SpecializedError::TokenizerMismatch);
        }
        if self.calibration_profile.as_deref()
            != self.decision_contract.calibration_profile.as_deref()
        {
            return Err(SpecializedError::CalibrationMismatch);
        }
        Ok(())
    }

    /// Verify that a produced model did not drift away from the recipe that trained it.
    pub fn validate_against_recipe(&self, recipe: &TrainingRecipe) -> Result<(), SpecializedError> {
        self.validate()?;
        recipe.validate()?;
        if self.recipe_id != recipe.id {
            return Err(SpecializedError::RecipeMismatch("recipe_id"));
        }
        if self.base_model != recipe.base_model {
            return Err(SpecializedError::RecipeMismatch("base_model"));
        }
        if self.decision_contract.prompt_ir != recipe.prompt_ir {
            return Err(SpecializedError::RecipeMismatch("prompt_ir"));
        }
        if self.decision_contract.renderer_profile != recipe.renderer_profile {
            return Err(SpecializedError::RecipeMismatch("renderer_profile"));
        }
        if self.decision_contract.candidate_encoding != recipe.candidate_encoding {
            return Err(SpecializedError::RecipeMismatch("candidate_encoding"));
        }
        if self.decision_contract.max_context_tokens != Some(recipe.max_context_tokens) {
            return Err(SpecializedError::RecipeMismatch("max_context_tokens"));
        }
        if self.training_dataset != recipe.splits.train {
            return Err(SpecializedError::RecipeMismatch("training_dataset"));
        }
        Ok(())
    }
}

/// One pair in a controlled generic-vs-specialized comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPair {
    pub generic_model: String,
    pub specialized_model: String,
}

/// Same-benchmark evaluation plan for specialization experiments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecializationEvaluationPlan {
    pub contract: String,
    pub id: String,
    pub benchmark_suite: String,
    pub pairs: Vec<ModelPair>,
    /// Metrics are explicit so reports cannot silently change the target set.
    pub metrics: Vec<String>,
}

impl SpecializationEvaluationPlan {
    pub fn validate(&self) -> Result<(), SpecializedError> {
        if self.contract != SPECIALIZATION_EVALUATION_CONTRACT {
            return Err(SpecializedError::UnsupportedContract(self.contract.clone()));
        }
        require_text("id", &self.id)?;
        require_text("benchmark_suite", &self.benchmark_suite)?;
        if self.pairs.is_empty() {
            return Err(SpecializedError::EmptyPairs);
        }
        for pair in &self.pairs {
            require_text("pairs.generic_model", &pair.generic_model)?;
            require_text("pairs.specialized_model", &pair.specialized_model)?;
            if pair.generic_model == pair.specialized_model {
                return Err(SpecializedError::IdenticalComparisonModel);
            }
        }
        require_unique_nonempty("metrics", &self.metrics)?;
        for required in [
            "accuracy",
            "nll",
            "brier",
            "ece",
            "latency",
            "peak-memory",
            "decisions-per-second",
            "candidate-order-robustness",
            "paraphrase-robustness",
        ] {
            if !self.metrics.iter().any(|metric| metric == required) {
                return Err(SpecializedError::MissingMetric(required));
            }
        }
        Ok(())
    }
}

fn validate_candidate_encoding(policy: &CandidateEncodingPolicy) -> Result<(), SpecializedError> {
    if let CandidateEncodingPolicy::Profile(profile) = policy {
        require_text("candidate_encoding.profile", profile)?;
    }
    Ok(())
}

fn require_text(field: &'static str, value: &str) -> Result<(), SpecializedError> {
    if value.trim().is_empty() {
        Err(SpecializedError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_unique_nonempty(field: &'static str, values: &[String]) -> Result<(), SpecializedError> {
    if values.is_empty() {
        return Err(SpecializedError::EmptyList(field));
    }
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if value.trim().is_empty() {
            return Err(SpecializedError::EmptyField(field));
        }
        if !seen.insert(value.as_str()) {
            return Err(SpecializedError::DuplicateValue(field));
        }
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SpecializedError {
    #[error("unsupported specialized-model contract {0:?}")]
    UnsupportedContract(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must contain at least one value")]
    EmptyList(&'static str),
    #[error("{0} contains duplicate values")]
    DuplicateValue(&'static str),
    #[error("{0} must be finite and greater than zero")]
    InvalidPositive(&'static str),
    #[error("{0} must be finite and non-negative")]
    InvalidNonNegative(&'static str),
    #[error("{0} must be a finite ratio in [0, 1]")]
    InvalidRatio(&'static str),
    #[error("adapter rank must be greater than zero")]
    ZeroRank,
    #[error("adapter dropout must be finite and satisfy 0 <= dropout < 1")]
    InvalidDropout,
    #[error("max_context_tokens must be greater than zero")]
    ZeroContext,
    #[error("train, validation, calibration and benchmark split ids must be pairwise distinct")]
    SplitLeakage,
    #[error("specialized decision contract is invalid: {0}")]
    DecisionContract(String),
    #[error("specialized manifest tokenizer revision does not match its base model")]
    TokenizerMismatch,
    #[error("specialized manifest calibration profile does not match its decision contract")]
    CalibrationMismatch,
    #[error("specialized manifest does not match its training recipe field {0}")]
    RecipeMismatch(&'static str),
    #[error("specialization evaluation must contain at least one model pair")]
    EmptyPairs,
    #[error("generic and specialized model ids must differ")]
    IdenticalComparisonModel,
    #[error("specialization evaluation is missing required metric {0}")]
    MissingMetric(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> TrainingRecipe {
        TrainingRecipe {
            contract: TRAINING_RECIPE_CONTRACT.into(),
            id: "fixture-0.6b-lora-v1".into(),
            base_model: BaseModelIdentity {
                id: "fixture/base-0.6b".into(),
                revision: "abc123".into(),
                tokenizer_revision: "tok123".into(),
            },
            method: FineTuneMethod::Lora,
            objective: TrainingObjective::CandidateLogitCrossEntropy,
            prompt_ir: "turbojev.prompt-ir.v2".into(),
            renderer_profile: "fixture-renderer-v1".into(),
            candidate_encoding: CandidateEncodingPolicy::Dynamic,
            max_context_tokens: 4096,
            adapter: AdapterConfig {
                rank: 16,
                alpha: 32.0,
                dropout: 0.05,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
            },
            optimization: OptimizationConfig {
                learning_rate: 0.0002,
                epochs: 3,
                micro_batch_size: 4,
                gradient_accumulation_steps: 8,
                warmup_ratio: 0.03,
                weight_decay: 0.01,
                seed: 42,
            },
            debiasing: DebiasingPolicy {
                candidate_order_permutations: 4,
                candidate_code_permutations: 4,
                seed: 43,
            },
            data: DataPolicy {
                languages: vec!["en".into(), "es".into()],
                require_contrastive_pairs: true,
                require_negative_no_match: true,
                require_adversarial_criteria: true,
                require_long_shared_state: true,
            },
            splits: DatasetSplits {
                train: "train-v1".into(),
                validation: "validation-v1".into(),
                calibration: "calibration-v1".into(),
                benchmark: "benchmark-v1".into(),
            },
        }
    }

    #[test]
    fn valid_recipe_builds_runtime_contract() {
        let recipe = recipe();
        recipe.validate().unwrap();
        let contract = recipe.decision_contract(Some("cal-v1".into()));
        assert_eq!(contract.tokenizer_revision.as_deref(), Some("tok123"));
        assert_eq!(contract.max_context_tokens, Some(4096));
    }

    #[test]
    fn leakage_between_training_and_calibration_is_rejected() {
        let mut recipe = recipe();
        recipe.splits.calibration = recipe.splits.train.clone();
        assert_eq!(
            recipe.validate().unwrap_err(),
            SpecializedError::SplitLeakage
        );
    }

    #[test]
    fn manifest_must_match_recipe_semantics() {
        let recipe = recipe();
        let manifest = SpecializedModelManifest {
            contract: SPECIALIZED_MODEL_CONTRACT.into(),
            id: "fixture-specialized".into(),
            revision: "specialized123".into(),
            recipe_id: recipe.id.clone(),
            base_model: recipe.base_model.clone(),
            decision_contract: recipe.decision_contract(Some("cal-v1".into())),
            training_dataset: recipe.splits.train.clone(),
            calibration_profile: Some("cal-v1".into()),
        };
        manifest.validate_against_recipe(&recipe).unwrap();
    }
}
