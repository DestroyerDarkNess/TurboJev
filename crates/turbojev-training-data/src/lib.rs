#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable semantic training-data contracts for TurboJev specialization.
//!
//! The crate owns deterministic dataset expansion, candidate-order debiasing,
//! candidate-code assignment debiasing and provenance. It deliberately does
//! **not** own tokenization, model loading, LoRA/QLoRA execution, filesystems or
//! GPU APIs. A host trainer supplies candidate codes that it has verified at
//! the selected checkpoint's exact answer boundary.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use turbojev_backend::{Candidate, DecisionTask};
use turbojev_core::compile_decision_batch;
use turbojev_prompt::{compile_document, PromptDocument, PROMPT_IR_CONTRACT};
use turbojev_schema::{EvaluationRequest, Question};
use turbojev_specialized::{TrainingRecipe, TRAINING_RECIPE_CONTRACT};

/// Versioned semantic source-dataset contract.
pub const SEMANTIC_DATASET_CONTRACT: &str = "turbojev.semantic-training-dataset.v1";
/// Versioned materialized candidate-logit record bundle.
pub const MATERIALIZED_DATASET_CONTRACT: &str = "turbojev.materialized-training-dataset.v1";

/// Ground-truth label for one typed question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum TrainingTarget {
    /// Stable choice key from `Question::Choice.criteria`.
    #[serde(rename = "choice")]
    Choice(String),
    /// Boolean/noul truth value.
    #[serde(rename = "noul")]
    Noul(bool),
    /// Zero-based ordered score level.
    #[serde(rename = "score")]
    Score(usize),
}

impl TrainingTarget {
    fn semantic_key(&self) -> String {
        match self {
            Self::Choice(key) => key.clone(),
            Self::Noul(value) => value.to_string(),
            Self::Score(index) => index.to_string(),
        }
    }
}

/// Explicit data-quality/provenance markers used by a recipe's data policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TrainingExampleProperties {
    /// Examples sharing this id form a controlled contrastive group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrastive_group: Option<String>,
    /// This example intentionally teaches a none/no-match decision.
    #[serde(default)]
    pub negative_no_match: bool,
    /// Candidate criteria are intentionally close/confusable.
    #[serde(default)]
    pub adversarial_criteria: bool,
    /// Shared evidence is intentionally long enough to test state retention.
    #[serde(default)]
    pub long_shared_state: bool,
}

/// One semantic source example before candidate/order augmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticTrainingExample {
    pub id: String,
    /// Language/profile label. TurboJev preserves source text and never translates it.
    pub language: String,
    pub request: EvaluationRequest,
    /// Exactly one target per request question, keyed by question id.
    pub targets: IndexMap<String, TrainingTarget>,
    /// Relative source-example weight. Materialization conserves this weight per decision.
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default)]
    pub properties: TrainingExampleProperties,
}

fn default_weight() -> f64 {
    1.0
}

/// Versioned semantic dataset consumed by the deterministic materializer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticTrainingDataset {
    pub contract: String,
    /// Must equal the training split id pinned by the recipe.
    pub id: String,
    pub examples: Vec<SemanticTrainingExample>,
}

impl SemanticTrainingDataset {
    /// Validate schema-level and typed-target invariants independent of a recipe.
    pub fn validate(&self) -> Result<(), TrainingDataError> {
        if self.contract != SEMANTIC_DATASET_CONTRACT {
            return Err(TrainingDataError::UnsupportedContract(
                self.contract.clone(),
            ));
        }
        require_text("dataset.id", &self.id)?;
        if self.examples.is_empty() {
            return Err(TrainingDataError::EmptyDataset);
        }

        let mut ids = HashSet::with_capacity(self.examples.len());
        for example in &self.examples {
            require_text("example.id", &example.id)?;
            require_text("example.language", &example.language)?;
            if !ids.insert(example.id.as_str()) {
                return Err(TrainingDataError::DuplicateExampleId(example.id.clone()));
            }
            if !example.weight.is_finite() || example.weight <= 0.0 {
                return Err(TrainingDataError::InvalidWeight(example.id.clone()));
            }
            if let Some(group) = &example.properties.contrastive_group {
                require_text("example.properties.contrastive_group", group)?;
            }

            example
                .request
                .validate()
                .map_err(|error| TrainingDataError::InvalidRequest {
                    example: example.id.clone(),
                    message: error.to_string(),
                })?;

            if example.targets.len() != example.request.questions.len() {
                return Err(TrainingDataError::TargetCountMismatch {
                    example: example.id.clone(),
                    questions: example.request.questions.len(),
                    targets: example.targets.len(),
                });
            }
            for (question_id, question) in &example.request.questions {
                let target = example.targets.get(question_id).ok_or_else(|| {
                    TrainingDataError::MissingTarget {
                        example: example.id.clone(),
                        question: question_id.clone(),
                    }
                })?;
                validate_target(&example.id, question_id, question, target)?;
            }
            for question_id in example.targets.keys() {
                if !example.request.questions.contains_key(question_id) {
                    return Err(TrainingDataError::UnknownTarget {
                        example: example.id.clone(),
                        question: question_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate that this source dataset satisfies the recipe's explicit data policy.
    pub fn validate_against_recipe(
        &self,
        recipe: &TrainingRecipe,
    ) -> Result<(), TrainingDataError> {
        self.validate()?;
        recipe
            .validate()
            .map_err(|error| TrainingDataError::InvalidRecipe(error.to_string()))?;
        if recipe.contract != TRAINING_RECIPE_CONTRACT {
            return Err(TrainingDataError::InvalidRecipe(
                "unexpected training recipe contract".to_owned(),
            ));
        }
        if self.id != recipe.splits.train {
            return Err(TrainingDataError::TrainingSplitMismatch {
                dataset: self.id.clone(),
                recipe: recipe.splits.train.clone(),
            });
        }

        let expected_languages: HashSet<&str> =
            recipe.data.languages.iter().map(String::as_str).collect();
        let actual_languages: HashSet<&str> = self
            .examples
            .iter()
            .map(|example| example.language.as_str())
            .collect();

        for language in &expected_languages {
            if !actual_languages.contains(language) {
                return Err(TrainingDataError::MissingLanguage((*language).to_owned()));
            }
        }
        for language in &actual_languages {
            if !expected_languages.contains(language) {
                return Err(TrainingDataError::UnexpectedLanguage(
                    (*language).to_owned(),
                ));
            }
        }

        if recipe.data.require_contrastive_pairs && !has_contrastive_pair(&self.examples) {
            return Err(TrainingDataError::MissingRequiredProperty(
                "contrastive-pair",
            ));
        }
        if recipe.data.require_negative_no_match
            && !self
                .examples
                .iter()
                .any(|example| example.properties.negative_no_match)
        {
            return Err(TrainingDataError::MissingRequiredProperty(
                "negative-no-match",
            ));
        }
        if recipe.data.require_adversarial_criteria
            && !self
                .examples
                .iter()
                .any(|example| example.properties.adversarial_criteria)
        {
            return Err(TrainingDataError::MissingRequiredProperty(
                "adversarial-criteria",
            ));
        }
        if recipe.data.require_long_shared_state
            && !self
                .examples
                .iter()
                .any(|example| example.properties.long_shared_state)
        {
            return Err(TrainingDataError::MissingRequiredProperty(
                "long-shared-state",
            ));
        }
        Ok(())
    }
}

fn has_contrastive_pair(examples: &[SemanticTrainingExample]) -> bool {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for example in examples {
        if let Some(group) = example.properties.contrastive_group.as_deref() {
            *counts.entry(group).or_default() += 1;
        }
    }
    counts.values().any(|count| *count >= 2)
}

fn validate_target(
    example_id: &str,
    question_id: &str,
    question: &Question,
    target: &TrainingTarget,
) -> Result<(), TrainingDataError> {
    match (question, target) {
        (Question::Choice { criteria, .. }, TrainingTarget::Choice(key)) => {
            if criteria.contains_key(key) {
                Ok(())
            } else {
                Err(TrainingDataError::UnknownChoiceTarget {
                    example: example_id.to_owned(),
                    question: question_id.to_owned(),
                    target: key.clone(),
                })
            }
        }
        (Question::Noul { .. }, TrainingTarget::Noul(_)) => Ok(()),
        (Question::Score { criteria, .. }, TrainingTarget::Score(index)) => {
            if *index < criteria.len() {
                Ok(())
            } else {
                Err(TrainingDataError::ScoreTargetOutOfRange {
                    example: example_id.to_owned(),
                    question: question_id.to_owned(),
                    target: *index,
                    levels: criteria.len(),
                })
            }
        }
        _ => Err(TrainingDataError::TargetTypeMismatch {
            example: example_id.to_owned(),
            question: question_id.to_owned(),
        }),
    }
}

/// Context passed to a model/tokenizer-aware candidate-code provider.
pub struct CandidateCodeRequest<'a> {
    pub source_example_id: &'a str,
    pub question_id: &'a str,
    pub evidence: &'a Value,
    pub task: &'a DecisionTask,
    pub required_codes: usize,
}

/// Supplies candidate codes already verified for the selected checkpoint boundary.
///
/// A production trainer should implement this with the same renderer/tokenizer
/// boundary rules used by inference. The materializer then permutes the verified
/// codes without knowing anything about model families or token ids.
pub trait CandidateCodeProvider {
    /// Stable provider/profile id persisted into materialized provenance.
    fn profile(&self) -> &str;

    /// Return exactly `request.required_codes` unique, non-empty codes.
    fn codes_for(&self, request: CandidateCodeRequest<'_>) -> Result<Vec<String>, String>;
}

/// Provenance for one deterministic augmentation expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AugmentationProvenance {
    pub order_variant: usize,
    pub order_variants: usize,
    pub code_variant: usize,
    pub code_variants: usize,
}

/// One candidate-logit training record ready for model-specific rendering/tokenization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateLogitTrainingRecord {
    pub id: String,
    pub source_example_id: String,
    pub question_id: String,
    pub language: String,
    /// Semantic IR with the concrete candidate order and code assignment.
    pub document: PromptDocument,
    /// Stable application-level key of the correct candidate.
    pub target_key: String,
    /// Correct row in `document.options` after order permutation.
    pub target_index: usize,
    /// Concrete candidate code whose logit is the CE target.
    pub target_code: String,
    /// Augmented records share the original source-decision weight.
    pub weight: f64,
    pub augmentation: AugmentationProvenance,
}

/// Aggregate audit counters for a materialized dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterializationStats {
    pub source_examples: usize,
    pub source_decisions: usize,
    pub records: usize,
    pub total_record_weight: f64,
}

/// Deterministically expanded dataset consumed by a host-side LoRA/QLoRA trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterializedTrainingDataset {
    pub contract: String,
    pub id: String,
    pub source_dataset: String,
    pub recipe_id: String,
    pub prompt_ir: String,
    pub renderer_profile: String,
    pub candidate_code_profile: String,
    pub seed: u64,
    pub records: Vec<CandidateLogitTrainingRecord>,
    pub stats: MaterializationStats,
}

impl MaterializedTrainingDataset {
    /// Validate internal provenance, target positions and record-weight accounting.
    pub fn validate(&self) -> Result<(), TrainingDataError> {
        if self.contract != MATERIALIZED_DATASET_CONTRACT {
            return Err(TrainingDataError::UnsupportedContract(
                self.contract.clone(),
            ));
        }
        for (field, value) in [
            ("materialized.id", self.id.as_str()),
            ("materialized.source_dataset", self.source_dataset.as_str()),
            ("materialized.recipe_id", self.recipe_id.as_str()),
            ("materialized.prompt_ir", self.prompt_ir.as_str()),
            (
                "materialized.renderer_profile",
                self.renderer_profile.as_str(),
            ),
            (
                "materialized.candidate_code_profile",
                self.candidate_code_profile.as_str(),
            ),
        ] {
            require_text(field, value)?;
        }
        if self.prompt_ir != PROMPT_IR_CONTRACT {
            return Err(TrainingDataError::PromptIrMismatch(self.prompt_ir.clone()));
        }
        if self.records.is_empty() {
            return Err(TrainingDataError::EmptyMaterializedDataset);
        }
        if self.stats.records != self.records.len() {
            return Err(TrainingDataError::StatsMismatch("records"));
        }

        let mut ids = HashSet::with_capacity(self.records.len());
        let mut total_weight = 0.0_f64;
        for record in &self.records {
            if !ids.insert(record.id.as_str()) {
                return Err(TrainingDataError::DuplicateRecordId(record.id.clone()));
            }
            if !record.weight.is_finite() || record.weight <= 0.0 {
                return Err(TrainingDataError::InvalidRecordWeight(record.id.clone()));
            }
            if record.document.contract != self.prompt_ir {
                return Err(TrainingDataError::PromptIrMismatch(
                    record.document.contract.clone(),
                ));
            }
            if record.target_index >= record.document.options.len() {
                return Err(TrainingDataError::MaterializedTargetOutOfRange(
                    record.id.clone(),
                ));
            }
            let option = &record.document.options[record.target_index];
            if option.key != record.target_key || option.code != record.target_code {
                return Err(TrainingDataError::MaterializedTargetMismatch(
                    record.id.clone(),
                ));
            }
            let mut keys = HashSet::with_capacity(record.document.options.len());
            let mut codes = HashSet::with_capacity(record.document.options.len());
            for option in &record.document.options {
                if !keys.insert(option.key.as_str()) {
                    return Err(TrainingDataError::DuplicateMaterializedKey(
                        record.id.clone(),
                    ));
                }
                if option.code.trim().is_empty() || !codes.insert(option.code.as_str()) {
                    return Err(TrainingDataError::InvalidMaterializedCodes(
                        record.id.clone(),
                    ));
                }
            }
            total_weight += record.weight;
        }
        if (total_weight - self.stats.total_record_weight).abs() > 1e-9 {
            return Err(TrainingDataError::StatsMismatch("total_record_weight"));
        }
        Ok(())
    }
}

/// Expand semantic source data into deterministic candidate-logit training records.
pub fn materialize_training_dataset<P: CandidateCodeProvider>(
    dataset: &SemanticTrainingDataset,
    recipe: &TrainingRecipe,
    code_provider: &P,
    output_id: impl Into<String>,
) -> Result<MaterializedTrainingDataset, TrainingDataError> {
    dataset.validate_against_recipe(recipe)?;
    require_text("candidate_code_provider.profile", code_provider.profile())?;
    if recipe.prompt_ir != PROMPT_IR_CONTRACT {
        return Err(TrainingDataError::PromptIrMismatch(
            recipe.prompt_ir.clone(),
        ));
    }

    let output_id = output_id.into();
    require_text("materialized.id", &output_id)?;
    let mut records = Vec::new();
    let mut source_decisions = 0_usize;
    let mut expected_total_weight = 0.0_f64;

    for example in &dataset.examples {
        let batch = compile_decision_batch(&example.request);
        for task in &batch.tasks {
            source_decisions += 1;
            expected_total_weight += example.weight;
            let target =
                example
                    .targets
                    .get(&task.id)
                    .ok_or_else(|| TrainingDataError::MissingTarget {
                        example: example.id.clone(),
                        question: task.id.clone(),
                    })?;
            let target_key = target.semantic_key();

            let order_scope = format!("{}/{}/order", example.id, task.id);
            let order_variants = deterministic_permutations(
                task.candidates.len(),
                recipe.debiasing.candidate_order_permutations as usize,
                recipe.debiasing.seed,
                &order_scope,
            );

            for (order_variant, order) in order_variants.iter().enumerate() {
                let ordered_task = reorder_task(task, order);
                let base_codes = code_provider
                    .codes_for(CandidateCodeRequest {
                        source_example_id: &example.id,
                        question_id: &task.id,
                        evidence: &batch.state,
                        task: &ordered_task,
                        required_codes: ordered_task.candidates.len(),
                    })
                    .map_err(|message| TrainingDataError::CodeProvider {
                        example: example.id.clone(),
                        question: task.id.clone(),
                        message,
                    })?;
                validate_codes(
                    &example.id,
                    &task.id,
                    &base_codes,
                    ordered_task.candidates.len(),
                )?;

                let code_scope = format!("{}/{}/order-{order_variant}/code", example.id, task.id);
                let code_permutations = deterministic_permutations(
                    base_codes.len(),
                    recipe.debiasing.candidate_code_permutations as usize,
                    recipe.debiasing.seed,
                    &code_scope,
                );
                let variant_count = order_variants.len() * code_permutations.len();
                let record_weight = example.weight / variant_count as f64;

                for (code_variant, code_order) in code_permutations.iter().enumerate() {
                    let codes = reorder_strings(&base_codes, code_order);
                    let target_index = ordered_task
                        .candidates
                        .iter()
                        .position(|candidate| candidate.key == target_key)
                        .ok_or_else(|| TrainingDataError::TargetMissingAfterPermutation {
                            example: example.id.clone(),
                            question: task.id.clone(),
                        })?;
                    let document = compile_document(&batch.state, &ordered_task, &codes)
                        .map_err(|error| TrainingDataError::Prompt(error.to_string()))?;
                    let target_code = codes[target_index].clone();
                    records.push(CandidateLogitTrainingRecord {
                        id: format!(
                            "{}::{}::o{}::c{}",
                            example.id, task.id, order_variant, code_variant
                        ),
                        source_example_id: example.id.clone(),
                        question_id: task.id.clone(),
                        language: example.language.clone(),
                        document,
                        target_key: target_key.clone(),
                        target_index,
                        target_code,
                        weight: record_weight,
                        augmentation: AugmentationProvenance {
                            order_variant,
                            order_variants: order_variants.len(),
                            code_variant,
                            code_variants: code_permutations.len(),
                        },
                    });
                }
            }
        }
    }

    let total_record_weight: f64 = records.iter().map(|record| record.weight).sum();
    if (total_record_weight - expected_total_weight).abs() > 1e-9 {
        return Err(TrainingDataError::WeightConservation {
            expected: expected_total_weight,
            actual: total_record_weight,
        });
    }

    let materialized = MaterializedTrainingDataset {
        contract: MATERIALIZED_DATASET_CONTRACT.to_owned(),
        id: output_id,
        source_dataset: dataset.id.clone(),
        recipe_id: recipe.id.clone(),
        prompt_ir: recipe.prompt_ir.clone(),
        renderer_profile: recipe.renderer_profile.clone(),
        candidate_code_profile: code_provider.profile().to_owned(),
        seed: recipe.debiasing.seed,
        stats: MaterializationStats {
            source_examples: dataset.examples.len(),
            source_decisions,
            records: records.len(),
            total_record_weight,
        },
        records,
    };
    materialized.validate()?;
    Ok(materialized)
}

fn reorder_task(task: &DecisionTask, order: &[usize]) -> DecisionTask {
    DecisionTask {
        id: task.id.clone(),
        intent: task.intent,
        instruction: task.instruction.clone(),
        candidates: order
            .iter()
            .map(|index| task.candidates[*index].clone())
            .collect::<Vec<Candidate>>(),
    }
}

fn reorder_strings(values: &[String], order: &[usize]) -> Vec<String> {
    order.iter().map(|index| values[*index].clone()).collect()
}

fn validate_codes(
    example: &str,
    question: &str,
    codes: &[String],
    expected: usize,
) -> Result<(), TrainingDataError> {
    if codes.len() != expected {
        return Err(TrainingDataError::CandidateCodeCount {
            example: example.to_owned(),
            question: question.to_owned(),
            expected,
            actual: codes.len(),
        });
    }
    let mut unique = HashSet::with_capacity(codes.len());
    for code in codes {
        if code.trim().is_empty() || !unique.insert(code.as_str()) {
            return Err(TrainingDataError::InvalidCandidateCodes {
                example: example.to_owned(),
                question: question.to_owned(),
            });
        }
    }
    Ok(())
}

/// Generate up to `requested` unique permutations using a stable seed.
///
/// The effective count is capped at `n!`, so a boolean decision requesting four
/// variants correctly yields its two possible orderings instead of duplicate rows.
fn deterministic_permutations(
    n: usize,
    requested: usize,
    seed: u64,
    scope: &str,
) -> Vec<Vec<usize>> {
    debug_assert!(n >= 2);
    debug_assert!(requested >= 1);
    let target = permutation_capacity(n, requested);
    let identity: Vec<usize> = (0..n).collect();
    let mut result = vec![identity];
    if target == 1 {
        return result;
    }

    // Factoradic unranking gives a one-to-one mapping from integer ranks to
    // permutations. We keep rank 0 as the explicit identity baseline and pick
    // a seeded non-zero starting rank, then walk consecutive ranks modulo the
    // non-identity permutation space. This guarantees uniqueness without a
    // random retry loop and is stable across platforms/Rust versions.
    let capacity = factorial_saturated(n);
    let non_identity = capacity - 1;
    let seed_rank =
        1 + (mix64(seed ^ stable_hash(scope.as_bytes())) % non_identity as u64) as usize;
    for offset in 0..target - 1 {
        let rank = 1 + (((seed_rank - 1) as u128 + offset as u128) % non_identity as u128) as usize;
        result.push(unrank_permutation(n, rank));
    }
    result
}

fn permutation_capacity(n: usize, requested: usize) -> usize {
    factorial_saturated(n).min(requested)
}

fn factorial_saturated(n: usize) -> usize {
    (2..=n).fold(1_usize, |acc, value| acc.saturating_mul(value))
}

fn unrank_permutation(n: usize, mut rank: usize) -> Vec<usize> {
    let mut available: Vec<usize> = (0..n).collect();
    let mut permutation = Vec::with_capacity(n);
    for remaining in (1..=n).rev() {
        let block = factorial_saturated(remaining - 1);
        let index = if block == usize::MAX { 0 } else { rank / block };
        if block != usize::MAX {
            rank %= block;
        }
        permutation.push(available.remove(index));
    }
    permutation
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

const fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn require_text(field: &'static str, value: &str) -> Result<(), TrainingDataError> {
    if value.trim().is_empty() {
        Err(TrainingDataError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TrainingDataError {
    #[error("unsupported training-data contract {0:?}")]
    UnsupportedContract(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("semantic training dataset must contain at least one example")]
    EmptyDataset,
    #[error("materialized training dataset must contain at least one record")]
    EmptyMaterializedDataset,
    #[error("duplicate semantic example id {0:?}")]
    DuplicateExampleId(String),
    #[error("duplicate materialized record id {0:?}")]
    DuplicateRecordId(String),
    #[error("example {0:?} weight must be finite and greater than zero")]
    InvalidWeight(String),
    #[error("materialized record {0:?} weight must be finite and greater than zero")]
    InvalidRecordWeight(String),
    #[error("example {example:?} has invalid request: {message}")]
    InvalidRequest { example: String, message: String },
    #[error("example {example:?} has {questions} questions but {targets} targets")]
    TargetCountMismatch {
        example: String,
        questions: usize,
        targets: usize,
    },
    #[error("example {example:?} is missing target for question {question:?}")]
    MissingTarget { example: String, question: String },
    #[error("example {example:?} has target for unknown question {question:?}")]
    UnknownTarget { example: String, question: String },
    #[error("example {example:?} question {question:?} target type does not match question type")]
    TargetTypeMismatch { example: String, question: String },
    #[error(
        "example {example:?} question {question:?} references unknown choice target {target:?}"
    )]
    UnknownChoiceTarget {
        example: String,
        question: String,
        target: String,
    },
    #[error(
        "example {example:?} question {question:?} score target {target} exceeds {levels} levels"
    )]
    ScoreTargetOutOfRange {
        example: String,
        question: String,
        target: usize,
        levels: usize,
    },
    #[error("training recipe is invalid: {0}")]
    InvalidRecipe(String),
    #[error("semantic dataset id {dataset:?} does not match recipe training split {recipe:?}")]
    TrainingSplitMismatch { dataset: String, recipe: String },
    #[error("semantic dataset is missing recipe language {0:?}")]
    MissingLanguage(String),
    #[error("semantic dataset contains language {0:?} not declared by recipe")]
    UnexpectedLanguage(String),
    #[error("semantic dataset is missing required data property {0}")]
    MissingRequiredProperty(&'static str),
    #[error("candidate-code provider failed for {example:?}/{question:?}: {message}")]
    CodeProvider {
        example: String,
        question: String,
        message: String,
    },
    #[error("candidate-code count mismatch for {example:?}/{question:?}: expected {expected}, got {actual}")]
    CandidateCodeCount {
        example: String,
        question: String,
        expected: usize,
        actual: usize,
    },
    #[error("candidate codes for {example:?}/{question:?} must be non-empty and unique")]
    InvalidCandidateCodes { example: String, question: String },
    #[error("correct target disappeared after permutation for {example:?}/{question:?}")]
    TargetMissingAfterPermutation { example: String, question: String },
    #[error("prompt materialization failed: {0}")]
    Prompt(String),
    #[error("prompt IR {0:?} is incompatible with the current materializer")]
    PromptIrMismatch(String),
    #[error("materialized record {0:?} has a target index outside its options")]
    MaterializedTargetOutOfRange(String),
    #[error("materialized record {0:?} target key/code does not match target_index")]
    MaterializedTargetMismatch(String),
    #[error("materialized record {0:?} contains duplicate semantic candidate keys")]
    DuplicateMaterializedKey(String),
    #[error("materialized record {0:?} candidate codes must be non-empty and unique")]
    InvalidMaterializedCodes(String),
    #[error("materialized statistics do not match field {0}")]
    StatsMismatch(&'static str),
    #[error("materialization did not conserve source-decision weight: expected {expected}, got {actual}")]
    WeightConservation { expected: f64, actual: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::indexmap;
    use turbojev_model::CandidateEncodingPolicy;
    use turbojev_specialized::{
        AdapterConfig, BaseModelIdentity, DataPolicy, DatasetSplits, DebiasingPolicy,
        FineTuneMethod, OptimizationConfig, TrainingObjective,
    };

    struct FixtureCodes;

    impl CandidateCodeProvider for FixtureCodes {
        fn profile(&self) -> &str {
            "fixture-ascii-v1"
        }

        fn codes_for(&self, request: CandidateCodeRequest<'_>) -> Result<Vec<String>, String> {
            const CODES: &[&str] = &["A", "B", "C", "D", "E", "F", "G", "H"];
            if request.required_codes > CODES.len() {
                return Err("fixture code pool exhausted".to_owned());
            }
            Ok(CODES[..request.required_codes]
                .iter()
                .map(|code| (*code).to_owned())
                .collect())
        }
    }

    fn recipe() -> TrainingRecipe {
        TrainingRecipe {
            contract: TRAINING_RECIPE_CONTRACT.to_owned(),
            id: "fixture-recipe".into(),
            base_model: BaseModelIdentity {
                id: "fixture/model".into(),
                revision: "weights-v1".into(),
                tokenizer_revision: "tok-v1".into(),
            },
            method: FineTuneMethod::Lora,
            objective: TrainingObjective::CandidateLogitCrossEntropy,
            prompt_ir: PROMPT_IR_CONTRACT.into(),
            renderer_profile: "fixture-renderer-v1".into(),
            candidate_encoding: CandidateEncodingPolicy::Dynamic,
            max_context_tokens: 4096,
            adapter: AdapterConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.05,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
            },
            optimization: OptimizationConfig {
                learning_rate: 2e-4,
                epochs: 2,
                micro_batch_size: 2,
                gradient_accumulation_steps: 4,
                warmup_ratio: 0.05,
                weight_decay: 0.0,
                seed: 7,
            },
            debiasing: DebiasingPolicy {
                candidate_order_permutations: 4,
                candidate_code_permutations: 4,
                seed: 2701,
            },
            data: DataPolicy {
                languages: vec!["en".into()],
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

    fn dataset() -> SemanticTrainingDataset {
        let make = |id: &str, message: &str, target: &str| SemanticTrainingExample {
            id: id.into(),
            language: "en".into(),
            request: EvaluationRequest {
                model: None,
                state: serde_json::json!({"message": message, "history": ["old", "context"]}),
                questions: indexmap! {
                    "route".into() => Question::Choice {
                        instructions: "Which queue?".into(),
                        criteria: indexmap! {
                            "billing".into() => "Payment issue".into(),
                            "shipping".into() => "Delivery issue".into(),
                            "other".into() => "Neither".into(),
                        },
                    },
                    "urgent".into() => Question::Noul {
                        instructions: "Is urgent?".into(),
                    },
                },
            },
            targets: indexmap! {
                "route".into() => TrainingTarget::Choice(target.into()),
                "urgent".into() => TrainingTarget::Noul(false),
            },
            weight: 1.0,
            properties: TrainingExampleProperties {
                contrastive_group: Some("shipping-fact".into()),
                negative_no_match: target == "other",
                adversarial_criteria: true,
                long_shared_state: true,
            },
        };
        SemanticTrainingDataset {
            contract: SEMANTIC_DATASET_CONTRACT.into(),
            id: "train-v1".into(),
            examples: vec![
                make("a", "package arrived", "other"),
                make("b", "package never arrived", "shipping"),
            ],
        }
    }

    #[test]
    fn semantic_dataset_checks_recipe_policy_and_targets() {
        dataset().validate_against_recipe(&recipe()).unwrap();
    }

    #[test]
    fn materialization_is_deterministic_and_conserves_weight() {
        let first =
            materialize_training_dataset(&dataset(), &recipe(), &FixtureCodes, "mat-v1").unwrap();
        let second =
            materialize_training_dataset(&dataset(), &recipe(), &FixtureCodes, "mat-v1").unwrap();
        assert_eq!(first, second);
        // 2 examples * (choice: 4x4 + boolean: 2x2) = 40 records.
        assert_eq!(first.records.len(), 40);
        assert_eq!(first.stats.source_decisions, 4);
        assert!((first.stats.total_record_weight - 4.0).abs() < 1e-9);
        first.validate().unwrap();
    }

    #[test]
    fn every_target_tracks_semantics_after_both_permutations() {
        let output =
            materialize_training_dataset(&dataset(), &recipe(), &FixtureCodes, "mat-v1").unwrap();
        for record in output.records {
            let option = &record.document.options[record.target_index];
            assert_eq!(option.key, record.target_key);
            assert_eq!(option.code, record.target_code);
        }
    }

    #[test]
    fn boolean_permutations_are_capped_at_factorial_without_duplicates() {
        let variants = deterministic_permutations(2, 99, 42, "boolean");
        assert_eq!(variants.len(), 2);
        assert_ne!(variants[0], variants[1]);
    }
}
