#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable model descriptors.
//!
//! This crate intentionally does not know how to download, mmap, open, allocate,
//! or execute a model. Sources are opaque identifiers interpreted by runtime
//! adapters outside the portable core.

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use thiserror::Error;

/// Versioned runtime contract for decision-specialized checkpoints.
pub const DECISION_MODEL_CONTRACT: &str = "turbojev.decision-model.v1";

/// Opaque model source understood by a runtime adapter.
///
/// Examples may use URI-like identifiers such as `hf://...`, `file://...`,
/// `memory://...`, or application-defined schemes. TurboJev itself does not
/// perform I/O based on this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSource(String);

impl ModelSource {
    /// Create a validated opaque source identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelSpecError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ModelSpecError::EmptySource);
        }
        Ok(Self(value))
    }

    /// Borrow the opaque source string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ModelSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

/// Model metadata used for capability negotiation and reproducibility.
///
/// Architecture and format names are deliberately free-form. Supporting a new
/// model family must not require a TurboJev core release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable application-visible model identifier.
    pub id: String,
    /// Opaque location or model reference resolved by the selected runtime.
    pub source: ModelSource,
    /// Optional architecture name reported by model metadata/runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Optional weight/container format, for example `gguf` or `safetensors`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Optional tokenizer identifier if it is not embedded with the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// Optional prompt profile expected by a specialized checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_profile: Option<String>,
    /// Optional exact revision/checksum/manifest identity for reproducibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Optional decision-specialization contract. Generic models leave this unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_contract: Option<DecisionModelContract>,
}

impl ModelSpec {
    /// Construct the minimum model descriptor.
    pub fn new(id: impl Into<String>, source: ModelSource) -> Result<Self, ModelSpecError> {
        let result = Self {
            id: id.into(),
            source,
            architecture: None,
            format: None,
            tokenizer: None,
            prompt_profile: None,
            revision: None,
            decision_contract: None,
        };
        result.validate()?;
        Ok(result)
    }

    /// Validate model metadata without resolving or opening the source.
    pub fn validate(&self) -> Result<(), ModelSpecError> {
        if self.id.trim().is_empty() {
            return Err(ModelSpecError::EmptyId);
        }
        if let Some(contract) = &self.decision_contract {
            contract.validate()?;
        }
        Ok(())
    }
}

/// Runtime-independent model capability hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// Model/runtime can provide next-token scores.
    pub next_token_logits: bool,
    /// Runtime can score exact multi-token candidate sequences.
    pub sequence_scores: bool,
    /// Runtime can expose the final hidden state and selected output-head rows.
    pub selective_output_projection: bool,
    /// Runtime supports reusable prefix state for this model architecture.
    pub reusable_prefix_state: bool,
}

/// Model descriptor validation failure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelSpecError {
    /// Model identifier is blank.
    #[error("model id must not be empty")]
    EmptyId,
    /// Model source is blank.
    #[error("model source must not be empty")]
    EmptySource,
    /// A specialized-model contract identifier is unsupported.
    #[error("unsupported decision model contract {0:?}")]
    UnsupportedDecisionContract(String),
    /// A specialized-model contract field is blank.
    #[error("decision model contract field {0} must not be empty")]
    EmptyContractField(&'static str),
    /// Context limits must be positive when declared.
    #[error("max_context_tokens must be greater than zero")]
    ZeroContextLimit,
}

/// Candidate encoding policy expected by a checkpoint/profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "profile")]
pub enum CandidateEncodingPolicy {
    /// Discover tokenizer-safe codes dynamically at runtime.
    #[serde(rename = "dynamic")]
    Dynamic,
    /// Use a named encoding profile pinned by a specialized model contract.
    #[serde(rename = "profile")]
    Profile(String),
}

/// Reproducibility contract for a decision-specialized checkpoint.
///
/// Generic frozen language models do not need one. Specialized models can pin
/// the exact prompt/tokenizer/candidate policy used during training so the
/// runtime fails loudly rather than silently changing semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionModelContract {
    /// Contract schema identifier.
    pub contract: String,
    /// Prompt IR contract expected by the checkpoint.
    pub prompt_ir: String,
    /// Model-facing renderer/profile expected by training.
    pub renderer_profile: String,
    /// Candidate code encoding policy.
    pub candidate_encoding: CandidateEncodingPolicy,
    /// Optional tokenizer revision/hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    /// Optional maximum supported input context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<usize>,
    /// Optional calibration profile identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_profile: Option<String>,
}

impl DecisionModelContract {
    /// Validate non-empty contract identifiers and sensible numeric bounds.
    pub fn validate(&self) -> Result<(), ModelSpecError> {
        if self.contract != DECISION_MODEL_CONTRACT {
            return Err(ModelSpecError::UnsupportedDecisionContract(self.contract.clone()));
        }
        for (field, value) in [
            ("contract", self.contract.as_str()),
            ("prompt_ir", self.prompt_ir.as_str()),
            ("renderer_profile", self.renderer_profile.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ModelSpecError::EmptyContractField(field));
            }
        }
        if self.tokenizer_revision.as_ref().is_some_and(|value| value.trim().is_empty()) {
            return Err(ModelSpecError::EmptyContractField("tokenizer_revision"));
        }
        if self.calibration_profile.as_ref().is_some_and(|value| value.trim().is_empty()) {
            return Err(ModelSpecError::EmptyContractField("calibration_profile"));
        }
        if self.max_context_tokens == Some(0) {
            return Err(ModelSpecError::ZeroContextLimit);
        }
        if let CandidateEncodingPolicy::Profile(profile) = &self.candidate_encoding {
            if profile.trim().is_empty() {
                return Err(ModelSpecError::EmptyContractField("candidate_encoding.profile"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_is_opaque_and_platform_neutral() {
        for value in [
            "hf://org/model@revision",
            "memory://already-loaded/model-1",
            "custom+bundle://application/model",
        ] {
            let source = ModelSource::new(value).unwrap();
            assert_eq!(source.as_str(), value);
        }
    }

    #[test]
    fn empty_source_is_rejected_during_deserialization() {
        let error = serde_json::from_str::<ModelSource>(r#""   ""#).unwrap_err();
        assert!(error.to_string().contains("model source"));
    }

    #[test]
    fn model_spec_validates_attached_specialized_contract() {
        let source = ModelSource::new("memory://specialized").unwrap();
        let mut model = ModelSpec::new("specialized", source).unwrap();
        model.decision_contract = Some(DecisionModelContract {
            contract: DECISION_MODEL_CONTRACT.into(),
            prompt_ir: "turbojev.prompt-ir.v2".into(),
            renderer_profile: "fixture-renderer-v1".into(),
            candidate_encoding: CandidateEncodingPolicy::Dynamic,
            tokenizer_revision: Some("tok-v1".into()),
            max_context_tokens: Some(4096),
            calibration_profile: Some("cal-v1".into()),
        });
        model.validate().unwrap();
    }

    #[test]
    fn specialized_contract_is_explicit_and_validated() {
        let contract = DecisionModelContract {
            contract: DECISION_MODEL_CONTRACT.into(),
            prompt_ir: "turbojev.prompt-ir.v2".into(),
            renderer_profile: "example-profile".into(),
            candidate_encoding: CandidateEncodingPolicy::Dynamic,
            tokenizer_revision: None,
            max_context_tokens: Some(2048),
            calibration_profile: None,
        };
        contract.validate().unwrap();
    }
}

