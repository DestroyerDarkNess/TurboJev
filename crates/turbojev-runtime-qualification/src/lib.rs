#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Portable evidence contracts for runtime capability qualification.
//!
//! This crate never probes hardware, opens files, or decides how a runtime is
//! executed. It validates a caller-supplied fingerprint plus reproducible
//! conformance evidence and turns that evidence into capabilities that a host
//! may elect to enable.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const RUNTIME_QUALIFICATION_CONTRACT: &str = "turbojev.runtime-qualification.v1";
pub const LLAMACPP_SHIM_ABI_V5: u32 = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelFingerprint {
    /// Exact model source passed to the native host (normally an absolute path).
    pub source: String,
    /// SHA-256 of the exact model artifact.
    pub sha256: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub architecture: String,
    pub device: String,
    #[serde(default)]
    pub driver: Option<String>,
    #[serde(default)]
    pub os: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeFingerprint {
    pub runtime: String,
    pub runtime_revision: String,
    pub shim_abi: u32,
    pub backend: String,
    #[serde(default)]
    pub backend_revision: Option<String>,
    pub hardware: HardwareFingerprint,
    pub model: ModelFingerprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NumericalTolerance {
    pub absolute: f64,
    pub relative: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct QualificationSummary {
    pub rounds: usize,
    pub requests: usize,
    pub max_abs_logit_delta: f64,
    pub max_relative_logit_delta: f64,
    pub response_mismatches: usize,
    pub isolation_failures: usize,
    pub non_finite_values: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct QualifiedCapabilities {
    pub fused_batch: bool,
    pub cooperative_cancellation: bool,
    pub hard_deadlines: bool,
    #[serde(default)]
    pub max_fused_items: Option<usize>,
    #[serde(default)]
    pub estimated_fused_item_bytes: Option<u64>,
}

impl QualifiedCapabilities {
    pub const fn serial_only() -> Self {
        Self {
            fused_batch: false,
            cooperative_cancellation: false,
            hard_deadlines: false,
            max_fused_items: None,
            estimated_fused_item_bytes: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeQualificationManifest {
    pub contract: String,
    pub id: String,
    pub fingerprint: RuntimeFingerprint,
    pub harness_revision: String,
    pub tolerance: NumericalTolerance,
    pub summary: QualificationSummary,
    pub capabilities: QualifiedCapabilities,
    /// SHA-256 of the deterministic raw conformance observation/report.
    pub evidence_sha256: String,
}

impl RuntimeQualificationManifest {
    pub fn validate(&self) -> Result<(), QualificationError> {
        if self.contract != RUNTIME_QUALIFICATION_CONTRACT {
            return Err(QualificationError::Contract(self.contract.clone()));
        }
        require_non_empty("id", &self.id)?;
        require_non_empty("runtime", &self.fingerprint.runtime)?;
        require_non_empty("runtimeRevision", &self.fingerprint.runtime_revision)?;
        require_non_empty("backend", &self.fingerprint.backend)?;
        require_non_empty("hardware.architecture", &self.fingerprint.hardware.architecture)?;
        require_non_empty("hardware.device", &self.fingerprint.hardware.device)?;
        require_non_empty("model.source", &self.fingerprint.model.source)?;
        require_sha256("model.sha256", &self.fingerprint.model.sha256)?;
        require_sha256("evidenceSha256", &self.evidence_sha256)?;
        require_non_empty("harnessRevision", &self.harness_revision)?;
        if !self.tolerance.absolute.is_finite() || self.tolerance.absolute < 0.0
            || !self.tolerance.relative.is_finite() || self.tolerance.relative < 0.0
        {
            return Err(QualificationError::Invalid("tolerance must be finite and non-negative".into()));
        }
        let s = &self.summary;
        if s.rounds == 0 || s.requests == 0 {
            return Err(QualificationError::Invalid("summary rounds/requests must be non-zero".into()));
        }
        if !s.max_abs_logit_delta.is_finite() || !s.max_relative_logit_delta.is_finite() {
            return Err(QualificationError::Invalid("summary deltas must be finite".into()));
        }
        let clean = s.response_mismatches == 0 && s.isolation_failures == 0 && s.non_finite_values == 0
            && s.max_abs_logit_delta <= self.tolerance.absolute
            && s.max_relative_logit_delta <= self.tolerance.relative;
        if s.passed != clean {
            return Err(QualificationError::Invalid(
                "summary.passed must exactly reflect tolerance/mismatch/isolation results".into(),
            ));
        }
        if self.capabilities.fused_batch && !s.passed {
            return Err(QualificationError::Invalid("fusedBatch cannot be qualified by failed evidence".into()));
        }
        if self.capabilities.fused_batch && self.capabilities.max_fused_items.unwrap_or(0) < 2 {
            return Err(QualificationError::Invalid("fusedBatch requires maxFusedItems >= 2".into()));
        }
        if !self.capabilities.fused_batch
            && (self.capabilities.max_fused_items.is_some() || self.capabilities.estimated_fused_item_bytes.is_some())
        {
            return Err(QualificationError::Invalid("fused limits require fusedBatch=true".into()));
        }
        if matches!(self.capabilities.estimated_fused_item_bytes, Some(0)) {
            return Err(QualificationError::Invalid("estimatedFusedItemBytes must be > 0".into()));
        }
        Ok(())
    }

    /// Return qualified capabilities only when evidence is valid and the exact
    /// current runtime/model/backend/hardware fingerprint matches byte-for-byte.
    pub fn authorize(&self, current: &RuntimeFingerprint) -> Result<QualifiedCapabilities, QualificationError> {
        self.validate()?;
        if &self.fingerprint != current {
            return Err(QualificationError::FingerprintMismatch);
        }
        if !self.summary.passed {
            return Err(QualificationError::EvidenceFailed);
        }
        Ok(self.capabilities)
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), QualificationError> {
    if value.trim().is_empty() { Err(QualificationError::Missing(field)) } else { Ok(()) }
}

fn require_sha256(field: &'static str, value: &str) -> Result<(), QualificationError> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(QualificationError::Invalid(format!("{field} must be lowercase SHA-256 hex")))
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum QualificationError {
    #[error("unsupported qualification contract: {0}")]
    Contract(String),
    #[error("missing qualification field: {0}")]
    Missing(&'static str),
    #[error("invalid qualification: {0}")]
    Invalid(String),
    #[error("qualification fingerprint does not match the loaded target")]
    FingerprintMismatch,
    #[error("qualification evidence did not pass")]
    EvidenceFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> RuntimeQualificationManifest {
        RuntimeQualificationManifest {
            contract: RUNTIME_QUALIFICATION_CONTRACT.into(),
            id: "cpu-smoke".into(),
            fingerprint: RuntimeFingerprint {
                runtime: "llama.cpp".into(), runtime_revision: "deadbeef".into(), shim_abi: 5,
                backend: "cpu".into(), backend_revision: None,
                hardware: HardwareFingerprint { architecture: "x86_64".into(), device: "cpu".into(), driver: None, os: Some("linux".into()) },
                model: ModelFingerprint { source: "/models/tiny.gguf".into(), sha256: "a".repeat(64), revision: None, quantization: Some("Q4_0".into()) },
            },
            harness_revision: "v1".into(),
            tolerance: NumericalTolerance { absolute: 1e-5, relative: 1e-5 },
            summary: QualificationSummary { rounds: 48, requests: 192, max_abs_logit_delta: 0.0, max_relative_logit_delta: 0.0, response_mismatches: 0, isolation_failures: 0, non_finite_values: 0, passed: true },
            capabilities: QualifiedCapabilities { fused_batch: true, cooperative_cancellation: false, hard_deadlines: false, max_fused_items: Some(8), estimated_fused_item_bytes: None },
            evidence_sha256: "b".repeat(64),
        }
    }

    #[test]
    fn exact_fingerprint_authorizes() {
        let q = fixture();
        assert!(q.authorize(&q.fingerprint).unwrap().fused_batch);
    }

    #[test]
    fn mismatch_is_rejected() {
        let q = fixture();
        let mut f = q.fingerprint.clone();
        f.backend = "vulkan".into();
        assert_eq!(q.authorize(&f), Err(QualificationError::FingerprintMismatch));
    }

    #[test]
    fn failed_evidence_cannot_claim_fused() {
        let mut q = fixture();
        q.summary.response_mismatches = 1;
        q.summary.passed = false;
        assert!(q.validate().is_err());
    }
}
