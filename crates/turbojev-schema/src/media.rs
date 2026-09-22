//! Explicit multimodal evidence. The portable layer never opens media sources.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version marker for structured media evidence.
pub const MEDIA_CONTRACT: &str = "turbojev.media.v1";

/// Model-independent input modalities. Decoding belongs to the selected runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Still image decoded by the host.
    Image,
    /// Audio decoded by the host.
    Audio,
    /// Video decoded and sampled by the host.
    Video,
}

/// Inline media bytes, encoded as base64 for all JSON bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaEvidence {
    /// Must be `turbojev.media.v1`.
    pub contract: String,
    /// Additional textual evidence; may be empty for media-only tasks.
    #[serde(default)]
    pub text: String,
    /// Input modality, independent of the file extension.
    pub modality: Modality,
    /// Encoded media bytes. Decoding and codec validation belong to the host.
    pub base64: String,
}

impl MediaEvidence {
    /// Validate evidence and check the selected runtime's declared modalities.
    pub fn check_modalities(state: &Value, supported: &[String]) -> Result<(), String> {
        let media = Self::from_state(state)?;
        let required = match media.as_ref().map(|m| &m.modality) {
            None => "text",
            Some(Modality::Image) => "image",
            Some(Modality::Audio) => "audio",
            Some(Modality::Video) => "video",
        };
        if supported.iter().any(|m| m == required) {
            Ok(())
        } else {
            Err(format!(
                "UNSUPPORTED_MODALITY: runtime does not support {required}"
            ))
        }
    }

    /// Parse explicit media evidence without interpreting ordinary JSON objects.
    pub fn from_state(state: &Value) -> Result<Option<Self>, String> {
        let Some(contract) = state.get("contract").and_then(Value::as_str) else {
            return Ok(None);
        };
        if !contract.starts_with("turbojev.media.") {
            return Ok(None);
        }
        if contract != MEDIA_CONTRACT {
            return Err(format!("unsupported media contract: {contract}"));
        }
        let media: Self = serde_json::from_value(state.clone()).map_err(|e| e.to_string())?;
        if media.base64.is_empty() || media.base64.len() > 32 * 1024 * 1024 {
            return Err("media base64 must contain 1..33554432 characters".into());
        }
        Ok(Some(media))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn explicit_media_is_strict_and_plain_json_is_preserved() {
        assert!(
            MediaEvidence::from_state(&json!({"image":"ordinary evidence"}))
                .unwrap()
                .is_none()
        );
        for modality in ["image", "audio", "video"] {
            let state = json!({"contract": MEDIA_CONTRACT, "modality": modality, "base64":"AA=="});
            assert!(MediaEvidence::from_state(&state).unwrap().is_some());
        }
        for state in [
            json!({"contract":"turbojev.media.v2"}),
            json!({"contract":MEDIA_CONTRACT,"modality":"pdf","base64":"AA=="}),
            json!({"contract":MEDIA_CONTRACT,"modality":"image","base64":""}),
            json!({"contract":MEDIA_CONTRACT,"modality":"image","base64":"AA==","path":"secret"}),
        ] {
            assert!(MediaEvidence::from_state(&state).is_err());
        }
    }
}
