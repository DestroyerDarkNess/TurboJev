#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Language- and chat-template-neutral prompt intermediate representation.
//!
//! The portable core compiles *meaning*. A runtime/model adapter is responsible
//! for rendering that meaning into the instruction/chat format expected by a
//! particular checkpoint. TurboJev therefore does not need a mandatory English
//! prompt or a Qwen/Llama-specific chat template.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use turbojev_backend::{DecisionIntent, DecisionTask};

/// Stable semantic prompt IR contract.
pub const PROMPT_IR_CONTRACT: &str = "turbojev.prompt-ir.v2";

/// Semantic criterion independent of human-language rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "text")]
pub enum PromptCriterion {
    /// Caller supplied explicit criterion text. It is preserved verbatim.
    #[serde(rename = "criterion")]
    UserText(String),
    /// Generic best-match classification intent used by the convenience API.
    /// The model adapter decides how to express this intent for the checkpoint.
    #[serde(rename = "best_match")]
    BestMatch,
}

/// One opaque candidate code mapped to application semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptOption {
    /// Opaque output code, for example `A` or another tokenizer-safe sequence.
    pub code: String,
    /// Stable application-level output key.
    pub key: String,
    /// User-supplied description. TurboJev does not translate it.
    pub description: String,
}

/// Semantic prompt document independent of human language and model template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptDocument {
    /// Shared evidence/state exactly as supplied by the caller.
    pub evidence: Value,
    /// Semantic criterion.
    pub criterion: PromptCriterion,
    /// Opaque codes mapped to allowed meanings.
    pub options: Vec<PromptOption>,
    /// Prompt IR version.
    pub contract: String,
}

/// Build the portable semantic prompt IR for one task.
pub fn compile_document(
    state: &Value,
    task: &DecisionTask,
    codes: &[String],
) -> Result<PromptDocument, PromptError> {
    if task.candidates.len() != codes.len() {
        return Err(PromptError::CandidateCodeCount {
            candidates: task.candidates.len(),
            codes: codes.len(),
        });
    }

    let criterion = match task.intent {
        DecisionIntent::Criterion => {
            let text = task
                .instruction
                .as_ref()
                .ok_or(PromptError::MissingCriterion)?;
            PromptCriterion::UserText(text.clone())
        }
        DecisionIntent::BestMatch => PromptCriterion::BestMatch,
    };

    let options = task
        .candidates
        .iter()
        .zip(codes)
        .map(|(candidate, code)| PromptOption {
            code: code.clone(),
            key: candidate.key.clone(),
            description: candidate.description.clone(),
        })
        .collect();

    Ok(PromptDocument {
        evidence: state.clone(),
        criterion,
        options,
        contract: PROMPT_IR_CONTRACT.to_owned(),
    })
}

/// Rendered messages produced by a model-specific prompt adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedMessages {
    /// Optional system/instruction message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Main user/input message.
    pub user: String,
    /// Adapter profile identifier used for reproducibility.
    pub profile: String,
}

/// Model-specific prompt renderer.
///
/// Implementations may use any human language or instruction format appropriate
/// for the selected checkpoint. TurboJev imposes no global language.
pub trait PromptRenderer {
    /// Stable profile identifier.
    fn profile(&self) -> &str;

    /// Render semantic IR into model-facing messages.
    fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError>;
}

/// Prompt construction/rendering failure.
#[derive(Debug, Error)]
pub enum PromptError {
    /// Number of backend-selected codes did not match candidates.
    #[error("candidate/code count mismatch: {candidates} candidates, {codes} codes")]
    CandidateCodeCount { candidates: usize, codes: usize },
    /// A criterion task had no caller-supplied instruction.
    #[error("criterion decision task is missing instruction text")]
    MissingCriterion,
    /// Runtime prompt adapter failed.
    #[error("prompt renderer failed: {0}")]
    Renderer(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbojev_backend::Candidate;

    #[test]
    fn ir_preserves_non_english_text_without_translation() {
        let task = DecisionTask {
            id: "ruta".into(),
            intent: DecisionIntent::Criterion,
            instruction: Some("¿Qué equipo debe atender este caso?".into()),
            candidates: vec![
                Candidate {
                    key: "facturacion".into(),
                    description: "Problemas de pago".into(),
                },
                Candidate {
                    key: "envio".into(),
                    description: "Problemas de entrega".into(),
                },
            ],
        };
        let document = compile_document(
            &serde_json::json!({"mensaje":"Mi paquete no llegó"}),
            &task,
            &["A".into(), "B".into()],
        )
        .unwrap();
        assert_eq!(
            document.criterion,
            PromptCriterion::UserText("¿Qué equipo debe atender este caso?".into())
        );
        assert_eq!(document.options[1].description, "Problemas de entrega");
    }

    #[test]
    fn best_match_has_no_baked_in_human_language() {
        let task = DecisionTask {
            id: "classification".into(),
            intent: DecisionIntent::BestMatch,
            instruction: None,
            candidates: vec![
                Candidate { key: "a".into(), description: "uno".into() },
                Candidate { key: "b".into(), description: "dos".into() },
            ],
        };
        let document = compile_document(
            &serde_json::json!("estado"),
            &task,
            &["A".into(), "B".into()],
        )
        .unwrap();
        assert_eq!(document.criterion, PromptCriterion::BestMatch);
    }
}
