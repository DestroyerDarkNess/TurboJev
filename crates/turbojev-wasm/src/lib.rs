#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Browser/WASM binding over TurboJev's canonical asynchronous runtime bridge.
//! JavaScript supplies checkpoint/tokenizer/runtime primitives only; semantic
//! decisions, candidate discovery, planning, calibration and typed answers stay in Rust.

use js_sys::{Function, Promise, Reflect};
use serde::Deserialize;
use serde_json::Value;
use turbojev::{schema::EvaluationRequest, TurboJevAsync};
use turbojev_backend::{
    AsyncScoringBackend, BackendCapabilities, BackendError, BackendFuture, BackendMetrics,
    BackendOutput, BackendResourceHints, DecisionBatch, ExecutionPlan, TaskScores,
};
use turbojev_engine::{ExecutionBudget, PlanningPolicy};
use turbojev_prompt::{
    PromptCriterion, PromptDocument, PromptError, PromptRenderer, RenderedMessages,
};
use turbojev_runtime::{
    AsyncRuntimeExecutor, PromptCodec, RuntimeBackend, RuntimeCapabilities, RuntimeError,
    RuntimeFuture, RuntimeOutput, RuntimeTaskScores, TokenId,
};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DescriptorCapabilities {
    next_token_logits: bool,
    sequence_scores: bool,
    shared_prefix: bool,
    parallel_suffixes: bool,
    selective_output_head: bool,
    local_inference: bool,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeDescriptor {
    #[serde(rename = "name")]
    _name: String,
    model: String,
    #[serde(default)]
    codec_profile: String,
    #[serde(default)]
    scoring_mode: ScoringMode,
    #[serde(default = "text_modalities")]
    modalities: Vec<String>,
    capabilities: DescriptorCapabilities,
    #[serde(default)]
    resource_hints: RuntimeHints,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum ScoringMode {
    #[default]
    Tokens,
    Decisions,
}
fn text_modalities() -> Vec<String> {
    vec!["text".into()]
}

// Decision scorers receive original media evidence, avoiding lossy text tokenization.
// They own model-specific preprocessing; Rust still validates and normalizes scores.
enum BrowserBackend {
    Tokens(RuntimeBackend<BrowserRenderer, JsCodec, JsExecutor>),
    Decisions {
        runtime: JsValue,
        descriptor: RuntimeDescriptor,
    },
}
impl AsyncScoringBackend for BrowserBackend {
    fn name(&self) -> &'static str {
        match self {
            Self::Tokens(b) => b.name(),
            Self::Decisions { .. } => "browser-decision-runtime",
        }
    }
    fn capabilities(&self) -> BackendCapabilities {
        match self {
            Self::Tokens(b) => b.capabilities(),
            Self::Decisions { descriptor, .. } => BackendCapabilities {
                local_inference: descriptor.capabilities.local_inference,
                ..Default::default()
            },
        }
    }
    fn resource_hints(&self) -> BackendResourceHints {
        match self {
            Self::Tokens(b) => b.resource_hints(),
            _ => BackendResourceHints::default(),
        }
    }
    fn score_async<'a>(
        &'a self,
        batch: &'a DecisionBatch,
        plan: ExecutionPlan,
    ) -> BackendFuture<'a, Result<BackendOutput, BackendError>> {
        match self {
            Self::Tokens(b) => b.score_async(batch, plan),
            Self::Decisions {
                runtime,
                descriptor,
            } => Box::pin(async move {
                if !self.capabilities().supports(plan) {
                    return Err(BackendError::UnsupportedPlan);
                }
                turbojev::schema::media::MediaEvidence::check_modalities(
                    &batch.state,
                    &descriptor.modalities,
                )
                .map_err(BackendError::Unsupported)?;
                let batch_json = serde_json::to_string(batch)
                    .map_err(|e| BackendError::InvalidInput(e.to_string()))?;
                let plan_json = serde_json::to_string(&plan)
                    .map_err(|e| BackendError::InvalidInput(e.to_string()))?;
                let function = Reflect::get(runtime, &JsValue::from_str("scoreDecisionBatch"))
                    .map_err(|e| BackendError::Inference(js_message(e)))?
                    .dyn_into::<Function>()
                    .map_err(|_| BackendError::NotReady("scoreDecisionBatch is required".into()))?;
                let value = function
                    .call2(
                        runtime,
                        &JsValue::from_str(&batch_json),
                        &JsValue::from_str(&plan_json),
                    )
                    .map_err(|e| BackendError::Inference(js_message(e)))?;
                let resolved = JsFuture::from(Promise::resolve(&value))
                    .await
                    .map_err(|e| BackendError::Inference(js_message(e)))?;
                let text = resolved.as_string().ok_or_else(|| {
                    BackendError::Inference("scoreDecisionBatch must return JSON string".into())
                })?;
                let output = parse_runtime_output(&text, &descriptor.model)
                    .map_err(|e| BackendError::Inference(e.to_string()))?;
                Ok(BackendOutput {
                    model: output.model,
                    metrics: output.metrics,
                    scores: output
                        .scores
                        .into_iter()
                        .map(|s| TaskScores {
                            id: s.id,
                            logits: s.logits,
                        })
                        .collect(),
                })
            }),
        }
    }
}
#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RuntimeHints {
    max_parallel_tasks: Option<usize>,
    estimated_parallel_task_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptProfile {
    #[serde(default = "default_profile")]
    profile: String,
    #[serde(default = "default_system")]
    system_instruction: String,
    #[serde(default = "default_best_match")]
    best_match_instruction: String,
}
impl Default for PromptProfile {
    fn default() -> Self {
        Self {
            profile: default_profile(),
            system_instruction: default_system(),
            best_match_instruction: default_best_match(),
        }
    }
}
fn default_profile() -> String {
    "browser-direct-decision-en-v2".into()
}
fn default_system() -> String {
    "Make the requested decision from the supplied state. Follow the requested output format exactly.".into()
}
fn default_best_match() -> String {
    "Choose the allowed option that best matches the supplied evidence.".into()
}

#[derive(Clone)]
struct JsCodec {
    runtime: JsValue,
    profile: String,
}
impl JsCodec {
    fn call_string(&self, name: &str, args: &[JsValue]) -> Result<String, RuntimeError> {
        let function = Reflect::get(&self.runtime, &JsValue::from_str(name))
            .map_err(js_codec)?
            .dyn_into::<Function>()
            .map_err(|_| RuntimeError::Codec(format!("runtime method {name} is not a function")))?;
        let value = match args.len() {
            1 => function.call1(&self.runtime, &args[0]),
            2 => function.call2(&self.runtime, &args[0], &args[1]),
            _ => {
                return Err(RuntimeError::Codec(
                    "unsupported browser codec arity".into(),
                ))
            }
        }
        .map_err(js_codec)?;
        if value.is_instance_of::<Promise>() {
            return Err(RuntimeError::Codec(format!(
                "runtime {name} must be synchronous after initialization"
            )));
        }
        value.as_string().ok_or_else(|| {
            RuntimeError::Codec(format!("runtime {name} must return a JSON/text string"))
        })
    }
}
impl PromptCodec for JsCodec {
    fn profile(&self) -> &str {
        &self.profile
    }
    fn encode_prompt(&self, messages: &RenderedMessages) -> Result<Vec<TokenId>, RuntimeError> {
        let json =
            serde_json::to_string(messages).map_err(|e| RuntimeError::Codec(e.to_string()))?;
        let out = self.call_string("encodePrompt", &[JsValue::from_str(&json)])?;
        serde_json::from_str(&out).map_err(|e| RuntimeError::Codec(e.to_string()))
    }
    fn encode_prompt_with_answer(
        &self,
        messages: &RenderedMessages,
        answer: &str,
    ) -> Result<Vec<TokenId>, RuntimeError> {
        let json =
            serde_json::to_string(messages).map_err(|e| RuntimeError::Codec(e.to_string()))?;
        let out = self.call_string(
            "encodePromptWithAnswer",
            &[JsValue::from_str(&json), JsValue::from_str(answer)],
        )?;
        serde_json::from_str(&out).map_err(|e| RuntimeError::Codec(e.to_string()))
    }
    fn decode(&self, token_ids: &[TokenId]) -> Result<String, RuntimeError> {
        let json =
            serde_json::to_string(token_ids).map_err(|e| RuntimeError::Codec(e.to_string()))?;
        self.call_string("decodeTokens", &[JsValue::from_str(&json)])
    }
}

#[derive(Clone)]
struct BrowserRenderer {
    profile: PromptProfile,
}
impl PromptRenderer for BrowserRenderer {
    fn profile(&self) -> &str {
        &self.profile.profile
    }
    fn render(&self, document: &PromptDocument) -> Result<RenderedMessages, PromptError> {
        let criterion = match &document.criterion {
            PromptCriterion::UserText(t) => t.clone(),
            PromptCriterion::BestMatch => self.profile.best_match_instruction.clone(),
        };
        let evidence = match &document.evidence {
            Value::String(text) => text.clone(),
            value => {
                serde_json::to_string(value).map_err(|e| PromptError::Renderer(e.to_string()))?
            }
        };
        let options = document
            .options
            .iter()
            .map(|option| format!("{}. {}", option.code, option.description))
            .collect::<Vec<_>>()
            .join("\n");
        let codes = document
            .options
            .iter()
            .map(|option| option.code.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let label = if document.options.iter().all(|option| {
            option.code.len() == 1 && option.code.bytes().all(|c| c.is_ascii_alphabetic())
        }) {
            "letter"
        } else {
            "code"
        };
        let user = format!(
            "State:\n{evidence}\n\nQuestion:\n{criterion}\n\nAllowed options:\n{options}\n\nReply with exactly one option {label} from: {codes}."
        );
        Ok(RenderedMessages {
            system: Some(self.profile.system_instruction.clone()),
            user,
            profile: self.profile.profile.clone(),
        })
    }
}

#[derive(Clone)]
struct JsExecutor {
    runtime: JsValue,
    descriptor: RuntimeDescriptor,
}
impl JsExecutor {
    fn capabilities_value(&self) -> RuntimeCapabilities {
        let c = &self.descriptor.capabilities;
        RuntimeCapabilities {
            next_token_logits: c.next_token_logits,
            sequence_scores: c.sequence_scores,
            shared_prefix: c.shared_prefix,
            parallel_suffixes: c.parallel_suffixes,
            selective_output_head: c.selective_output_head,
            local_inference: c.local_inference,
        }
    }
}
impl AsyncRuntimeExecutor for JsExecutor {
    fn name(&self) -> &'static str {
        "browser-js-runtime"
    }
    fn capabilities(&self) -> RuntimeCapabilities {
        self.capabilities_value()
    }
    fn resource_hints(&self) -> BackendResourceHints {
        BackendResourceHints {
            max_parallel_tasks: self.descriptor.resource_hints.max_parallel_tasks,
            estimated_parallel_task_bytes: self
                .descriptor
                .resource_hints
                .estimated_parallel_task_bytes,
        }
    }
    fn score_async<'a>(
        &'a self,
        batch: &'a turbojev_runtime::RuntimeBatch,
        plan: ExecutionPlan,
    ) -> RuntimeFuture<'a, Result<RuntimeOutput, RuntimeError>> {
        Box::pin(async move {
            let tasks:Vec<Value>=batch.tasks.iter().map(|task|{
                let mut prompt=batch.common_prefix.clone();prompt.extend_from_slice(&task.prompt_suffix);
                serde_json::json!({"id":task.id,"promptTokens":prompt,"promptSuffix":task.prompt_suffix,"candidates":task.candidates.iter().map(|c|serde_json::json!({"symbol":c.symbol,"tokenIds":c.token_ids})).collect::<Vec<_>>()})
            }).collect();
            let batch_json = serde_json::to_string(
                &serde_json::json!({"commonPrefix":batch.common_prefix,"tasks":tasks}),
            )
            .map_err(|e| RuntimeError::Preparation(e.to_string()))?;
            let plan_json = serde_json::to_string(&plan)
                .map_err(|e| RuntimeError::Preparation(e.to_string()))?;
            let function = Reflect::get(&self.runtime, &JsValue::from_str("scoreRuntimeBatch"))
                .map_err(js_inference)?
                .dyn_into::<Function>()
                .map_err(|_| {
                    RuntimeError::Inference("scoreRuntimeBatch is not a function".into())
                })?;
            let value = function
                .call2(
                    &self.runtime,
                    &JsValue::from_str(&batch_json),
                    &JsValue::from_str(&plan_json),
                )
                .map_err(js_inference)?;
            let resolved = if value.is_instance_of::<Promise>() {
                JsFuture::from(value.unchecked_into::<Promise>())
                    .await
                    .map_err(js_inference)?
            } else {
                value
            };
            let text = resolved.as_string().ok_or_else(|| {
                RuntimeError::Inference("scoreRuntimeBatch must resolve to JSON string".into())
            })?;
            parse_runtime_output(&text, &self.descriptor.model)
        })
    }
}

fn parse_runtime_output(text: &str, default_model: &str) -> Result<RuntimeOutput, RuntimeError> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| RuntimeError::Inference(e.to_string()))?;
    let scores = value
        .get("scores")
        .and_then(Value::as_array)
        .ok_or_else(|| RuntimeError::Inference("runtime output missing scores".into()))?
        .iter()
        .map(|row| {
            let id = row
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| RuntimeError::Inference("score row missing id".into()))?
                .to_owned();
            let logits = row
                .get("logits")
                .and_then(Value::as_array)
                .ok_or_else(|| RuntimeError::Inference("score row missing logits".into()))?
                .iter()
                .map(|v| {
                    v.as_f64().map(|x| x as f32).ok_or_else(|| {
                        RuntimeError::Inference("logit must be finite number".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if logits.iter().any(|x| !x.is_finite()) {
                return Err(RuntimeError::Inference(
                    "runtime returned non-finite logit".into(),
                ));
            }
            Ok(RuntimeTaskScores { id, logits })
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    let metrics = value
        .get("metrics")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(RuntimeOutput {
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| Some(default_model.to_owned())),
        scores,
        metrics: BackendMetrics {
            prefill_seconds: num(&metrics, "prefill_seconds"),
            scoring_seconds: num(&metrics, "scoring_seconds"),
            total_seconds: num(&metrics, "total_seconds"),
            input_tokens: uint(&metrics, "input_tokens"),
            parallel_batches: uint(&metrics, "parallel_batches"),
            peak_parallel_tasks: uint(&metrics, "peak_parallel_tasks"),
            candidate_rows_projected: uint(&metrics, "candidate_rows_projected"),
            ..BackendMetrics::default()
        },
    })
}
fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}
fn uint(v: &Value, key: &str) -> Option<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
}
fn js_codec(v: JsValue) -> RuntimeError {
    RuntimeError::Codec(js_message(v))
}
fn js_inference(v: JsValue) -> RuntimeError {
    RuntimeError::Inference(js_message(v))
}
fn js_message(v: JsValue) -> String {
    v.as_string()
        .unwrap_or_else(|| format!("JavaScript error: {v:?}"))
}

/// High-level browser engine. The runtime object must already be initialized.
#[wasm_bindgen]
pub struct WasmEngine {
    engine: TurboJevAsync<BrowserBackend>,
}

#[wasm_bindgen]
impl WasmEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(
        runtime: JsValue,
        descriptor_json: &str,
        prompt_profile_json: Option<String>,
    ) -> Result<WasmEngine, JsValue> {
        let descriptor: RuntimeDescriptor =
            serde_json::from_str(descriptor_json).map_err(js_json)?;
        if descriptor.modalities.is_empty()
            || descriptor
                .modalities
                .iter()
                .any(|m| !matches!(m.as_str(), "text" | "image" | "audio" | "video"))
        {
            return Err(JsValue::from_str("invalid runtime modalities"));
        }
        if descriptor.scoring_mode == ScoringMode::Decisions {
            if !Reflect::get(&runtime, &JsValue::from_str("scoreDecisionBatch"))?.is_function() {
                return Err(JsValue::from_str(
                    "decision runtime requires scoreDecisionBatch",
                ));
            }
            return Ok(Self {
                engine: TurboJevAsync::new(BrowserBackend::Decisions {
                    runtime,
                    descriptor,
                }),
            });
        }
        if descriptor.modalities.iter().any(|m| m != "text") {
            return Err(JsValue::from_str(
                "multimodal runtimes require scoringMode=decisions",
            ));
        }
        if !descriptor.capabilities.next_token_logits {
            return Err(JsValue::from_str("runtime must expose next-token logits"));
        }
        let profile: PromptProfile = match prompt_profile_json {
            Some(v) => serde_json::from_str(&v).map_err(js_json)?,
            None => PromptProfile::default(),
        };
        let max_tokens = if descriptor.capabilities.sequence_scores {
            4
        } else {
            1
        };
        let backend = RuntimeBackend::new(
            BrowserRenderer { profile },
            JsCodec {
                runtime: runtime.clone(),
                profile: descriptor.codec_profile.clone(),
            },
            JsExecutor {
                runtime,
                descriptor,
            },
        )
        .with_max_tokens_per_code(max_tokens);
        Ok(Self {
            engine: TurboJevAsync::new(BrowserBackend::Tokens(backend))
                .with_planning_policy(PlanningPolicy::Auto)
                .with_execution_budget(ExecutionBudget::unlimited()),
        })
    }

    pub async fn evaluate_json(&self, input: &str) -> Result<String, JsValue> {
        let request: EvaluationRequest =
            turbojev::wire::decode_request(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let response = self.engine.evaluate(request).await.map_err(core_error)?;
        turbojev::wire::encode_response(&response).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    pub async fn classify_json(&self, state: &str, options_json: &str) -> Result<String, JsValue> {
        let options: Vec<String> = serde_json::from_str(options_json).map_err(js_json)?;
        let result = self
            .engine
            .classify(state.to_owned(), options)
            .await
            .map_err(core_error)?;
        serde_json::to_string(
            &serde_json::json!({"choice":result.choice,"probabilities":result.probabilities}),
        )
        .map_err(js_json)
    }
    pub async fn evaluate_compatible_json(
        &self,
        profile: &str,
        input: &str,
    ) -> Result<String, JsValue> {
        let profile = parse_profile(profile)?;
        let request = turbojev::compat::decode_request(profile, input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let response = self
            .engine
            .evaluate(request.clone())
            .await
            .map_err(core_error)?;
        turbojev::compat::encode_response(profile, &request, &response)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn parse_profile(v: &str) -> Result<turbojev::compat::WireProfile, JsValue> {
    match v {
        "turbojev" => Ok(turbojev::compat::WireProfile::TurboJev),
        "typesafe-systemone-v1" | "typesafe" => {
            Ok(turbojev::compat::WireProfile::TypeSafeSystemOne)
        }
        "ai-sdk-evaluation-v4" | "ai-sdk" => Ok(turbojev::compat::WireProfile::AiSdkEvaluationV4),
        _ => Err(JsValue::from_str("unknown compatibility profile")),
    }
}
fn js_json(e: serde_json::Error) -> JsValue {
    JsValue::from_str(&e.to_string())
}
fn core_error(e: turbojev::Error) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub fn turbojev_wasm_contract() -> String {
    "turbojev.wasm.v2".to_owned()
}
