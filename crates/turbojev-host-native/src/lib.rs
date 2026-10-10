#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Shared native host/session layer used by Node, Python, C/.NET and mobile bindings.
//!
//! Filesystem/network/package-manager concerns remain in the language host. This crate
//! accepts an already-resolved local model path (or a deterministic `fixture://` spec)
//! and owns exactly one thread-affine model session.

mod control;
pub mod worker;
pub use control::GenerationControl;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use turbojev::{OrderRobustness, TurboJevEngine};
use turbojev_backend::{FusedBackendCapabilities, FusedScoringBackend};
use turbojev_backend_mock::FixtureBackend;
use turbojev_compat::{self, WireProfile};
use turbojev_engine::{ExecutionBudget, PlanningPolicy};
#[cfg(feature = "llamacpp")]
use turbojev_runtime_qualification::LLAMACPP_SHIM_ABI_V5;
use turbojev_runtime_qualification::{RuntimeFingerprint, RuntimeQualificationManifest};
use turbojev_schema::{EvaluationRequest, EvaluationResponse};

#[cfg(feature = "llamacpp")]
use turbojev_runtime::RuntimeBackend;
#[cfg(feature = "llamacpp")]
use turbojev_runtime_llamacpp::{
    ffi::{FfiLlamaCpp, NativeConfig},
    llamacpp_backend, LlamaCppApi, LlamaCppCodec, LlamaCppExecutor, ReferencePromptConfig,
    ReferencePromptRenderer,
};

/// Load-time execution policy exposed uniformly to native language bindings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum HostPlanningPolicy {
    #[default]
    Auto,
    Reference,
    DeterministicReuse,
}

/// Load-time candidate-order policy shared by every native language binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum HostOrderRobustness {
    #[default]
    None,
    Cyclic,
    Balanced,
}

impl From<HostOrderRobustness> for OrderRobustness {
    fn from(value: HostOrderRobustness) -> Self {
        match value {
            HostOrderRobustness::None => OrderRobustness::None,
            HostOrderRobustness::Cyclic => OrderRobustness::Cyclic,
            HostOrderRobustness::Balanced => OrderRobustness::Balanced,
        }
    }
}

impl From<HostPlanningPolicy> for PlanningPolicy {
    fn from(value: HostPlanningPolicy) -> Self {
        match value {
            HostPlanningPolicy::Auto => PlanningPolicy::Auto,
            HostPlanningPolicy::Reference => PlanningPolicy::Reference,
            HostPlanningPolicy::DeterministicReuse => PlanningPolicy::DeterministicReuse,
        }
    }
}

/// Portable scheduling budget requested by a native binding.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HostExecutionOptions {
    #[serde(default)]
    pub policy: HostPlanningPolicy,
    /// Opt-in cyclic logit or canonical balanced probability averaging for choices.
    #[serde(default)]
    pub order_robustness: HostOrderRobustness,
    #[serde(default)]
    pub max_parallel_tasks: Option<usize>,
    #[serde(default)]
    pub memory_budget_bytes: Option<u64>,
    /// Conservative opt-in for llama.cpp cross-request multi-root execution.
    /// Default false: enable only after the concrete llama.cpp backend/hardware
    /// combination passes TurboJev fused-vs-independent conformance.
    #[serde(default)]
    pub llamacpp_multi_root_fused: bool,
    /// Evidence produced by a real model/runtime/backend/hardware qualification run.
    /// Required when `llamacppMultiRootFused=true` from v0.26 onward.
    #[serde(default)]
    pub llamacpp_qualification: Option<RuntimeQualificationManifest>,
    /// Exact fingerprint of the runtime target being opened. Hosts construct this
    /// from their deployment/model provenance; TurboJev compares it byte-for-byte
    /// with the qualification evidence before enabling fused execution.
    #[serde(default)]
    pub runtime_fingerprint: Option<RuntimeFingerprint>,
}

/// Stable native-host execution-capability contract. Server transports map this
/// into their own portable cross-request capability schema instead of guessing
/// from a runtime/model name.
pub const NATIVE_EXECUTION_CAPABILITIES_CONTRACT: &str =
    "turbojev.native-execution-capabilities.v1";

/// Capabilities exposed by one loaded native session for independent requests.
///
/// These flags describe *cross-request* execution only. llama.cpp may still batch
/// suffix branches inside a single EvaluationRequest without implying fused HTTP
/// request batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeExecutionCapabilities {
    pub fused_batch: bool,
    pub cooperative_cancellation: bool,
    pub hard_deadlines: bool,
    pub max_fused_items: Option<usize>,
    pub estimated_fused_item_bytes: Option<u64>,
}

impl NativeExecutionCapabilities {
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

impl Default for NativeExecutionCapabilities {
    fn default() -> Self {
        Self::serial_only()
    }
}

impl From<FusedBackendCapabilities> for NativeExecutionCapabilities {
    fn from(value: FusedBackendCapabilities) -> Self {
        Self {
            fused_batch: value.fused_batch,
            cooperative_cancellation: false,
            hard_deadlines: false,
            max_fused_items: value.max_fused_items,
            estimated_fused_item_bytes: value.estimated_fused_item_bytes,
        }
    }
}

/// Adapter-owned control-language profile. Caller state and criteria are never translated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HostPromptProfile {
    #[serde(default = "default_profile")]
    pub profile: String,
    #[serde(default = "default_system_instruction")]
    pub system_instruction: String,
    #[serde(default = "default_best_match_instruction")]
    pub best_match_instruction: String,
}

impl Default for HostPromptProfile {
    fn default() -> Self {
        Self {
            profile: default_profile(),
            system_instruction: default_system_instruction(),
            best_match_instruction: default_best_match_instruction(),
        }
    }
}

fn default_profile() -> String {
    "generic-decision-en-v3".to_owned()
}
fn default_system_instruction() -> String {
    "Make the requested decision from the supplied state. Follow the requested output format exactly.".to_owned()
}
fn default_best_match_instruction() -> String {
    "Choose the allowed option that best matches the supplied evidence.".to_owned()
}
fn default_n_ctx() -> u32 {
    4096
}
fn default_n_batch() -> u32 {
    512
}
fn default_n_seq_max() -> u32 {
    8
}
fn default_gpu_layers() -> i32 {
    0
}

/// Accelerator requested for the llama.cpp model runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NativeDevice {
    Cpu,
    Gpu,
}

/// Host-neutral native model/session options. `model` must already be a local path.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeLoadOptions {
    /// Optional trusted local scorer, configured only at model load time.
    #[serde(default)]
    pub worker: Option<worker::WorkerOptions>,
    pub model: String,
    /// Generate bounded model reasoning before scoring. Off by default.
    #[serde(default)]
    pub thinking: bool,
    #[serde(default = "default_max_thinking_tokens")]
    pub max_thinking_tokens: u32,
    /// Optional llama.cpp multimodal projector GGUF matching `model`.
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default = "default_n_ctx")]
    pub n_ctx: u32,
    #[serde(default = "default_n_batch")]
    pub n_batch: u32,
    #[serde(default = "default_n_seq_max")]
    pub n_seq_max: u32,
    #[serde(default = "default_gpu_layers")]
    pub n_gpu_layers: i32,
    /// Explicit runtime device. Omitted means CPU unless legacy GPU layers were requested.
    #[serde(default)]
    pub device: Option<NativeDevice>,
    #[serde(default)]
    pub n_threads: i32,
    #[serde(default)]
    pub n_threads_batch: i32,
    #[serde(default)]
    pub execution: HostExecutionOptions,
    #[serde(default)]
    pub prompt: HostPromptProfile,
    /// Exact logits for deterministic binding/conformance tests. Used only for `fixture://` models.
    #[serde(default)]
    pub fixture_logits: BTreeMap<String, Vec<f32>>,
}

fn default_max_thinking_tokens() -> u32 {
    512
}

impl NativeLoadOptions {
    fn validate(&self) -> Result<(), HostError> {
        if self.model.trim().is_empty() {
            return Err(HostError::InvalidOptions("model must not be empty".into()));
        }
        if self.thinking
            && (self.max_thinking_tokens == 0 || self.max_thinking_tokens >= self.n_ctx)
        {
            return Err(HostError::InvalidOptions(
                "maxThinkingTokens must be positive and smaller than nCtx".into(),
            ));
        }
        if self.thinking
            && (self.worker.is_some()
                || self.mmproj.is_some()
                || self.model.starts_with("fixture://"))
        {
            return Err(HostError::InvalidOptions(
                "thinking requires the native llama.cpp text runtime".into(),
            ));
        }
        if self.n_ctx == 0 || self.n_batch == 0 || self.n_seq_max == 0 {
            return Err(HostError::InvalidOptions(
                "nCtx, nBatch and nSeqMax must be greater than zero".into(),
            ));
        }
        if self.device == Some(NativeDevice::Cpu) && self.n_gpu_layers != 0 {
            return Err(HostError::InvalidOptions(
                "device=cpu cannot be combined with nonzero nGpuLayers".into(),
            ));
        }
        if self.prompt.profile.trim().is_empty()
            || self.prompt.system_instruction.trim().is_empty()
            || self.prompt.best_match_instruction.trim().is_empty()
        {
            return Err(HostError::InvalidOptions(
                "prompt profile fields must not be empty".into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "llamacpp")]
    fn effective_gpu_layers(&self) -> i32 {
        match self.device {
            Some(NativeDevice::Cpu) => 0,
            Some(NativeDevice::Gpu) if self.n_gpu_layers == 0 => -1,
            Some(NativeDevice::Gpu) | None => self.n_gpu_layers,
        }
    }
}

#[cfg(feature = "llamacpp")]
type LlamaBackend = RuntimeBackend<
    ReferencePromptRenderer,
    LlamaCppCodec<FfiLlamaCpp>,
    LlamaCppExecutor<FfiLlamaCpp>,
>;
#[cfg(feature = "llamacpp")]
type LlamaEngine = TurboJevEngine<LlamaBackend>;

type FixtureEngine = TurboJevEngine<FixtureBackend>;

enum NativeEngine {
    Worker(TurboJevEngine<worker::WorkerBackend>),
    Fixture(FixtureEngine),
    #[cfg(feature = "llamacpp")]
    Llama(LlamaEngine),
}

/// One loaded native session. The type intentionally does not implement `Clone`.
pub struct NativeSession {
    engine: NativeEngine,
    model: String,
    modalities: Vec<String>,
    qualification_id: Option<String>,
}

impl NativeSession {
    /// Load from a typed configuration.
    pub fn open(options: NativeLoadOptions) -> Result<Self, HostError> {
        options.validate()?;
        if let Some(worker_options) = &options.worker {
            let backend = worker::WorkerBackend::open(worker_options)
                .map_err(|e| HostError::Runtime(e.to_string()))?;
            let modalities = backend.modalities().to_vec();
            return Ok(Self {
                engine: NativeEngine::Worker(
                    TurboJevEngine::new(backend)
                        .with_planning_policy(options.execution.policy.into())
                        .with_order_robustness(options.execution.order_robustness.into())
                        .with_execution_budget(execution_budget(options.execution)),
                ),
                model: options.model,
                modalities,
                qualification_id: None,
            });
        }
        if options.model.starts_with("fixture://") {
            let mut builder = FixtureBackend::builder().model(options.model.clone());
            for (id, logits) in &options.fixture_logits {
                builder = builder.with_logits(id.clone(), logits.clone());
            }
            let mut engine = TurboJevEngine::new(builder.build())
                .with_planning_policy(options.execution.policy.into())
                .with_order_robustness(options.execution.order_robustness.into());
            engine = engine.with_execution_budget(execution_budget(options.execution));
            return Ok(Self {
                engine: NativeEngine::Fixture(engine),
                model: options.model,
                modalities: vec!["text".to_owned()],
                qualification_id: None,
            });
        }

        #[cfg(feature = "llamacpp")]
        {
            let native = NativeConfig {
                n_ctx: options.n_ctx,
                n_batch: options.n_batch,
                n_seq_max: if options.thinking {
                    // Leave space for thoughts while bounding the attention graph.
                    // Use nSeqMax=1 explicitly for longer prompts/thoughts.
                    options.n_seq_max.min(4)
                } else {
                    options.n_seq_max
                },
                n_gpu_layers: options.effective_gpu_layers(),
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
            };
            let qualification = qualify_llamacpp_multi_root(&options)?;
            let loaded = match options.mmproj.as_deref() {
                Some(mmproj) => FfiLlamaCpp::open_multimodal(&options.model, mmproj, native),
                None => FfiLlamaCpp::open(&options.model, native),
            };
            let mut runtime = loaded
                .map_err(|error| HostError::Runtime(error.to_string()))?
                .with_thinking(options.thinking, options.max_thinking_tokens);
            let mut modalities = vec!["text".to_owned()];
            if runtime.image_input_supported() {
                modalities.push("image".to_owned());
            }
            if runtime.audio_input_supported() {
                modalities.push("audio".to_owned());
            }
            if runtime.video_input_supported() {
                modalities.push("video".to_owned());
            }
            runtime = runtime.with_multi_root_enabled(qualification.enabled);
            if let Some(bytes) = qualification.estimated_fused_item_bytes {
                runtime = runtime.with_estimated_multi_root_item_bytes(bytes);
            }
            let prompt = ReferencePromptRenderer::new(ReferencePromptConfig::new(
                options.prompt.profile,
                options.prompt.system_instruction,
                options.prompt.best_match_instruction,
            ));
            let backend = llamacpp_backend(prompt, runtime);
            let engine = TurboJevEngine::new(backend)
                .with_planning_policy(options.execution.policy.into())
                .with_order_robustness(options.execution.order_robustness.into())
                .with_execution_budget(execution_budget(options.execution));
            return Ok(Self {
                engine: NativeEngine::Llama(engine),
                model: options.model,
                modalities,
                qualification_id: qualification.id,
            });
        }

        #[cfg(not(feature = "llamacpp"))]
        Err(HostError::RuntimeUnavailable(
            "this TurboJev native build has no concrete model runtime; rebuild with the `llamacpp` feature or use fixture:// for binding tests".into(),
        ))
    }

    /// Parse JSON load options using the shared binding contract.
    pub fn open_json(input: &str) -> Result<Self, HostError> {
        let options: NativeLoadOptions = serde_json::from_str(input).map_err(HostError::Json)?;
        Self::open(options)
    }

    /// Evaluate one already-validated canonical request.
    ///
    /// Host transports use this typed path so JSON/profile translation stays in
    /// `turbojev-wire`, `turbojev-compat`, or `turbojev-server-protocol` rather
    /// than being duplicated around the concrete runtime session.
    pub fn evaluate(&self, request: EvaluationRequest) -> Result<EvaluationResponse, HostError> {
        match &self.engine {
            NativeEngine::Worker(engine) => engine
                .evaluate(request)
                .map_err(|e| HostError::Core(e.to_string())),
            NativeEngine::Fixture(engine) => engine
                .evaluate(request)
                .map_err(|e| HostError::Core(e.to_string())),
            #[cfg(feature = "llamacpp")]
            NativeEngine::Llama(engine) => engine
                .evaluate(request)
                .map_err(|e| HostError::Core(e.to_string())),
        }
    }

    /// Truthful cross-request execution capabilities for this concrete session.
    ///
    /// The deterministic fixture and llama.cpp ABI v5 expose real fused backend
    /// operations. llama.cpp eligibility remains request-shape dependent: only
    /// independent single-token candidate roots that fit native limits are fused.
    pub fn execution_capabilities(&self) -> NativeExecutionCapabilities {
        match &self.engine {
            NativeEngine::Worker(_) => NativeExecutionCapabilities::serial_only(),
            NativeEngine::Fixture(engine) => engine.backend().fused_capabilities().into(),
            #[cfg(feature = "llamacpp")]
            NativeEngine::Llama(engine) => engine.backend().fused_capabilities().into(),
        }
    }

    /// Shape-aware fused eligibility for the concrete loaded session.
    pub fn can_evaluate_fused(&self, requests: &[EvaluationRequest]) -> Result<bool, HostError> {
        match &self.engine {
            NativeEngine::Worker(_) => Ok(false),
            NativeEngine::Fixture(engine) => engine
                .can_evaluate_fused(requests)
                .map_err(|error| HostError::Core(error.to_string())),
            #[cfg(feature = "llamacpp")]
            NativeEngine::Llama(engine) => engine
                .can_evaluate_fused(requests)
                .map_err(|error| HostError::Core(error.to_string())),
        }
    }

    /// Runtime bridge for future exact fused cross-request implementations.
    ///
    /// Callers must check `execution_capabilities().fused_batch` first. The
    /// deterministic fixture and eligible llama.cpp ABI v5 workloads implement
    /// this method with one true fused backend operation. Ineligible request
    /// shapes are rejected here and should be routed through `can_evaluate_fused`.
    pub fn evaluate_fused(
        &self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<EvaluationResponse>, HostError> {
        match &self.engine {
            NativeEngine::Worker(_) => Err(HostError::UnsupportedCapability(
                "worker fused execution".into(),
            )),
            NativeEngine::Fixture(engine) => engine
                .evaluate_fused(requests)
                .map_err(|error| HostError::Core(error.to_string())),
            #[cfg(feature = "llamacpp")]
            NativeEngine::Llama(engine) => engine
                .evaluate_fused(requests)
                .map_err(|error| HostError::Core(error.to_string())),
        }
    }

    /// Evaluate canonical TurboJev JSON and return canonical TurboJev JSON.
    pub fn evaluate_json(&self, input: &str) -> Result<String, HostError> {
        let request =
            turbojev_wire::decode_request(input).map_err(|e| HostError::Wire(e.to_string()))?;
        let response = self.evaluate(request)?;
        turbojev_wire::encode_response(&response).map_err(|e| HostError::Wire(e.to_string()))
    }

    /// Additive JSON Schema API; all field semantics live in the portable core.
    pub fn generate_json(&self, input: &str) -> Result<String, HostError> {
        self.generate_json_controlled(input, &GenerationControl::new(None)?)
    }

    /// Cancellation and timeout are cooperative between model operations, not inside a decode.
    pub fn generate_json_controlled(
        &self,
        input: &str,
        control: &GenerationControl,
    ) -> Result<String, HostError> {
        control.check()?;
        let payload: serde_json::Value = serde_json::from_str(input).map_err(HostError::Json)?;
        let mut request: turbojev_schema::fields::GenerateRequest =
            serde_json::from_value(payload.clone()).map_err(HostError::Json)?;
        if payload.get("seed").is_none() && request.samples() {
            use std::hash::{BuildHasher, Hasher};
            // Entropy belongs to the host; explicit seeds never consume this stream.
            request.seed = std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish();
        }
        let response = match &self.engine {
            NativeEngine::Worker(engine) => engine.generate_controlled(request, || {
                control
                    .check()
                    .map_err(|error| turbojev_backend::BackendError::Inference(error.to_string()))
            }),
            NativeEngine::Fixture(engine) => engine.generate_controlled(request, || {
                control
                    .check()
                    .map_err(|error| turbojev_backend::BackendError::Inference(error.to_string()))
            }),
            #[cfg(feature = "llamacpp")]
            NativeEngine::Llama(engine) => engine.generate_controlled(request, || {
                control
                    .check()
                    .map_err(|error| turbojev_backend::BackendError::Inference(error.to_string()))
            }),
        }
        .map_err(|error| {
            control
                .check()
                .err()
                .map(|cause| HostError::Generation {
                    cause: Box::new(cause),
                    usage: error.usage().cloned().unwrap_or_default(),
                })
                .unwrap_or_else(|| HostError::Core(error.to_string()))
        })?;
        control.check()?;
        serde_json::to_string(&response).map_err(HostError::Json)
    }

    /// Standalone candidate normalization, shared by thin SDK helpers.
    pub fn candidate_softmax_json(input: &str) -> Result<String, HostError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Input {
            logits: indexmap::IndexMap<String, f64>,
            temperature: f64,
        }
        let input: Input = serde_json::from_str(input)?;
        let values: Vec<_> = input.logits.values().copied().collect();
        let probabilities = turbojev::calibration::candidate_softmax(&values, input.temperature)
            .map_err(|error| HostError::InvalidOptions(error.to_string()))?;
        let distribution: indexmap::IndexMap<_, _> =
            input.logits.into_keys().zip(probabilities).collect();
        serde_json::to_string(&distribution).map_err(HostError::Json)
    }

    /// Compile schemas without inference; the result is shared by all bindings.
    pub fn compile_schema_json(input: &str) -> Result<String, HostError> {
        turbojev_schema::fields::compile_schema_json(input)
            .map_err(|error| HostError::InvalidOptions(error.to_string()))
    }

    /// Evaluate one external compatibility profile without duplicating semantic logic.
    pub fn evaluate_compatible_json(
        &self,
        profile: &str,
        input: &str,
    ) -> Result<String, HostError> {
        let profile = parse_profile(profile)?;
        let request = turbojev_compat::decode_request(profile, input)
            .map_err(|e| HostError::Compatibility(e.to_string()))?;
        let response = self.evaluate(request.clone())?;
        turbojev_compat::encode_response(profile, &request, &response)
            .map_err(|e| HostError::Compatibility(e.to_string()))
    }

    /// Stable metadata used by bindings for diagnostics.
    pub fn info_json(&self) -> Result<String, HostError> {
        serde_json::to_string(&serde_json::json!({
            "model": self.model,
            "runtime": match &self.engine {
                NativeEngine::Worker(_) => "local-worker",
                NativeEngine::Fixture(_) => "fixture",
                #[cfg(feature = "llamacpp")]
                NativeEngine::Llama(_) => "llama.cpp",
            },
            "modalities": self.modalities,
            "contract": "turbojev.host-native.v1",
            "executionCapabilitiesContract": NATIVE_EXECUTION_CAPABILITIES_CONTRACT,
            "executionCapabilities": self.execution_capabilities(),
            "runtimeQualificationId": self.qualification_id,
        }))
        .map_err(HostError::Json)
    }
}

#[cfg(feature = "llamacpp")]
struct LlamaQualificationDecision {
    enabled: bool,
    estimated_fused_item_bytes: Option<u64>,
    id: Option<String>,
}

#[cfg(feature = "llamacpp")]
fn qualify_llamacpp_multi_root(
    options: &NativeLoadOptions,
) -> Result<LlamaQualificationDecision, HostError> {
    if !options.execution.llamacpp_multi_root_fused {
        return Ok(LlamaQualificationDecision {
            enabled: false,
            estimated_fused_item_bytes: None,
            id: None,
        });
    }
    let manifest = options
        .execution
        .llamacpp_qualification
        .as_ref()
        .ok_or_else(|| {
            HostError::InvalidOptions(
                "llamacppMultiRootFused=true requires execution.llamacppQualification evidence"
                    .into(),
            )
        })?;
    let current = options
        .execution
        .runtime_fingerprint
        .as_ref()
        .ok_or_else(|| {
            HostError::InvalidOptions(
                "llamacppMultiRootFused=true requires execution.runtimeFingerprint".into(),
            )
        })?;
    if current.runtime != "llama.cpp" || current.shim_abi != LLAMACPP_SHIM_ABI_V5 {
        return Err(HostError::InvalidOptions(
            "runtimeFingerprint must identify llama.cpp with TurboJev shim ABI v5".into(),
        ));
    }
    if current.model.source != options.model {
        return Err(HostError::InvalidOptions(
            "runtimeFingerprint.model.source must exactly match the loaded model path".into(),
        ));
    }
    let capabilities = manifest.authorize(current).map_err(|error| {
        HostError::InvalidOptions(format!("runtime qualification rejected: {error}"))
    })?;
    if !capabilities.fused_batch {
        return Err(HostError::InvalidOptions(
            "runtime qualification does not authorize fusedBatch".into(),
        ));
    }
    Ok(LlamaQualificationDecision {
        enabled: true,
        estimated_fused_item_bytes: capabilities.estimated_fused_item_bytes,
        id: Some(manifest.id.clone()),
    })
}

fn execution_budget(options: HostExecutionOptions) -> ExecutionBudget {
    let mut budget = ExecutionBudget::unlimited();
    if let Some(value) = options.max_parallel_tasks {
        budget = budget.with_max_parallel_tasks(value.max(1));
    }
    if let Some(value) = options.memory_budget_bytes {
        budget = budget.with_memory_budget_bytes(value);
    }
    budget
}

fn parse_profile(value: &str) -> Result<WireProfile, HostError> {
    match value {
        "turbojev" => Ok(WireProfile::TurboJev),
        "typesafe-systemone-v1" | "typesafe" | "systemone" => Ok(WireProfile::TypeSafeSystemOne),
        "ai-sdk-evaluation-v4" | "ai-sdk" => Ok(WireProfile::AiSdkEvaluationV4),
        other => Err(HostError::InvalidOptions(format!(
            "unknown compatibility profile: {other}"
        ))),
    }
}

/// Stable host-layer error category.
#[derive(Debug, Error)]
pub enum HostError {
    #[error("{cause}\nTURBOJEV_USAGE: {}", serde_json::to_string(usage).unwrap_or_default())]
    Generation {
        cause: Box<HostError>,
        usage: turbojev_schema::Usage,
    },
    #[error("generation cancelled")]
    GenerationCancelled,
    #[error("generation deadline exceeded")]
    GenerationTimeout,
    #[error("invalid native host options: {0}")]
    InvalidOptions(String),
    #[error("runtime unavailable: {0}")]
    RuntimeUnavailable(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("wire error: {0}")]
    Wire(String),
    #[error("compatibility error: {0}")]
    Compatibility(String),
    #[error("TurboJev core error: {0}")]
    Core(String),
    #[error("unsupported native execution capability: {0}")]
    UnsupportedCapability(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl HostError {
    /// Stable machine-readable error code shared by host bindings.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Generation { cause, .. } => cause.code(),
            Self::GenerationCancelled => "GENERATION_CANCELLED",
            Self::GenerationTimeout => "GENERATION_TIMEOUT",
            Self::InvalidOptions(_) => "INVALID_OPTIONS",
            Self::RuntimeUnavailable(_) => "RUNTIME_UNAVAILABLE",
            Self::Runtime(_) => "RUNTIME_ERROR",
            Self::Wire(_) => "WIRE_ERROR",
            Self::Compatibility(_) => "COMPATIBILITY_ERROR",
            Self::Core(_) => "CORE_ERROR",
            Self::UnsupportedCapability(_) => "UNSUPPORTED_CAPABILITY",
            Self::Json(_) => "JSON_ERROR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_projector_option_is_parsed_from_binding_json() {
        let options: NativeLoadOptions =
            serde_json::from_str(r#"{"model":"model.gguf","mmproj":"projector.gguf"}"#).unwrap();
        assert_eq!(options.mmproj.as_deref(), Some("projector.gguf"));
    }

    #[test]
    fn fixture_session_uses_the_real_core() {
        let session = NativeSession::open(NativeLoadOptions {
            worker: None,
            model: "fixture://host".into(),
            thinking: false,
            max_thinking_tokens: 512,
            mmproj: None,
            n_ctx: 64,
            n_batch: 64,
            n_seq_max: 2,
            n_gpu_layers: 0,
            device: None,
            n_threads: 0,
            n_threads_batch: 0,
            execution: HostExecutionOptions {
                order_robustness: HostOrderRobustness::None,
                ..HostExecutionOptions::default()
            },
            prompt: HostPromptProfile::default(),
            fixture_logits: BTreeMap::from([("route".into(), vec![0.0, 3.0])]),
        })
        .unwrap();
        let output = session.evaluate_json(r#"{"state":"lost","questions":{"route":{"type":"choice","instructions":"route","criteria":{"billing":"billing","shipping":"shipping"}}}}"#).unwrap();
        let value: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["answers"]["route"]["choice"], "shipping");
        let info: serde_json::Value = serde_json::from_str(&session.info_json().unwrap()).unwrap();
        assert_eq!(info["modalities"], serde_json::json!(["text"]));
        let caps = session.execution_capabilities();
        assert!(caps.fused_batch);
        assert!(!caps.cooperative_cancellation);
        assert!(!caps.hard_deadlines);
        assert_eq!(caps.max_fused_items, Some(64));
        assert_eq!(caps.estimated_fused_item_bytes, None);
    }

    #[test]
    fn fixture_fused_bridge_matches_independent_reference() {
        let session = NativeSession::open(NativeLoadOptions {
            worker: None,
            model: "fixture://host-fused".into(),
            thinking: false,
            max_thinking_tokens: 512,
            mmproj: None,
            n_ctx: 64,
            n_batch: 64,
            n_seq_max: 2,
            n_gpu_layers: 0,
            device: None,
            n_threads: 0,
            n_threads_batch: 0,
            execution: HostExecutionOptions {
                order_robustness: HostOrderRobustness::None,
                ..HostExecutionOptions::default()
            },
            prompt: HostPromptProfile::default(),
            fixture_logits: BTreeMap::from([("route".into(), vec![0.0, 3.0])]),
        })
        .unwrap();
        let make_request = |state: &str| -> EvaluationRequest {
            serde_json::from_value(serde_json::json!({
                "state": state,
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "route",
                        "criteria": {"billing": "billing", "shipping": "shipping"}
                    }
                }
            }))
            .unwrap()
        };
        let requests = vec![make_request("one"), make_request("two")];
        let reference: Vec<_> = requests
            .iter()
            .cloned()
            .map(|request| session.evaluate(request).unwrap())
            .collect();
        let fused = session.evaluate_fused(&requests).unwrap();
        assert_eq!(fused, reference);
    }

    #[test]
    fn llamacpp_multi_root_is_conservative_opt_in() {
        let options: NativeLoadOptions = serde_json::from_str(r#"{"model":"model.gguf"}"#).unwrap();
        assert!(!options.execution.llamacpp_multi_root_fused);
        assert_eq!(
            options.execution.order_robustness,
            HostOrderRobustness::None
        );

        let enabled: NativeLoadOptions = serde_json::from_str(
            r#"{"model":"model.gguf","execution":{"llamacppMultiRootFused":true}}"#,
        )
        .unwrap();
        assert!(enabled.execution.llamacpp_multi_root_fused);

        let corrected: NativeLoadOptions = serde_json::from_str(
            r#"{"model":"model.gguf","execution":{"orderRobustness":"cyclic"}}"#,
        )
        .unwrap();
        assert_eq!(
            corrected.execution.order_robustness,
            HostOrderRobustness::Cyclic
        );

        let fast: NativeLoadOptions = serde_json::from_str(
            r#"{"model":"model.gguf","execution":{"orderRobustness":"none"}}"#,
        )
        .unwrap();
        assert_eq!(fast.execution.order_robustness, HostOrderRobustness::None);
    }

    #[test]
    fn no_runtime_build_reports_a_stable_error() {
        #[cfg(not(feature = "llamacpp"))]
        {
            let err = NativeSession::open_json(r#"{"model":"/tmp/model.gguf"}"#)
                .err()
                .expect("opening without a runtime must fail");
            assert_eq!(err.code(), "RUNTIME_UNAVAILABLE");
        }
    }

    #[test]
    fn cpu_is_default_and_gpu_is_explicit() {
        let cpu: NativeLoadOptions = serde_json::from_str(r#"{"model":"fixture://cpu"}"#).unwrap();
        assert_eq!(cpu.device, None);
        assert_eq!(cpu.n_gpu_layers, 0);
        #[cfg(feature = "llamacpp")]
        assert_eq!(cpu.effective_gpu_layers(), 0);

        let gpu: NativeLoadOptions =
            serde_json::from_str(r#"{"model":"fixture://gpu","device":"gpu"}"#).unwrap();
        #[cfg(feature = "llamacpp")]
        assert_eq!(gpu.effective_gpu_layers(), -1);
        let _ = gpu;
    }

    #[test]
    fn explicit_cpu_rejects_gpu_layer_override() {
        let options: NativeLoadOptions =
            serde_json::from_str(r#"{"model":"fixture://conflict","device":"cpu","nGpuLayers":4}"#)
                .unwrap();
        assert!(options.validate().is_err());
    }
    #[test]
    fn thinking_defaults_off_and_rejects_invalid_budgets_and_fixture_claims() {
        let mut options: NativeLoadOptions =
            serde_json::from_str(r#"{"model":"qwen.gguf"}"#).unwrap();
        assert!(!options.thinking);
        assert_eq!(options.max_thinking_tokens, 512);
        options.thinking = true;
        options.max_thinking_tokens = 0;
        assert!(options.validate().is_err());
        options.max_thinking_tokens = options.n_ctx;
        assert!(options.validate().is_err());
        options.max_thinking_tokens = 128;
        options.model = "fixture://test".into();
        assert!(options.validate().is_err());
    }
}
