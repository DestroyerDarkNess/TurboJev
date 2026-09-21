//! FFI binding to TurboJev's stable llama.cpp shim.
//!
//! Unsafe code is intentionally isolated in this module. The shim ABI is ours;
//! llama.cpp API changes are absorbed by `native/turbojev_llama_shim.cpp`
//! instead of leaking through the Rust public API.

use std::{
    ffi::{c_char, c_float, c_int, c_void, CStr, CString},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

use turbojev_runtime::TokenId;

use crate::{LlamaCppApi, LlamaCppError, LlamaCppRootTask, LlamaCppSuffixTask};

const SHIM_ABI_VERSION: u32 = 5;
const ERROR_CAPACITY: usize = 2048;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NativeConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    /// Maximum logical sequences in the llama.cpp context. Sequence 0 is the
    /// immutable shared prefix; remaining sequences are parallel task branches.
    pub n_seq_max: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_batch: 512,
            n_seq_max: 8,
            n_gpu_layers: -1,
            n_threads: 0,
            n_threads_batch: 0,
        }
    }
}

#[cfg(feature = "ffi")]
#[link(name = "turbojev_llama_shim")]
extern "C" {
    fn tj_llama_abi_version() -> u32;
    fn tj_llama_open(
        model_path: *const c_char,
        config: *const NativeConfig,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    fn tj_llama_close(runtime: *mut c_void);
    fn tj_llama_model_desc(
        runtime: *mut c_void,
        out: *mut c_char,
        capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i64;
    fn tj_llama_encode_chat(
        runtime: *mut c_void,
        system: *const c_char,
        user: *const c_char,
        answer: *const c_char,
        out_tokens: *mut u32,
        capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i64;
    fn tj_llama_decode_tokens(
        runtime: *mut c_void,
        tokens: *const u32,
        token_count: usize,
        out_text: *mut c_char,
        capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i64;
    fn tj_llama_score_next(
        runtime: *mut c_void,
        prompt_tokens: *const u32,
        prompt_count: usize,
        candidate_tokens: *const u32,
        candidate_count: usize,
        out_logits: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_sequence_scores_supported(runtime: *mut c_void) -> c_int;
    fn tj_llama_score_sequences(
        runtime: *mut c_void,
        prompt_tokens: *const u32,
        prompt_count: usize,
        candidate_tokens: *const u32,
        candidate_offsets: *const usize,
        candidate_count: usize,
        out_scores: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_multi_root_supported(runtime: *mut c_void) -> c_int;
    fn tj_llama_max_multi_root_tasks(runtime: *mut c_void) -> usize;
    fn tj_llama_max_multi_root_tokens(runtime: *mut c_void) -> usize;
    fn tj_llama_score_root_batch(
        runtime: *mut c_void,
        prompt_tokens: *const u32,
        prompt_offsets: *const usize,
        candidate_tokens: *const u32,
        candidate_offsets: *const usize,
        task_count: usize,
        out_logits: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_shared_prefix_supported(runtime: *mut c_void) -> c_int;
    fn tj_llama_parallel_suffix_supported(runtime: *mut c_void) -> c_int;
    fn tj_llama_max_parallel_tasks(runtime: *mut c_void) -> usize;
    fn tj_llama_max_parallel_suffix_tokens(runtime: *mut c_void) -> usize;
    fn tj_llama_prefill_shared(
        runtime: *mut c_void,
        prefix_tokens: *const u32,
        prefix_count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_score_suffix(
        runtime: *mut c_void,
        suffix_tokens: *const u32,
        suffix_count: usize,
        candidate_tokens: *const u32,
        candidate_count: usize,
        out_logits: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_score_suffix_sequences(
        runtime: *mut c_void,
        suffix_tokens: *const u32,
        suffix_count: usize,
        candidate_tokens: *const u32,
        candidate_offsets: *const usize,
        candidate_count: usize,
        out_scores: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn tj_llama_score_suffix_batch(
        runtime: *mut c_void,
        suffix_tokens: *const u32,
        suffix_offsets: *const usize,
        candidate_tokens: *const u32,
        candidate_offsets: *const usize,
        task_count: usize,
        out_logits: *mut c_float,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

/// Loaded llama.cpp model through the stable TurboJev shim.
///
/// This handle is intentionally thread-affine because one native llama_context
/// owns both the independent reference path and the reusable prefix sequence.
/// Hosts that need concurrency should create independent sessions or add an
/// explicit synchronization layer outside the portable TurboJev core.
#[cfg(feature = "ffi")]
pub struct FfiLlamaCpp {
    handle: NonNull<c_void>,
    model_id: String,
    profile: String,
    max_parallel_tasks: Option<usize>,
    estimated_parallel_task_bytes: Option<u64>,
    max_multi_root_tasks: Option<usize>,
    multi_root_enabled: bool,
    estimated_multi_root_item_bytes: Option<u64>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[cfg(feature = "ffi")]
impl FfiLlamaCpp {
    /// Open one GGUF model using the native shim.
    pub fn open(model_path: &str, config: NativeConfig) -> Result<Self, LlamaCppError> {
        let path = CString::new(model_path)
            .map_err(|_| LlamaCppError::InvalidInput("model path contains NUL".into()))?;
        let mut error = vec![0 as c_char; ERROR_CAPACITY];

        let abi = unsafe { tj_llama_abi_version() };
        if abi != SHIM_ABI_VERSION {
            return Err(LlamaCppError::Native(format!(
                "shim ABI mismatch: Rust expects {SHIM_ABI_VERSION}, native reports {abi}"
            )));
        }

        let raw = unsafe {
            tj_llama_open(
                path.as_ptr(),
                &config,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(raw).ok_or_else(|| native_error(&error, "failed to open model"))?;
        let native_parallel_tasks = unsafe { tj_llama_max_parallel_tasks(handle.as_ptr()) };
        let native_multi_root_tasks = unsafe { tj_llama_max_multi_root_tasks(handle.as_ptr()) };
        let mut runtime = Self {
            handle,
            model_id: String::new(),
            profile: "llamacpp-model-template-v1".into(),
            max_parallel_tasks: (native_parallel_tasks > 0).then_some(native_parallel_tasks),
            estimated_parallel_task_bytes: None,
            max_multi_root_tasks: (native_multi_root_tasks > 0).then_some(native_multi_root_tasks),
            multi_root_enabled: false,
            estimated_multi_root_item_bytes: None,
            _not_send_sync: PhantomData,
        };
        runtime.model_id = runtime.read_model_desc()?;
        Ok(runtime)
    }

    /// Supply a conservative incremental memory estimate for one parallel task
    /// branch. The portable planner uses it only when callers also provide a
    /// memory budget; zero clears the estimate.
    pub fn with_estimated_parallel_task_bytes(mut self, bytes: u64) -> Self {
        self.estimated_parallel_task_bytes = (bytes > 0).then_some(bytes);
        self
    }


    /// Enable the experimental llama.cpp multi-root fused path for this loaded
    /// runtime. It is disabled by default because conformance must be proven on
    /// the concrete llama.cpp build/backend/hardware combination before a host
    /// advertises cross-request fused execution.
    pub fn with_multi_root_enabled(mut self, enabled: bool) -> Self {
        self.multi_root_enabled = enabled;
        self
    }

    /// Supply a conservative per-request memory estimate for multi-root fused
    /// scheduling. Zero clears the estimate. This is host policy data; the shim
    /// itself does not guess memory use from model architecture.
    pub fn with_estimated_multi_root_item_bytes(mut self, bytes: u64) -> Self {
        self.estimated_multi_root_item_bytes = (bytes > 0).then_some(bytes);
        self
    }

    fn read_model_desc(&mut self) -> Result<String, LlamaCppError> {
        let required = self.call_text(|out, cap, error, error_cap| unsafe {
            tj_llama_model_desc(self.handle.as_ptr(), out, cap, error, error_cap)
        })?;
        if required.is_empty() {
            Ok("llama.cpp GGUF".into())
        } else {
            Ok(required)
        }
    }

    fn call_text<F>(&self, mut call: F) -> Result<String, LlamaCppError>
    where
        F: FnMut(*mut c_char, usize, *mut c_char, usize) -> i64,
    {
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let needed = call(std::ptr::null_mut(), 0, error.as_mut_ptr(), error.len());
        if needed < 0 {
            return Err(native_error(&error, "native text query failed"));
        }
        let mut bytes = vec![0u8; needed as usize + 1];
        let written = call(
            bytes.as_mut_ptr().cast(),
            bytes.len(),
            error.as_mut_ptr(),
            error.len(),
        );
        if written < 0 {
            return Err(native_error(&error, "native text query failed"));
        }
        bytes.truncate(written as usize);
        String::from_utf8(bytes)
            .map_err(|error| LlamaCppError::Native(format!("native UTF-8 error: {error}")))
    }

    fn encode_impl(
        &self,
        system: Option<&str>,
        user: &str,
        answer: Option<&str>,
    ) -> Result<Vec<TokenId>, LlamaCppError> {
        let system = optional_c_string(system)?;
        let user = CString::new(user)
            .map_err(|_| LlamaCppError::InvalidInput("user prompt contains NUL".into()))?;
        let answer = optional_c_string(answer)?;
        let mut error = vec![0 as c_char; ERROR_CAPACITY];

        let needed = unsafe {
            tj_llama_encode_chat(
                self.handle.as_ptr(),
                system.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                user.as_ptr(),
                answer.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                std::ptr::null_mut(),
                0,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if needed < 0 {
            return Err(native_error(&error, "chat tokenization failed"));
        }
        let mut tokens = vec![0u32; needed as usize];
        let written = unsafe {
            tj_llama_encode_chat(
                self.handle.as_ptr(),
                system.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                user.as_ptr(),
                answer.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                tokens.as_mut_ptr(),
                tokens.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if written < 0 {
            return Err(native_error(&error, "chat tokenization failed"));
        }
        tokens.truncate(written as usize);
        Ok(tokens)
    }
}

#[cfg(feature = "ffi")]
impl LlamaCppApi for FfiLlamaCpp {
    fn model_id(&self) -> &str {
        &self.model_id
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
        self.encode_impl(system, user, answer)
    }

    fn decode_tokens(&self, tokens: &[TokenId]) -> Result<String, LlamaCppError> {
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let needed = unsafe {
            tj_llama_decode_tokens(
                self.handle.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                std::ptr::null_mut(),
                0,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if needed < 0 {
            return Err(native_error(&error, "token decode failed"));
        }
        let mut bytes = vec![0u8; needed as usize + 1];
        let written = unsafe {
            tj_llama_decode_tokens(
                self.handle.as_ptr(),
                tokens.as_ptr(),
                tokens.len(),
                bytes.as_mut_ptr().cast(),
                bytes.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if written < 0 {
            return Err(native_error(&error, "token decode failed"));
        }
        bytes.truncate(written as usize);
        String::from_utf8(bytes)
            .map_err(|error| LlamaCppError::Tokenization(format!("decoded bytes are not UTF-8: {error}")))
    }

    fn score_next_token(
        &self,
        prompt_tokens: &[TokenId],
        candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError> {
        if prompt_tokens.is_empty() {
            return Err(LlamaCppError::InvalidInput("prompt tokens are empty".into()));
        }
        if candidate_token_ids.len() < 2 {
            return Err(LlamaCppError::InvalidInput(
                "at least two candidate tokens are required".into(),
            ));
        }
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let mut logits = vec![0.0f32; candidate_token_ids.len()];
        let status = unsafe {
            tj_llama_score_next(
                self.handle.as_ptr(),
                prompt_tokens.as_ptr(),
                prompt_tokens.len(),
                candidate_token_ids.as_ptr(),
                candidate_token_ids.len(),
                logits.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "next-token scoring failed"));
        }
        Ok(logits)
    }

    fn sequence_scores_supported(&self) -> bool {
        unsafe { tj_llama_sequence_scores_supported(self.handle.as_ptr()) != 0 }
    }

    fn score_sequences(
        &self,
        prompt_tokens: &[TokenId],
        candidate_sequences: &[Vec<TokenId>],
    ) -> Result<Vec<f32>, LlamaCppError> {
        let (flat, offsets) = flatten_sequences(candidate_sequences)?;
        let mut scores = vec![0.0f32; candidate_sequences.len()];
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            tj_llama_score_sequences(
                self.handle.as_ptr(),
                prompt_tokens.as_ptr(),
                prompt_tokens.len(),
                flat.as_ptr(),
                offsets.as_ptr(),
                candidate_sequences.len(),
                scores.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "sequence scoring failed"));
        }
        Ok(scores)
    }

    fn multi_root_supported(&self) -> bool {
        self.multi_root_enabled
            && unsafe { tj_llama_multi_root_supported(self.handle.as_ptr()) != 0 }
    }

    fn max_multi_root_tasks(&self) -> Option<usize> {
        self.multi_root_enabled.then_some(self.max_multi_root_tasks).flatten()
    }

    fn max_multi_root_tokens(&self) -> Option<usize> {
        if !self.multi_root_enabled {
            return None;
        }
        let value = unsafe { tj_llama_max_multi_root_tokens(self.handle.as_ptr()) };
        (value > 0).then_some(value)
    }

    fn estimated_multi_root_item_bytes(&self) -> Option<u64> {
        self.multi_root_enabled
            .then_some(self.estimated_multi_root_item_bytes)
            .flatten()
    }

    fn score_multi_root_next_token(
        &self,
        tasks: &[LlamaCppRootTask<'_>],
    ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
        if !self.multi_root_enabled {
            return Err(LlamaCppError::Unsupported(
                "llama.cpp multi-root fused scoring is disabled until explicitly enabled after target-specific conformance".to_owned(),
            ));
        }
        if tasks.len() < 2 {
            return Err(LlamaCppError::InvalidInput(
                "multi-root scoring requires at least two roots".to_owned(),
            ));
        }
        if let Some(maximum) = self.max_multi_root_tasks {
            if tasks.len() > maximum {
                return Err(LlamaCppError::InvalidInput(format!(
                    "multi-root batch has {} roots but runtime limit is {maximum}",
                    tasks.len()
                )));
            }
        }

        let mut prompt_tokens = Vec::new();
        let mut prompt_offsets = Vec::with_capacity(tasks.len() + 1);
        prompt_offsets.push(0usize);
        let mut candidate_tokens = Vec::new();
        let mut candidate_offsets = Vec::with_capacity(tasks.len() + 1);
        candidate_offsets.push(0usize);
        for task in tasks {
            if task.prompt_tokens.is_empty() {
                return Err(LlamaCppError::InvalidInput(
                    "multi-root prompts must not be empty".to_owned(),
                ));
            }
            if task.candidate_token_ids.len() < 2 {
                return Err(LlamaCppError::InvalidInput(
                    "each multi-root task requires at least two candidates".to_owned(),
                ));
            }
            prompt_tokens.extend_from_slice(task.prompt_tokens);
            prompt_offsets.push(prompt_tokens.len());
            candidate_tokens.extend_from_slice(task.candidate_token_ids);
            candidate_offsets.push(candidate_tokens.len());
        }
        if let Some(maximum) = self.max_multi_root_tokens() {
            if prompt_tokens.len() > maximum {
                return Err(LlamaCppError::InvalidInput(format!(
                    "multi-root batch has {} prompt tokens but runtime limit is {maximum}",
                    prompt_tokens.len()
                )));
            }
        }

        let mut flat_logits = vec![0.0f32; candidate_tokens.len()];
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            tj_llama_score_root_batch(
                self.handle.as_ptr(),
                prompt_tokens.as_ptr(),
                prompt_offsets.as_ptr(),
                candidate_tokens.as_ptr(),
                candidate_offsets.as_ptr(),
                tasks.len(),
                flat_logits.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "multi-root scoring failed"));
        }
        let mut rows = Vec::with_capacity(tasks.len());
        for window in candidate_offsets.windows(2) {
            rows.push(flat_logits[window[0]..window[1]].to_vec());
        }
        Ok(rows)
    }

    fn shared_prefix_supported(&self) -> bool {
        unsafe { tj_llama_shared_prefix_supported(self.handle.as_ptr()) != 0 }
    }

    fn parallel_suffix_supported(&self) -> bool {
        unsafe { tj_llama_parallel_suffix_supported(self.handle.as_ptr()) != 0 }
    }

    fn max_parallel_tasks(&self) -> Option<usize> {
        self.max_parallel_tasks
    }

    fn estimated_parallel_task_bytes(&self) -> Option<u64> {
        self.estimated_parallel_task_bytes
    }

    fn max_parallel_suffix_tokens(&self) -> Option<usize> {
        let value = unsafe { tj_llama_max_parallel_suffix_tokens(self.handle.as_ptr()) };
        (value > 0).then_some(value)
    }

    fn prefill_shared_prefix(&self, prefix_tokens: &[TokenId]) -> Result<(), LlamaCppError> {
        if prefix_tokens.is_empty() {
            return Err(LlamaCppError::InvalidInput(
                "shared prefix tokens are empty".to_owned(),
            ));
        }
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            tj_llama_prefill_shared(
                self.handle.as_ptr(),
                prefix_tokens.as_ptr(),
                prefix_tokens.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "shared-prefix prefill failed"));
        }
        Ok(())
    }

    fn score_suffix_next_token(
        &self,
        suffix_tokens: &[TokenId],
        candidate_token_ids: &[TokenId],
    ) -> Result<Vec<f32>, LlamaCppError> {
        if candidate_token_ids.len() < 2 {
            return Err(LlamaCppError::InvalidInput(
                "at least two candidate tokens are required".to_owned(),
            ));
        }
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let mut logits = vec![0.0f32; candidate_token_ids.len()];
        let suffix_ptr = if suffix_tokens.is_empty() {
            std::ptr::null()
        } else {
            suffix_tokens.as_ptr()
        };
        let status = unsafe {
            tj_llama_score_suffix(
                self.handle.as_ptr(),
                suffix_ptr,
                suffix_tokens.len(),
                candidate_token_ids.as_ptr(),
                candidate_token_ids.len(),
                logits.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "shared-suffix scoring failed"));
        }
        Ok(logits)
    }

    fn score_suffix_sequences(
        &self,
        suffix_tokens: &[TokenId],
        candidate_sequences: &[Vec<TokenId>],
    ) -> Result<Vec<f32>, LlamaCppError> {
        let (flat, offsets) = flatten_sequences(candidate_sequences)?;
        let mut scores = vec![0.0f32; candidate_sequences.len()];
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let suffix_ptr = if suffix_tokens.is_empty() { std::ptr::null() } else { suffix_tokens.as_ptr() };
        let status = unsafe {
            tj_llama_score_suffix_sequences(
                self.handle.as_ptr(),
                suffix_ptr,
                suffix_tokens.len(),
                flat.as_ptr(),
                offsets.as_ptr(),
                candidate_sequences.len(),
                scores.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "shared-prefix sequence scoring failed"));
        }
        Ok(scores)
    }

    fn score_suffix_batch_next_token(
        &self,
        tasks: &[LlamaCppSuffixTask<'_>],
    ) -> Result<Vec<Vec<f32>>, LlamaCppError> {
        if tasks.len() < 2 {
            return Err(LlamaCppError::InvalidInput(
                "parallel suffix scoring requires at least two tasks".to_owned(),
            ));
        }
        if let Some(maximum) = self.max_parallel_tasks {
            if tasks.len() > maximum {
                return Err(LlamaCppError::InvalidInput(format!(
                    "parallel suffix batch has {} tasks but runtime limit is {maximum}",
                    tasks.len()
                )));
            }
        }

        let mut suffix_tokens = Vec::new();
        let mut suffix_offsets = Vec::with_capacity(tasks.len() + 1);
        suffix_offsets.push(0usize);
        let mut candidate_tokens = Vec::new();
        let mut candidate_offsets = Vec::with_capacity(tasks.len() + 1);
        candidate_offsets.push(0usize);

        for task in tasks {
            if task.candidate_token_ids.len() < 2 {
                return Err(LlamaCppError::InvalidInput(
                    "each parallel suffix task requires at least two candidates".to_owned(),
                ));
            }
            suffix_tokens.extend_from_slice(task.suffix_tokens);
            suffix_offsets.push(suffix_tokens.len());
            candidate_tokens.extend_from_slice(task.candidate_token_ids);
            candidate_offsets.push(candidate_tokens.len());
        }

        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let mut flat_logits = vec![0.0f32; candidate_tokens.len()];
        let suffix_ptr = if suffix_tokens.is_empty() {
            std::ptr::null()
        } else {
            suffix_tokens.as_ptr()
        };
        let status = unsafe {
            tj_llama_score_suffix_batch(
                self.handle.as_ptr(),
                suffix_ptr,
                suffix_offsets.as_ptr(),
                candidate_tokens.as_ptr(),
                candidate_offsets.as_ptr(),
                tasks.len(),
                flat_logits.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(&error, "parallel shared-suffix scoring failed"));
        }

        let mut rows = Vec::with_capacity(tasks.len());
        for window in candidate_offsets.windows(2) {
            rows.push(flat_logits[window[0]..window[1]].to_vec());
        }
        Ok(rows)
    }
}

#[cfg(feature = "ffi")]
impl Drop for FfiLlamaCpp {
    fn drop(&mut self) {
        unsafe { tj_llama_close(self.handle.as_ptr()) };
    }
}

fn flatten_sequences(
    sequences: &[Vec<TokenId>],
) -> Result<(Vec<TokenId>, Vec<usize>), LlamaCppError> {
    if sequences.len() < 2 {
        return Err(LlamaCppError::InvalidInput(
            "at least two candidate sequences are required".to_owned(),
        ));
    }
    let mut flat = Vec::new();
    let mut offsets = Vec::with_capacity(sequences.len() + 1);
    offsets.push(0);
    for sequence in sequences {
        if sequence.is_empty() {
            return Err(LlamaCppError::InvalidInput(
                "candidate sequences must not be empty".to_owned(),
            ));
        }
        flat.extend_from_slice(sequence);
        offsets.push(flat.len());
    }
    Ok((flat, offsets))
}

fn optional_c_string(value: Option<&str>) -> Result<Option<CString>, LlamaCppError> {
    value
        .map(|value| {
            CString::new(value)
                .map_err(|_| LlamaCppError::InvalidInput("text contains NUL".into()))
        })
        .transpose()
}

fn native_error(buffer: &[c_char], fallback: &str) -> LlamaCppError {
    if buffer.is_empty() || buffer[0] == 0 {
        return LlamaCppError::Native(fallback.to_owned());
    }
    let ptr = buffer.as_ptr();
    let message = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    if message.is_empty() {
        LlamaCppError::Native(fallback.to_owned())
    } else {
        LlamaCppError::Native(message)
    }
}
