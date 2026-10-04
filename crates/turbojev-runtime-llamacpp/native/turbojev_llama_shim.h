#ifndef TURBOJEV_LLAMA_SHIM_H
#define TURBOJEV_LLAMA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TURBOJEV_LLAMA_SHIM_ABI_VERSION 9u

typedef struct tj_llama_runtime tj_llama_runtime;

/* Text-only, bounded model reasoning followed by exact candidate logits.
 * Fails on unsupported templates or an unfinished reasoning budget.
 * Candidate tokens are verified again at the actual final-answer boundary. */
int32_t tj_llama_score_thinking_next(
    tj_llama_runtime * runtime, const char * system, const char * user,
    const char * assistant_prefix, const uint32_t * candidate_tokens,
    size_t candidate_count, uint32_t max_thinking_tokens, float * out_logits,
    size_t * generated_tokens, char * error, size_t error_capacity);

typedef struct tj_llama_config {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_seq_max;
    int32_t n_gpu_layers;
    int32_t n_threads;
    int32_t n_threads_batch;
} tj_llama_config;

uint32_t tj_llama_abi_version(void);

tj_llama_runtime * tj_llama_open(
    const char * model_path,
    const tj_llama_config * config,
    char * error,
    size_t error_capacity);
tj_llama_runtime * tj_llama_open_multimodal(
    const char * model_path,
    const char * mmproj_path,
    const tj_llama_config * config,
    char * error,
    size_t error_capacity);
int32_t tj_llama_supports_image(const tj_llama_runtime * runtime);
int32_t tj_llama_supports_audio(const tj_llama_runtime * runtime);
int32_t tj_llama_supports_video(const tj_llama_runtime * runtime);
int32_t tj_llama_score_media_next(
    tj_llama_runtime * runtime,
    const char * system,
    const char * user,
    const char * assistant_prefix,
    int32_t modality,
    const char * media_base64,
    size_t media_base64_len,
    const char * const * frame_base64,
    const size_t * frame_base64_lengths,
    const uint64_t * frame_timestamps_ms,
    size_t frame_count,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    char * error,
    size_t error_capacity);

void tj_llama_close(tj_llama_runtime * runtime);

int64_t tj_llama_model_desc(
    tj_llama_runtime * runtime,
    char * out,
    size_t capacity,
    char * error,
    size_t error_capacity);

int64_t tj_llama_encode_chat(
    tj_llama_runtime * runtime,
    const char * system,
    const char * user,
    const char * answer,
    uint32_t * out_tokens,
    size_t capacity,
    char * error,
    size_t error_capacity);

int64_t tj_llama_decode_tokens(
    tj_llama_runtime * runtime,
    const uint32_t * tokens,
    size_t token_count,
    char * out_text,
    size_t capacity,
    char * error,
    size_t error_capacity);

int32_t tj_llama_score_next(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    size_t prompt_count,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    char * error,
    size_t error_capacity);

// Returns 1 when exact prefix-free multi-token candidate scoring is available.
int32_t tj_llama_sequence_scores_supported(tj_llama_runtime * runtime);

// Scores candidate token sequences with exact conditional log-likelihoods.
// candidate_offsets has candidate_count + 1 entries and delimits candidate_tokens.
int32_t tj_llama_score_sequences(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    size_t prompt_count,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    float * out_scores,
    char * error,
    size_t error_capacity);

// Returns 1 when the loaded llama.cpp context exposes reusable sequence memory.
int32_t tj_llama_shared_prefix_supported(tj_llama_runtime * runtime);

// Returns 1 when at least two independent suffix branches can be evaluated in
// one decode call after a shared-prefix prefill.
int32_t tj_llama_parallel_suffix_supported(tj_llama_runtime * runtime);

// Maximum task branches accepted by tj_llama_score_suffix_batch. Sequence 0 is
// reserved for the immutable prefix and is therefore excluded from this count.
size_t tj_llama_max_parallel_tasks(tj_llama_runtime * runtime);

// Maximum total suffix tokens that can be submitted in one parallel decode
// after the current shared prefix has been prefetched. Returns 0 when unknown
// or when no shared prefix is ready.
size_t tj_llama_max_parallel_suffix_tokens(tj_llama_runtime * runtime);

// Clears the context and prefills sequence 0 with one reusable common prefix.
// The final-prefix logits are retained so an empty task suffix can be scored.
int32_t tj_llama_prefill_shared(
    tj_llama_runtime * runtime,
    const uint32_t * prefix_tokens,
    size_t prefix_count,
    char * error,
    size_t error_capacity);

// Clones the prefetched sequence 0 into an ephemeral sequence 1, evaluates one
// suffix and gathers next-token logits. The reusable prefix remains untouched.
int32_t tj_llama_score_suffix(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    size_t suffix_count,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    char * error,
    size_t error_capacity);

// Shared-prefix counterpart of tj_llama_score_sequences.
int32_t tj_llama_score_suffix_sequences(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    size_t suffix_count,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    float * out_scores,
    char * error,
    size_t error_capacity);

// Evaluates several isolated suffix branches in one llama_decode call.
// `suffix_offsets` and `candidate_offsets` each contain task_count + 1 entries,
// start at zero, are monotonic, and delimit the corresponding flat arrays.
// Output logits are written in the same flattened candidate order.

// Returns 1 when the context can evaluate independent prompt roots in one
// llama_decode call using isolated sequence IDs.
int32_t tj_llama_multi_root_supported(tj_llama_runtime * runtime);

// Maximum independent root sequences accepted by one multi-root submission.
size_t tj_llama_max_multi_root_tasks(tj_llama_runtime * runtime);

// Maximum total prompt tokens accepted by one multi-root submission. Returns
// zero when the runtime cannot expose a finite bound.
size_t tj_llama_max_multi_root_tokens(tj_llama_runtime * runtime);

// Evaluates independent full prompt roots in one llama_decode call.
// `prompt_offsets` and `candidate_offsets` contain task_count + 1 entries, start
// at zero, and delimit the corresponding flat arrays. Every root owns a unique
// llama.cpp sequence ID and positions start at zero independently. Output logits
// preserve task/candidate order.
int32_t tj_llama_score_root_batch(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    const size_t * prompt_offsets,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t task_count,
    float * out_logits,
    char * error,
    size_t error_capacity);

int32_t tj_llama_score_suffix_batch(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    const size_t * suffix_offsets,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t task_count,
    float * out_logits,
    char * error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
