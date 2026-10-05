#ifndef TURBOJEV_LLAMA_TEST_STUB_H
#define TURBOJEV_LLAMA_TEST_STUB_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;
typedef struct llama_model llama_model;
typedef struct llama_context llama_context;
typedef struct llama_vocab llama_vocab;
typedef struct llama_sampler llama_sampler;
typedef struct llama_token_data { llama_token id; float logit; float p; } llama_token_data;
typedef struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
    int64_t selected;
    bool sorted;
} llama_token_data_array;
typedef struct llama_memory_i * llama_memory_t;
typedef struct llama_model_params { int32_t n_gpu_layers; } llama_model_params;
typedef struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_seq_max;
    bool no_perf;
    bool offload_kqv;
    bool op_offload;
} llama_context_params;
typedef struct llama_chat_message { const char * role; const char * content; } llama_chat_message;
typedef struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
} llama_batch;

llama_model_params llama_model_default_params(void);
llama_context_params llama_context_default_params(void);
void llama_backend_init(void);
llama_model * llama_model_load_from_file(const char *, llama_model_params);
void llama_model_free(llama_model *);
const llama_vocab * llama_model_get_vocab(const llama_model *);
const char * llama_model_chat_template(const llama_model *, const char *);
llama_context * llama_init_from_model(llama_model *, llama_context_params);
void llama_free(llama_context *);
void llama_set_n_threads(llama_context *, int32_t, int32_t);
uint32_t llama_n_ctx(const llama_context *);
uint32_t llama_n_ctx_seq(const llama_context *);
uint32_t llama_n_batch(const llama_context *);
uint32_t llama_n_seq_max(const llama_context *);
int32_t llama_chat_apply_template(const char *, const llama_chat_message *, size_t, bool, char *, int32_t);
int32_t llama_tokenize(const llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool);
int32_t llama_detokenize(const llama_vocab *, const llama_token *, int32_t, char *, int32_t, bool, bool);
int32_t llama_model_desc(const llama_model *, char *, size_t);
llama_memory_t llama_get_memory(const llama_context *);
void llama_memory_clear(llama_memory_t, bool);
bool llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos, llama_pos);
void llama_memory_seq_cp(llama_memory_t, llama_seq_id, llama_seq_id, llama_pos, llama_pos);
llama_batch llama_batch_get_one(llama_token *, int32_t);
llama_batch llama_batch_init(int32_t, int32_t, int32_t);
void llama_batch_free(llama_batch);
int32_t llama_decode(llama_context *, llama_batch);
float * llama_get_logits_ith(llama_context *, int32_t);
int32_t llama_vocab_n_tokens(const llama_vocab *);
bool llama_vocab_is_eog(const llama_vocab *, llama_token);
llama_sampler * llama_sampler_init_grammar(const llama_vocab *, const char *, const char *);
void llama_sampler_apply(llama_sampler *, llama_token_data_array *);
void llama_sampler_accept(llama_sampler *, llama_token);
void llama_sampler_free(llama_sampler *);
#ifdef __cplusplus
}
#endif
#endif
