#include "turbojev_llama_shim.h"
#include "decision_chat.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"

#include "llama.h"
#include "ggml-backend.h"
#if defined(TURBOJEV_HAS_MTMD)
#include "mtmd-helper.h"
#endif

#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct tj_llama_runtime {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
#if defined(TURBOJEV_HAS_MTMD)
    mtmd_context * mtmd = nullptr;
#endif
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr chat_templates;
    // Keep template date/time functions stable across prompt/codebook probes.
    std::chrono::system_clock::time_point chat_time = std::chrono::system_clock::now();
    std::string cached_system;
    std::string cached_user;
    std::string cached_prompt;
    bool prompt_cached = false;
    std::string field_result;
    size_t shared_prefix_len = 0;
    std::vector<float> shared_prefix_logits;
    bool shared_prefix_ready = false;
    uint32_t n_ctx = 0;
    uint32_t n_ctx_seq = 0;
    uint32_t n_batch = 0;
    uint32_t n_seq_max = 0;
};

namespace {

std::once_flag backend_once;

void init_backend_once() {
    std::call_once(backend_once, [] { llama_backend_init(); });
}

void set_error(char * error, size_t capacity, const std::string & message) {
    if (error == nullptr || capacity == 0) {
        return;
    }
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

bool checked_i32(size_t value, int32_t & out) {
    if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

#if defined(TURBOJEV_HAS_MTMD)
bool decode_base64(const char * input, size_t length, std::vector<unsigned char> & output) {
    if (input == nullptr || length == 0 || length > 32u * 1024u * 1024u || length % 4 != 0) {
        return false;
    }
    auto value = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    output.clear();
    output.reserve((length / 4) * 3);
    for (size_t i = 0; i < length; i += 4) {
        const bool pad2 = input[i + 2] == '=';
        const bool pad3 = input[i + 3] == '=';
        const int a = value(static_cast<unsigned char>(input[i]));
        const int b = value(static_cast<unsigned char>(input[i + 1]));
        const int c = pad2 ? 0 : value(static_cast<unsigned char>(input[i + 2]));
        const int d = pad3 ? 0 : value(static_cast<unsigned char>(input[i + 3]));
        const bool last = i + 4 == length;
        if (a < 0 || b < 0 || c < 0 || d < 0 || (!last && (pad2 || pad3))
            || (pad2 && !pad3)) {
            return false;
        }
        const uint32_t bits = (static_cast<uint32_t>(a) << 18)
            | (static_cast<uint32_t>(b) << 12)
            | (static_cast<uint32_t>(c) << 6) | static_cast<uint32_t>(d);
        output.push_back(static_cast<unsigned char>((bits >> 16) & 0xff));
        if (!pad2) output.push_back(static_cast<unsigned char>((bits >> 8) & 0xff));
        if (!pad3) output.push_back(static_cast<unsigned char>(bits & 0xff));
    }
    return !output.empty();
}
#endif

bool checked_pos(size_t value, llama_pos & out) {
    if (value > static_cast<size_t>(std::numeric_limits<llama_pos>::max())) {
        return false;
    }
    out = static_cast<llama_pos>(value);
    return true;
}

int64_t copy_text(const std::string & value, char * out, size_t capacity) {
    if (out == nullptr || capacity == 0) {
        return static_cast<int64_t>(value.size());
    }
    if (capacity <= value.size()) {
        return static_cast<int64_t>(value.size());
    }
    std::memcpy(out, value.data(), value.size());
    out[value.size()] = '\0';
    return static_cast<int64_t>(value.size());
}

bool render_chat(
    tj_llama_runtime * runtime,
    const char * system,
    const char * user,
    std::string & rendered,
    std::string & failure) {
    if (runtime == nullptr || runtime->model == nullptr || user == nullptr) {
        failure = "invalid runtime or user message";
        return false;
    }
    if (!runtime->chat_templates) {
        failure = "GGUF model does not expose a llama.cpp-supported chat template";
        return false;
    }

    const char * system_text = system == nullptr ? "" : system;
    if (runtime->prompt_cached && runtime->cached_system == system_text
        && runtime->cached_user == user) {
        rendered = runtime->cached_prompt;
        return true;
    }

    try {
        rendered = tj_render_decision_chat(
            runtime->chat_templates.get(), system, user, runtime->chat_time);
    } catch (const std::exception & exception) {
        failure = std::string("GGUF decision chat template failed: ") + exception.what();
        return false;
    }
    // Codebook probes render the same messages with different answer suffixes.
    // Reuse their exact prefix instead of re-running Jinja for every candidate.
    runtime->cached_system = system_text;
    runtime->cached_user = user;
    runtime->cached_prompt = rendered;
    runtime->prompt_cached = true;
    return true;
}

bool tokenize(
    tj_llama_runtime * runtime,
    const std::string & text,
    std::vector<llama_token> & tokens,
    std::string & failure) {
    int32_t text_len = 0;
    if (!checked_i32(text.size(), text_len)) {
        failure = "prompt exceeds llama.cpp int32 tokenization input limit";
        return false;
    }

    const int32_t needed = llama_tokenize(
        runtime->vocab, text.data(), text_len, nullptr, 0, true, true);
    if (needed == std::numeric_limits<int32_t>::min()) {
        failure = "llama_tokenize overflow";
        return false;
    }
    const int32_t count = needed < 0 ? -needed : needed;
    if (count <= 0) {
        failure = "llama_tokenize returned no tokens";
        return false;
    }
    tokens.resize(static_cast<size_t>(count));
    const int32_t written = llama_tokenize(
        runtime->vocab, text.data(), text_len, tokens.data(), count, true, true);
    if (written < 0 || written > count) {
        failure = "llama_tokenize failed";
        return false;
    }
    tokens.resize(static_cast<size_t>(written));
    return true;
}

bool copy_tokens(
    const uint32_t * source,
    size_t count,
    std::vector<llama_token> & out,
    std::string & failure) {
    if (source == nullptr && count != 0) {
        failure = "token pointer is null";
        return false;
    }
    if (count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        failure = "token sequence is too large for llama.cpp";
        return false;
    }
    out.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (source[i] > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            failure = "token id exceeds llama.cpp range";
            return false;
        }
        out[i] = static_cast<llama_token>(source[i]);
    }
    return true;
}

bool gather_candidate_logits(
    tj_llama_runtime * runtime,
    const float * logits,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    std::string & failure) {
    if (runtime == nullptr || runtime->vocab == nullptr || logits == nullptr
        || candidate_tokens == nullptr || candidate_count < 2 || out_logits == nullptr) {
        failure = "invalid candidate-logit arguments";
        return false;
    }
    const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
    if (vocab_size <= 0) {
        failure = "llama.cpp vocabulary is empty";
        return false;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidate_tokens[i] >= static_cast<uint32_t>(vocab_size)) {
            failure = "candidate token exceeds vocabulary size";
            return false;
        }
        out_logits[i] = logits[candidate_tokens[i]];
    }
    return true;
}


struct sequence_trie_node {
    std::map<uint32_t, size_t> children;
    int64_t terminal = -1;
};

bool build_sequence_trie(
    tj_llama_runtime * runtime,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    std::vector<sequence_trie_node> & nodes,
    std::string & failure) {
    if (runtime == nullptr || runtime->vocab == nullptr || candidate_tokens == nullptr
        || candidate_offsets == nullptr || candidate_count < 2 || candidate_offsets[0] != 0) {
        failure = "invalid candidate sequence arguments";
        return false;
    }
    const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
    if (vocab_size <= 0) {
        failure = "llama.cpp vocabulary is empty";
        return false;
    }
    nodes.clear();
    nodes.push_back(sequence_trie_node{});
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const size_t begin = candidate_offsets[candidate];
        const size_t end = candidate_offsets[candidate + 1];
        if (end <= begin || (candidate > 0 && begin < candidate_offsets[candidate - 1])) {
            failure = "candidate sequence offsets are invalid";
            return false;
        }
        size_t node = 0;
        for (size_t offset = begin; offset < end; ++offset) {
            if (nodes[node].terminal >= 0) {
                failure = "candidate sequence codebook is not prefix-free";
                return false;
            }
            const uint32_t token = candidate_tokens[offset];
            if (token >= static_cast<uint32_t>(vocab_size)) {
                failure = "candidate sequence token exceeds vocabulary size";
                return false;
            }
            auto found = nodes[node].children.find(token);
            if (found == nodes[node].children.end()) {
                const size_t child = nodes.size();
                nodes.push_back(sequence_trie_node{});
                nodes[node].children.emplace(token, child);
                node = child;
            } else {
                node = found->second;
            }
        }
        if (nodes[node].terminal >= 0) {
            failure = "duplicate candidate sequence";
            return false;
        }
        if (!nodes[node].children.empty()) {
            failure = "candidate sequence codebook is not prefix-free";
            return false;
        }
        nodes[node].terminal = static_cast<int64_t>(candidate);
    }
    return true;
}

bool full_vocab_logsumexp(const std::vector<float> & logits, float & out, std::string & failure) {
    if (logits.empty()) {
        failure = "empty vocabulary logits";
        return false;
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : logits) {
        if (!std::isfinite(value)) {
            failure = "non-finite vocabulary logit";
            return false;
        }
        maximum = std::max(maximum, value);
    }
    double sum = 0.0;
    for (float value : logits) {
        sum += std::exp(static_cast<double>(value - maximum));
    }
    if (!std::isfinite(sum) || sum <= 0.0) {
        failure = "invalid vocabulary logsumexp";
        return false;
    }
    out = maximum + static_cast<float>(std::log(sum));
    return true;
}

bool score_sequence_trie(
    tj_llama_runtime * runtime,
    llama_seq_id work_seq,
    size_t base_len,
    const std::vector<float> & root_logits,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    float * out_scores,
    std::string & failure) {
    if (runtime == nullptr || runtime->context == nullptr || out_scores == nullptr) {
        failure = "invalid sequence scoring runtime";
        return false;
    }
    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        failure = "llama.cpp context does not expose sequence memory";
        return false;
    }
    std::vector<sequence_trie_node> nodes;
    if (!build_sequence_trie(
            runtime, candidate_tokens, candidate_offsets, candidate_count, nodes, failure)) {
        return false;
    }
    const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
    if (root_logits.size() != static_cast<size_t>(vocab_size)) {
        failure = "root vocabulary logits have the wrong size";
        return false;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        out_scores[i] = -std::numeric_limits<float>::infinity();
    }

    std::function<bool(size_t, size_t, float, const std::vector<float> &)> visit;
    visit = [&](size_t node_index, size_t depth, float cumulative, const std::vector<float> & logits) -> bool {
        float log_z = 0.0f;
        if (!full_vocab_logsumexp(logits, log_z, failure)) {
            return false;
        }
        const auto & node = nodes[node_index];
        for (const auto & [token, child_index] : node.children) {
            const float score = cumulative + logits[token] - log_z;
            const auto & child = nodes[child_index];
            if (child.terminal >= 0) {
                out_scores[static_cast<size_t>(child.terminal)] = score;
            }
            if (!child.children.empty()) {
                const size_t position_size = base_len + depth;
                llama_pos position = 0;
                if (!checked_pos(position_size, position)) {
                    failure = "candidate sequence exceeds llama.cpp position range";
                    return false;
                }
                if (runtime->n_ctx > 0 && position_size >= static_cast<size_t>(runtime->n_ctx)) {
                    failure = "candidate sequence exceeds llama.cpp context window";
                    return false;
                }
                llama_batch batch = llama_batch_init(1, 0, 1);
                batch.n_tokens = 1;
                batch.token[0] = static_cast<llama_token>(token);
                batch.pos[0] = position;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = work_seq;
                batch.logits[0] = 1;
                const int32_t status = llama_decode(runtime->context, batch);
                llama_batch_free(batch);
                if (status != 0) {
                    failure = "llama_decode failed while traversing candidate sequence trie";
                    return false;
                }
                const float * child_logits_ptr = llama_get_logits_ith(runtime->context, -1);
                if (child_logits_ptr == nullptr) {
                    failure = "llama_get_logits_ith returned null during candidate sequence scoring";
                    return false;
                }
                std::vector<float> child_logits(
                    child_logits_ptr, child_logits_ptr + static_cast<size_t>(vocab_size));
                if (!visit(child_index, depth + 1, score, child_logits)) {
                    return false;
                }
                const size_t remove_from_size = base_len + depth;
                llama_pos remove_from = 0;
                if (!checked_pos(remove_from_size, remove_from)) {
                    failure = "candidate sequence backtrack position overflow";
                    return false;
                }
                llama_memory_seq_rm(memory, work_seq, remove_from, -1);
            }
        }
        return true;
    };

    if (!visit(0, 0, 0.0f, root_logits)) {
        return false;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!std::isfinite(out_scores[i])) {
            failure = "candidate sequence did not receive a finite score";
            return false;
        }
    }
    return true;
}

} // namespace

uint32_t tj_llama_abi_version(void) {
    return TURBOJEV_LLAMA_SHIM_ABI_VERSION;
}

tj_llama_runtime * tj_llama_open(
    const char * model_path,
    const tj_llama_config * config,
    char * error,
    size_t error_capacity) {
    if (model_path == nullptr || config == nullptr) {
        set_error(error, error_capacity, "model_path and config are required");
        return nullptr;
    }
    init_backend_once();

    if (config->n_gpu_layers != 0
        && ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr
        && ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) == nullptr) {
        set_error(error, error_capacity,
            "GPU was requested, but this llama.cpp build has no available GPU backend/device");
        return nullptr;
    }

    auto runtime = std::make_unique<tj_llama_runtime>();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config->n_gpu_layers;
    runtime->model = llama_model_load_from_file(model_path, model_params);
    if (runtime->model == nullptr) {
        set_error(error, error_capacity, "llama_model_load_from_file failed");
        return nullptr;
    }

    runtime->vocab = llama_model_get_vocab(runtime->model);
    if (runtime->vocab == nullptr) {
        set_error(error, error_capacity, "model has no vocabulary");
        llama_model_free(runtime->model);
        return nullptr;
    }

    const char * chat_template = llama_model_chat_template(runtime->model, nullptr);
    if (chat_template == nullptr || chat_template[0] == '\0') {
        set_error(error, error_capacity, "GGUF model does not expose a chat template");
        llama_model_free(runtime->model);
        return nullptr;
    }
    try {
        runtime->chat_templates = common_chat_templates_init(runtime->model, chat_template);
    } catch (const std::exception & exception) {
        set_error(error, error_capacity,
            std::string("GGUF chat template initialization failed: ") + exception.what());
        llama_model_free(runtime->model);
        return nullptr;
    }

    llama_context_params context_params = llama_context_default_params();
    // A CPU request must also disable llama.cpp's automatic operation offload.
    // Zero GPU weight layers alone can still allocate a Vulkan compute graph.
    context_params.offload_kqv = config->n_gpu_layers != 0;
    context_params.op_offload = config->n_gpu_layers != 0;
    if (config->n_ctx > 0) {
        context_params.n_ctx = config->n_ctx;
    }
    if (config->n_batch > 0) {
        context_params.n_batch = config->n_batch;
    }
    const uint32_t requested_sequences = config->n_seq_max > 0
        ? config->n_seq_max
        : context_params.n_seq_max;
    // Sequence 0 is the immutable prefix; at least one branch is required for
    // cached-serial compatibility. v0.8 uses any additional sequences for
    // bounded parallel suffix batches. Multimodal prefill is deliberately
    // single-sequence so it can use the full context instead of splitting it.
    context_params.n_seq_max = requested_sequences == 1
        ? 1u
        : std::max<uint32_t>(requested_sequences, 2u);
    context_params.no_perf = true;
    runtime->context = llama_init_from_model(runtime->model, context_params);
    if (runtime->context == nullptr) {
        set_error(error, error_capacity, "llama_init_from_model failed");
        llama_model_free(runtime->model);
        return nullptr;
    }
    // Query effective values after context creation. llama.cpp may adjust
    // requested context/batch/sequence capacities for the selected backend.
    runtime->n_ctx = llama_n_ctx(runtime->context);
    runtime->n_ctx_seq = llama_n_ctx_seq(runtime->context);
    runtime->n_batch = llama_n_batch(runtime->context);
    runtime->n_seq_max = llama_n_seq_max(runtime->context);
    if (config->n_threads > 0 || config->n_threads_batch > 0) {
        const int32_t threads = config->n_threads > 0 ? config->n_threads : config->n_threads_batch;
        const int32_t batch_threads = config->n_threads_batch > 0 ? config->n_threads_batch : threads;
        llama_set_n_threads(runtime->context, threads, batch_threads);
    }
    return runtime.release();
}

tj_llama_runtime * tj_llama_open_multimodal(
    const char * model_path,
    const char * mmproj_path,
    const tj_llama_config * config,
    char * error,
    size_t error_capacity) {
#if defined(TURBOJEV_HAS_MTMD)
    if (config == nullptr) {
        set_error(error, error_capacity, "config is required");
        return nullptr;
    }
    if (mmproj_path == nullptr || mmproj_path[0] == '\0') {
        set_error(error, error_capacity, "a compatible mmproj GGUF path is required for multimodal inference");
        return nullptr;
    }
    // Media scoring processes one decision sequence at a time. Avoid dividing
    // the model context among text-runtime sequence slots: multi-frame prompts
    // need the full context for a single mtmd prefill.
    tj_llama_config multimodal_config = *config;
    multimodal_config.n_seq_max = 1;
    std::unique_ptr<tj_llama_runtime, decltype(&tj_llama_close)> runtime(
        tj_llama_open(model_path, &multimodal_config, error, error_capacity), &tj_llama_close);
    if (!runtime) return nullptr;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = config->n_gpu_layers != 0;
    params.n_threads = config->n_threads > 0 ? config->n_threads : 0;
    params.print_timings = false;
    params.warmup = true;
    runtime->mtmd = mtmd_init_from_file(mmproj_path, runtime->model, params);
    if (runtime->mtmd == nullptr) {
        set_error(error, error_capacity,
            "mtmd_init_from_file failed; verify that the mmproj GGUF matches the text model and llama.cpp build");
        return nullptr;
    }
    return runtime.release();
#else
    (void) model_path;
    (void) mmproj_path;
    (void) config;
    set_error(error, error_capacity, "this build does not include the mtmd multimodal library");
    return nullptr;
#endif
}

int32_t tj_llama_supports_image(const tj_llama_runtime * runtime) {
#if defined(TURBOJEV_HAS_MTMD)
    return runtime != nullptr && runtime->mtmd != nullptr && mtmd_support_vision(runtime->mtmd);
#else
    (void) runtime;
    return 0;
#endif
}

int32_t tj_llama_supports_audio(const tj_llama_runtime * runtime) {
#if defined(TURBOJEV_HAS_MTMD)
    return runtime != nullptr && runtime->mtmd != nullptr && mtmd_support_audio(runtime->mtmd);
#else
    (void) runtime;
    return 0;
#endif
}

int32_t tj_llama_supports_video(const tj_llama_runtime * runtime) {
#if defined(TURBOJEV_HAS_MTMD)
    // mtmd represents video as a timestamped sequence of mergeable image
    // bitmaps. The model must at least have a vision projector; temporal
    // reasoning quality remains checkpoint-dependent.
    return runtime != nullptr && runtime->mtmd != nullptr && mtmd_support_vision(runtime->mtmd);
#else
    (void) runtime;
    return 0;
#endif
}

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
    size_t error_capacity) {
#if defined(TURBOJEV_HAS_MTMD)
    if (runtime == nullptr || runtime->context == nullptr || runtime->mtmd == nullptr
        || runtime->vocab == nullptr || user == nullptr || candidate_tokens == nullptr
        || candidate_count < 2 || out_logits == nullptr || modality < 0 || modality > 2
        || media_base64 == nullptr) {
        set_error(error, error_capacity, "invalid multimodal scoring arguments or runtime was not opened with mmproj");
        return -1;
    }
    const bool image_list = modality == 0 && frame_count > 0;
    if (((modality == 2 || image_list) && (frame_count == 0 || frame_count > 32
            || frame_base64 == nullptr || frame_base64_lengths == nullptr || frame_timestamps_ms == nullptr
            || media_base64_len != 0))
        || (modality != 2 && !image_list && (frame_count != 0 || media_base64_len == 0))) {
        set_error(error, error_capacity, "video requires 1..32 timestamped image frames; image/audio require one encoded payload");
        return -1;
    }
    if ((modality == 0 && !tj_llama_supports_image(runtime))
        || (modality == 1 && !tj_llama_supports_audio(runtime))
        || (modality == 2 && !tj_llama_supports_video(runtime))) {
        set_error(error, error_capacity, "loaded mmproj does not support the requested input modality");
        return -1;
    }

    std::string rendered;
    std::string failure;
    if (!render_chat(runtime, system, user, rendered, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }

    const auto options = mtmd_helper_init_opt_default();
    // Match the text codec's exact candidate boundary, including answer prefill.
    if (assistant_prefix != nullptr) rendered.append(assistant_prefix);
    std::vector<std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>> bitmap_storage;
    std::vector<const mtmd_bitmap *> bitmaps;
    if (modality == 2 || image_list) {
        bitmap_storage.reserve(frame_count);
        bitmaps.reserve(frame_count);
        uint64_t previous_timestamp = 0;
        for (size_t i = 0; i < frame_count; ++i) {
            if (frame_base64[i] == nullptr || frame_base64_lengths[i] == 0
                || frame_base64_lengths[i] > 32u * 1024u * 1024u
                || (i > 0 && frame_timestamps_ms[i] <= previous_timestamp)) {
                set_error(error, error_capacity, "video frame payloads must be bounded and timestamps strictly increasing");
                return -1;
            }
            previous_timestamp = frame_timestamps_ms[i];
            std::vector<unsigned char> frame_bytes;
            if (!decode_base64(frame_base64[i], frame_base64_lengths[i], frame_bytes)) {
                set_error(error, error_capacity, "video frame is not valid bounded base64");
                return -1;
            }
            auto wrapper = mtmd_helper_bitmap_init_from_buf(
                runtime->mtmd, frame_bytes.data(), frame_bytes.size(), false, options);
            if (wrapper.video_ctx != nullptr) {
                mtmd_helper_video_free(wrapper.video_ctx);
            }
            if (wrapper.bitmap == nullptr || mtmd_bitmap_is_audio(wrapper.bitmap)) {
                mtmd_bitmap_free(wrapper.bitmap);
                set_error(error, error_capacity, "video frames must decode to still images supported by mtmd");
                return -1;
            }
            mtmd_bitmap_set_mergeable(wrapper.bitmap, true);
            bitmap_storage.emplace_back(wrapper.bitmap, &mtmd_bitmap_free);
            bitmaps.push_back(wrapper.bitmap);
        }
    } else {
        std::vector<unsigned char> media;
        if (!decode_base64(media_base64, media_base64_len, media)) {
            set_error(error, error_capacity, "media payload is not valid bounded base64");
            return -1;
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            runtime->mtmd, media.data(), media.size(), false, options);
        if (wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(wrapper.video_ctx);
        }
        if (wrapper.bitmap == nullptr) {
            set_error(error, error_capacity, "mtmd could not decode the media payload; check its file format and model support");
            return -1;
        }
        if ((modality == 0 && mtmd_bitmap_is_audio(wrapper.bitmap))
            || (modality == 1 && !mtmd_bitmap_is_audio(wrapper.bitmap))) {
            mtmd_bitmap_free(wrapper.bitmap);
            set_error(error, error_capacity, "media bytes do not match the declared modality");
            return -1;
        }
        bitmaps.push_back(wrapper.bitmap);
        bitmap_storage.emplace_back(wrapper.bitmap, &mtmd_bitmap_free);
    }

    mtmd_input_chunks * raw_chunks = mtmd_input_chunks_init();
    if (raw_chunks == nullptr) {
        set_error(error, error_capacity, "mtmd_input_chunks_init failed");
        return -1;
    }
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        raw_chunks, &mtmd_input_chunks_free);
    const std::string marker = mtmd_get_marker(runtime->mtmd);
    const size_t expected_markers = (modality == 2 || image_list) ? frame_count : 1;
    size_t marker_count = 0;
    for (size_t position = 0; !marker.empty()
         && (position = rendered.find(marker, position)) != std::string::npos;
         position += marker.size()) {
        ++marker_count;
    }
    if (marker.empty() || marker_count != expected_markers) {
        set_error(error, error_capacity, "rendered chat template omitted the llama.cpp media marker");
        return -1;
    }
    mtmd_input_text prompt_text{rendered.data(), rendered.size(), true, true};
    const int32_t tokenize_status = mtmd_tokenize(
        runtime->mtmd, chunks.get(), &prompt_text, bitmaps.data(), bitmaps.size());
    if (tokenize_status != 0) {
        set_error(error, error_capacity, tokenize_status == 1
            ? "rendered prompt must contain exactly one media marker"
            : "mtmd media preprocessing failed");
        return -1;
    }

    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory != nullptr) llama_memory_clear(memory, true);
    runtime->shared_prefix_ready = false;
    runtime->shared_prefix_len = 0;
    runtime->shared_prefix_logits.clear();
    llama_pos n_past = 0;
    const int32_t eval_status = mtmd_helper_eval_chunks(
        runtime->mtmd, runtime->context, chunks.get(), 0, 0,
        static_cast<int32_t>(runtime->n_batch), true, &n_past);
    if (eval_status != 0) {
        set_error(error, error_capacity, "mtmd_helper_eval_chunks failed during multimodal prefill");
        return -1;
    }
    const float * logits = llama_get_logits_ith(runtime->context, -1);
    if (logits == nullptr || !gather_candidate_logits(
            runtime, logits, candidate_tokens, candidate_count, out_logits, failure)) {
        set_error(error, error_capacity, logits == nullptr
            ? "llama_get_logits_ith returned null after multimodal prefill" : failure);
        return -1;
    }
    return 0;
#else
    (void) runtime;
    (void) system;
    (void) user;
    (void) assistant_prefix;
    (void) modality;
    (void) media_base64;
    (void) media_base64_len;
    (void) frame_base64;
    (void) frame_base64_lengths;
    (void) frame_timestamps_ms;
    (void) frame_count;
    (void) candidate_tokens;
    (void) candidate_count;
    (void) out_logits;
    set_error(error, error_capacity, "this build does not include the mtmd multimodal library");
    return -1;
#endif
}

void tj_llama_close(tj_llama_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }
#if defined(TURBOJEV_HAS_MTMD)
    if (runtime->mtmd != nullptr) {
        mtmd_free(runtime->mtmd);
    }
#endif
    if (runtime->context != nullptr) {
        llama_free(runtime->context);
    }
    if (runtime->model != nullptr) {
        llama_model_free(runtime->model);
    }
    delete runtime;
}

int64_t tj_llama_model_desc(
    tj_llama_runtime * runtime,
    char * out,
    size_t capacity,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->model == nullptr) {
        set_error(error, error_capacity, "runtime is not initialized");
        return -1;
    }
    char buffer[512] = {};
    const int32_t length = llama_model_desc(runtime->model, buffer, sizeof(buffer));
    if (length < 0) {
        set_error(error, error_capacity, "llama_model_desc failed");
        return -1;
    }
    return copy_text(std::string(buffer, static_cast<size_t>(length)), out, capacity);
}

int64_t tj_llama_encode_chat(
    tj_llama_runtime * runtime,
    const char * system,
    const char * user,
    const char * answer,
    uint32_t * out_tokens,
    size_t capacity,
    char * error,
    size_t error_capacity) {
    std::string rendered;
    std::string failure;
    if (!render_chat(runtime, system, user, rendered, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    if (answer != nullptr) {
        rendered.append(answer);
    }

    std::vector<llama_token> tokens;
    if (!tokenize(runtime, rendered, tokens, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    if (out_tokens == nullptr || capacity == 0) {
        return static_cast<int64_t>(tokens.size());
    }
    if (capacity < tokens.size()) {
        set_error(error, error_capacity, "token output buffer is too small");
        return -1;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0) {
            set_error(error, error_capacity, "llama.cpp produced a negative token id");
            return -1;
        }
        out_tokens[i] = static_cast<uint32_t>(tokens[i]);
    }
    return static_cast<int64_t>(tokens.size());
}

int64_t tj_llama_decode_tokens(
    tj_llama_runtime * runtime,
    const uint32_t * tokens,
    size_t token_count,
    char * out_text,
    size_t capacity,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->vocab == nullptr || (tokens == nullptr && token_count != 0)) {
        set_error(error, error_capacity, "invalid decode arguments");
        return -1;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        set_error(error, error_capacity, "too many tokens to decode");
        return -1;
    }
    std::vector<llama_token> native_tokens(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        if (tokens[i] > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            set_error(error, error_capacity, "token id exceeds llama.cpp range");
            return -1;
        }
        native_tokens[i] = static_cast<llama_token>(tokens[i]);
    }

    const int32_t needed = llama_detokenize(
        runtime->vocab, native_tokens.data(), static_cast<int32_t>(native_tokens.size()),
        nullptr, 0, false, false);
    if (needed == std::numeric_limits<int32_t>::min()) {
        set_error(error, error_capacity, "llama_detokenize overflow");
        return -1;
    }
    const int32_t bytes_needed = needed < 0 ? -needed : needed;
    if (out_text == nullptr || capacity == 0) {
        return bytes_needed;
    }
    if (capacity <= static_cast<size_t>(bytes_needed)) {
        set_error(error, error_capacity, "decode output buffer is too small");
        return -1;
    }
    const int32_t written = llama_detokenize(
        runtime->vocab, native_tokens.data(), static_cast<int32_t>(native_tokens.size()),
        out_text, static_cast<int32_t>(capacity - 1), false, false);
    if (written < 0) {
        set_error(error, error_capacity, "llama_detokenize failed");
        return -1;
    }
    out_text[written] = '\0';
    return written;
}

int32_t tj_llama_score_next(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    size_t prompt_count,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || prompt_tokens == nullptr || prompt_count == 0
        || candidate_tokens == nullptr || candidate_count < 2 || out_logits == nullptr) {
        set_error(error, error_capacity, "invalid scoring arguments");
        return -1;
    }

    std::vector<llama_token> prompt;
    std::string failure;
    if (!copy_tokens(prompt_tokens, prompt_count, prompt, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }

    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory != nullptr) {
        llama_memory_clear(memory, true);
    }
    runtime->shared_prefix_ready = false;
    runtime->shared_prefix_len = 0;
    runtime->shared_prefix_logits.clear();

    if (prompt.size() >= llama_n_ctx_seq(runtime->context)) {
        set_error(error, error_capacity, "prompt exceeds the per-sequence context; increase nCtx or reduce nSeqMax");
        return -1;
    }
    size_t processed = 0;
    while (processed < prompt.size()) {
        const auto count = std::min(prompt.size() - processed, size_t(llama_n_batch(runtime->context)));
        llama_batch batch = llama_batch_get_one(prompt.data() + processed, static_cast<int32_t>(count));
        if (llama_decode(runtime->context, batch) != 0) {
            set_error(error, error_capacity, "llama_decode failed");
            return -1;
        }
        processed += count;
    }

    const float * logits = llama_get_logits_ith(runtime->context, -1);
    if (logits == nullptr) {
        set_error(error, error_capacity, "llama_get_logits_ith returned null");
        return -1;
    }
    if (!gather_candidate_logits(
            runtime, logits, candidate_tokens, candidate_count, out_logits, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}



int32_t tj_llama_score_thinking_next(
    tj_llama_runtime * runtime, const char * system, const char * user,
    const char * assistant_prefix, const uint32_t * candidate_tokens,
    size_t candidate_count, uint32_t max_thinking_tokens, float * out_logits,
    size_t * generated_tokens, char * error, size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || user == nullptr
        || assistant_prefix == nullptr || candidate_tokens == nullptr || candidate_count < 2
        || max_thinking_tokens == 0 || out_logits == nullptr || generated_tokens == nullptr) {
        set_error(error, error_capacity, "invalid thinking arguments"); return -1;
    }
    *generated_tokens = 0;
    try {
        const auto params = tj_render_chat_params(
            runtime->chat_templates.get(), system, user, runtime->chat_time, true);
        if (!params.supports_thinking || params.thinking_end_tags.empty()) {
            set_error(error, error_capacity, "THINKING_UNSUPPORTED: model template exposes no reasoning boundary");
            return -1;
        }
        std::string failure;
        std::vector<llama_token> prompt;
        if (!tokenize(runtime, params.prompt, prompt, failure)) {
            set_error(error, error_capacity, failure); return -1;
        }
        // Reserve the requested budget and the final prefill in this one sequence.
        if (prompt.size() + max_thinking_tokens + 32 >= llama_n_ctx_seq(runtime->context)) {
            set_error(error, error_capacity, "THINKING_CONTEXT_LIMIT: increase nCtx or reduce nSeqMax for the prompt and reasoning budget");
            return -1;
        }
        if (llama_get_memory(runtime->context)) llama_memory_clear(llama_get_memory(runtime->context), true);
        runtime->shared_prefix_ready = false;
        runtime->shared_prefix_len = 0;
        runtime->shared_prefix_logits.clear();
        size_t processed = 0;
        while (processed < prompt.size()) {
            const auto count = std::min(prompt.size() - processed, size_t(llama_n_batch(runtime->context)));
            auto batch = llama_batch_get_one(prompt.data() + processed, static_cast<int32_t>(count));
            if (llama_decode(runtime->context, batch) != 0) {
                set_error(error, error_capacity, "thinking prefill failed"); return -1;
            }
            processed += count;
        }
        std::string reasoning;
        bool ended = false;
        const auto vocab_size = llama_vocab_n_tokens(runtime->vocab);
        for (uint32_t i = 0; i < max_thinking_tokens; ++i) {
            const float * logits = llama_get_logits_ith(runtime->context, -1);
            if (logits == nullptr || vocab_size <= 0) {
                set_error(error, error_capacity, "thinking logits unavailable"); return -1;
            }
            llama_token next = static_cast<llama_token>(std::max_element(logits, logits + vocab_size) - logits);
            const auto needed = llama_detokenize(runtime->vocab, &next, 1, nullptr, 0, false, true);
            if (needed == std::numeric_limits<int32_t>::min()) {
                set_error(error, error_capacity, "thinking token decoding overflow"); return -1;
            }
            std::string piece(size_t(needed < 0 ? -needed : needed), '\0');
            const auto length = llama_detokenize(runtime->vocab, &next, 1, piece.data(),
                static_cast<int32_t>(piece.size()), false, true);
            if (length < 0) { set_error(error, error_capacity, "thinking token decoding failed"); return -1; }
            piece.resize(size_t(length));
            reasoning += piece;
            ++*generated_tokens;
            for (const auto & end : params.thinking_end_tags) {
                if (!end.empty() && reasoning.size() >= end.size()
                    && reasoning.compare(reasoning.size() - end.size(), end.size(), end) == 0) ended = true;
            }
            if (ended) break;
            auto batch = llama_batch_get_one(&next, 1);
            if (llama_decode(runtime->context, batch) != 0) {
                set_error(error, error_capacity, "thinking token inference failed"); return -1;
            }
        }
        if (!ended) {
            set_error(error, error_capacity, "THINKING_BUDGET_EXHAUSTED: reasoning did not finish; no answer was fabricated");
            return -1;
        }
        const std::string final_text = params.prompt + reasoning + "\n\n" + assistant_prefix;
        std::vector<llama_token> final_tokens;
        if (!tokenize(runtime, final_text, final_tokens, failure)) {
            set_error(error, error_capacity, failure); return -1;
        }
        // Verify each original candidate token against this exact, reasoned prefix.
        for (size_t i = 0; i < candidate_count; ++i) {
            if (candidate_tokens[i] >= uint32_t(vocab_size)) {
                set_error(error, error_capacity, "thinking candidate token out of range"); return -1;
            }
            const llama_token token = static_cast<llama_token>(candidate_tokens[i]);
            char text[1024];
            const auto count = llama_detokenize(runtime->vocab, &token, 1, text, sizeof(text), false, true);
            if (count <= 0) { set_error(error, error_capacity, "thinking candidate decode failed"); return -1; }
            std::vector<llama_token> extended;
            if (!tokenize(runtime, final_text + std::string(text, size_t(count)), extended, failure)
                || extended.size() != final_tokens.size() + 1
                || !std::equal(final_tokens.begin(), final_tokens.end(), extended.begin())
                || extended.back() != token) {
                set_error(error, error_capacity, "thinking candidate does not match the final-answer token boundary");
                return -1;
            }
        }
        // Independent reference scoring. Re-evaluate the completed thought exactly.
        std::vector<uint32_t> ids(final_tokens.begin(), final_tokens.end());
        const auto status = tj_llama_score_next(runtime, ids.data(), ids.size(), candidate_tokens,
            candidate_count, out_logits, error, error_capacity);
        if (status == 0) {
            runtime->field_result = nlohmann::ordered_json{
                {"logits", std::vector<float>(out_logits, out_logits + candidate_count)},
                {"thinking", reasoning}, {"input_tokens", prompt.size()}, {"generated_tokens", *generated_tokens}}.dump();
        }
        return status;
    } catch (const std::exception & exception) {
        set_error(error, error_capacity, std::string("thinking failed: ") + exception.what()); return -1;
    }
}

int32_t tj_llama_generate_field(tj_llama_runtime * runtime, const char * task_json,
    char * error, size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || task_json == nullptr) {
        set_error(error, error_capacity, "invalid field generation arguments"); return -1;
    }
    runtime->field_result.clear();
    try {
        using json = nlohmann::ordered_json;
        const auto task = json::parse(task_json);
        const auto state = task.at("state");
        const bool media = state.is_object() && state.value("contract", "") == "turbojev.media.v1";
        const bool thinking = task.at("thinking").get<bool>();
        const uint32_t thinking_budget = task.at("thinking_budget").get<uint32_t>();
        const uint32_t max_tokens = task.at("max_tokens").get<uint32_t>();
        const double temperature = task.value("temperature", 0.0);
        if (!std::isfinite(temperature) || temperature < 0.0) {
            set_error(error, error_capacity, "temperature must be finite and nonnegative"); return -1;
        }
        std::mt19937_64 random(task.value("seed", uint64_t(0)));
        if (max_tokens == 0 || max_tokens > 32768 || thinking_budget > 32768) {
            set_error(error, error_capacity, "invalid field token limits"); return -1;
        }
        if (media && thinking) {
            set_error(error, error_capacity, "THINKING_UNSUPPORTED: media reasoning is not implemented"); return -1;
        }
        const std::string system = "Answer the requested field from the supplied evidence. Return only the requested JSON object.";
        std::string context = media ? state.value("text", "") : (state.is_string() ? state.get<std::string>() : state.dump());
        if (media) {
#if defined(TURBOJEV_HAS_MTMD)
            if (!runtime->mtmd) { set_error(error, error_capacity, "multimodal field generation requires a matching mmproj"); return -1; }
            const std::string marker = mtmd_get_marker(runtime->mtmd);
            const size_t count = state.value("modality", "") == "video" ? state.at("frames").size() : state.contains("images") ? state.at("images").size() : 1;
            for (size_t i = 0; i < count; ++i) context += "\n" + marker;
#else
            set_error(error, error_capacity, "multimodal field generation is unavailable in this build"); return -1;
#endif
        }
        const std::string user = "State:\n" + context + "\n\nQuestion:\n" + task.at("instructions").get<std::string>()
            + "\n\nReturn exactly one JSON object with the key value and this value schema: " + task.at("output_schema").dump();
        const auto chat = tj_render_chat_params(runtime->chat_templates.get(), system.c_str(), user.c_str(), runtime->chat_time, thinking);
        if (thinking && (!chat.supports_thinking || chat.thinking_end_tags.empty() || thinking_budget == 0)) {
            set_error(error, error_capacity, "THINKING_UNSUPPORTED: model exposes no reasoning boundary"); return -1;
        }
        const auto object_schema = json{{"type", "object"}, {"properties", {{"value", task.at("output_schema")}}}, {"required", {"value"}}, {"additionalProperties", false}};
        const auto grammar_text = json_schema_to_grammar(common_json::parse(object_schema.dump()), true);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> grammar(
            llama_sampler_init_grammar(runtime->vocab, grammar_text.c_str(), "root"), &llama_sampler_free);
        if (!grammar) { set_error(error, error_capacity, "failed to compile output grammar"); return -1; }
        size_t input_tokens = 0;
        std::string failure;
        if (media) {
            const auto modality_name = state.at("modality").get<std::string>();
            const int modality = modality_name == "image" ? 0 : modality_name == "audio" ? 1 : modality_name == "video" ? 2 : -1;
            const std::string payload = state.value("base64", "");
            std::vector<std::string> storage; std::vector<const char *> frames;
            std::vector<size_t> lengths; std::vector<uint64_t> timestamps;
            for (const auto & frame : state.value("frames", json::array())) {
                storage.push_back(frame.at("base64").get<std::string>());
                timestamps.push_back(frame.at("timestamp_ms").get<uint64_t>());
            }
            if (state.contains("images")) {
                for (const auto & image : state.at("images")) {
                    storage.push_back(image.get<std::string>());
                    timestamps.push_back(timestamps.size());
                }
            }
            for (const auto & frame : storage) { frames.push_back(frame.c_str()); lengths.push_back(frame.size()); }
            // Reuse the exact mtmd preprocessing/prefill route. The two dummy
            // projections only expose its final logits; they do not choose output.
            const uint32_t ids[] = {0, 1}; float ignored[2];
            if (tj_llama_score_media_next(runtime, system.c_str(), user.c_str(), "", modality,
                payload.c_str(), payload.size(), frames.data(), lengths.data(), timestamps.data(), frames.size(),
                ids, 2, ignored, error, error_capacity) != 0) return -1;
        } else {
            std::vector<llama_token> tokens;
            if (!tokenize(runtime, chat.prompt, tokens, failure)) { set_error(error, error_capacity, failure); return -1; }
            input_tokens = tokens.size();
            if (tokens.size() + max_tokens + (thinking ? thinking_budget : 0) >= llama_n_ctx_seq(runtime->context)) {
                set_error(error, error_capacity, "GENERATION_CONTEXT_LIMIT: increase nCtx or reduce nSeqMax"); return -1;
            }
            if (llama_get_memory(runtime->context)) llama_memory_clear(llama_get_memory(runtime->context), true);
            runtime->shared_prefix_ready = false; runtime->shared_prefix_len = 0; runtime->shared_prefix_logits.clear();
            for (size_t i = 0; i < tokens.size();) {
                const auto count = std::min(tokens.size() - i, size_t(llama_n_batch(runtime->context)));
                auto batch = llama_batch_get_one(tokens.data() + i, static_cast<int32_t>(count));
                if (llama_decode(runtime->context, batch) != 0) { set_error(error, error_capacity, "field prefill failed"); return -1; }
                i += count;
            }
        }
        const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
        std::vector<llama_token_data> candidates(size_t(vocab_size), llama_token_data{});
        std::string thought, answer;
        bool thinking_done = !thinking;
        uint32_t thought_tokens = 0, answer_tokens = 0;
        const uint32_t total_budget = max_tokens + (thinking ? thinking_budget : 0);
        for (uint32_t generated = 0; generated < total_budget; ++generated) {
            if ((thinking_done && answer_tokens >= max_tokens) || (!thinking_done && thought_tokens >= thinking_budget)) break;
            const auto * logits = llama_get_logits_ith(runtime->context, -1);
            if (!logits) { set_error(error, error_capacity, "field logits unavailable"); return -1; }
            for (llama_token i = 0; i < vocab_size; ++i) candidates[size_t(i)] = {i, logits[i], 0.0f};
            llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
            if (thinking_done) llama_sampler_apply(grammar.get(), &array);
            const auto best = std::max_element(array.data, array.data + array.size, [](const auto & a, const auto & b) { return a.logit < b.logit; });
            if (best == array.data + array.size || !std::isfinite(best->logit)) { set_error(error, error_capacity, "grammar permits no token"); return -1; }
            llama_token next = best->id;
            if (temperature > 0.0) {
                std::vector<double> weights(array.size);
                for (size_t i = 0; i < array.size; ++i) {
                    weights[i] = std::exp((double(array.data[i].logit) - best->logit) / temperature);
                }
                std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
                next = array.data[distribution(random)].id;
            }
            if (llama_vocab_is_eog(runtime->vocab, next)) break;
            const int32_t required = llama_detokenize(runtime->vocab, &next, 1, nullptr, 0, false, true);
            if (required == std::numeric_limits<int32_t>::min()) { set_error(error, error_capacity, "token decoding overflow"); return -1; }
            std::string piece(size_t(required < 0 ? -required : required), '\0');
            const auto size = llama_detokenize(runtime->vocab, &next, 1, piece.data(), static_cast<int32_t>(piece.size()), false, true);
            if (size < 0) { set_error(error, error_capacity, "field token decoding failed"); return -1; }
            piece.resize(size_t(size));
            if (thinking_done) {
                ++answer_tokens; answer += piece; llama_sampler_accept(grammar.get(), next);
                const auto parsed = json::parse(answer, nullptr, false);
                if (!parsed.is_discarded() && parsed.is_object() && parsed.size() == 1 && parsed.contains("value")) {
                    runtime->field_result = json{{"value", parsed.at("value")}, {"logits", json::array()}, {"thinking", thought},
                        {"input_tokens", media ? json(nullptr) : json(input_tokens)}, {"generated_tokens", thought_tokens + answer_tokens}}.dump();
                    return 0;
                }
            } else {
                ++thought_tokens; thought += piece;
                for (const auto & end : chat.thinking_end_tags) {
                    if (!end.empty() && thought.size() >= end.size() && thought.compare(thought.size()-end.size(), end.size(), end) == 0) thinking_done = true;
                }
            }
            auto batch = llama_batch_get_one(&next, 1);
            if (llama_decode(runtime->context, batch) != 0) { set_error(error, error_capacity, "field token inference failed"); return -1; }
        }
        set_error(error, error_capacity, thinking_done ? "GENERATION_BUDGET_EXHAUSTED: unfinished field value" : "THINKING_BUDGET_EXHAUSTED: unfinished reasoning");
        return -1;
    } catch (const std::exception & exception) {
        set_error(error, error_capacity, std::string("field generation failed: ") + exception.what()); return -1;
    }
}

int64_t tj_llama_field_result(tj_llama_runtime * runtime, char * out, size_t capacity,
    char * error, size_t error_capacity) {
    if (!runtime || runtime->field_result.empty()) { set_error(error, error_capacity, "no completed field result"); return -1; }
    return copy_text(runtime->field_result, out, capacity);
}

int32_t tj_llama_sequence_scores_supported(tj_llama_runtime * runtime) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr) {
        return 0;
    }
    return llama_get_memory(runtime->context) != nullptr ? 1 : 0;
}

int32_t tj_llama_score_sequences(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    size_t prompt_count,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    float * out_scores,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || prompt_tokens == nullptr || prompt_count == 0 || candidate_tokens == nullptr
        || candidate_offsets == nullptr || candidate_count < 2 || out_scores == nullptr) {
        set_error(error, error_capacity, "invalid sequence scoring arguments");
        return -1;
    }
    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose sequence memory");
        return -1;
    }
    std::vector<llama_token> prompt;
    std::string failure;
    if (!copy_tokens(prompt_tokens, prompt_count, prompt, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    llama_memory_clear(memory, true);
    runtime->shared_prefix_ready = false;
    runtime->shared_prefix_len = 0;
    runtime->shared_prefix_logits.clear();
    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    if (llama_decode(runtime->context, batch) != 0) {
        set_error(error, error_capacity, "llama_decode failed before sequence scoring");
        return -1;
    }
    const float * root_ptr = llama_get_logits_ith(runtime->context, -1);
    const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
    if (root_ptr == nullptr || vocab_size <= 0) {
        set_error(error, error_capacity, "missing root logits for sequence scoring");
        return -1;
    }
    std::vector<float> root_logits(root_ptr, root_ptr + static_cast<size_t>(vocab_size));
    if (!score_sequence_trie(
            runtime, 0, prompt_count, root_logits, candidate_tokens, candidate_offsets,
            candidate_count, out_scores, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}

int32_t tj_llama_multi_root_supported(tj_llama_runtime * runtime) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || runtime->n_seq_max < 2) {
        return 0;
    }
    return llama_get_memory(runtime->context) != nullptr ? 1 : 0;
}

size_t tj_llama_max_multi_root_tasks(tj_llama_runtime * runtime) {
    if (tj_llama_multi_root_supported(runtime) == 0) {
        return 0;
    }
    return static_cast<size_t>(runtime->n_seq_max);
}

size_t tj_llama_max_multi_root_tokens(tj_llama_runtime * runtime) {
    if (tj_llama_multi_root_supported(runtime) == 0) {
        return 0;
    }
    size_t limit = std::numeric_limits<size_t>::max();
    if (runtime->n_batch > 0) {
        limit = std::min(limit, static_cast<size_t>(runtime->n_batch));
    }
    if (runtime->n_ctx > 0) {
        limit = std::min(limit, static_cast<size_t>(runtime->n_ctx));
    }
    return limit == std::numeric_limits<size_t>::max() ? 0 : limit;
}

int32_t tj_llama_score_root_batch(
    tj_llama_runtime * runtime,
    const uint32_t * prompt_tokens,
    const size_t * prompt_offsets,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t task_count,
    float * out_logits,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || prompt_tokens == nullptr || prompt_offsets == nullptr || candidate_tokens == nullptr
        || candidate_offsets == nullptr || task_count < 2 || out_logits == nullptr
        || prompt_offsets[0] != 0 || candidate_offsets[0] != 0) {
        set_error(error, error_capacity, "invalid multi-root scoring arguments");
        return -1;
    }
    const size_t max_tasks = tj_llama_max_multi_root_tasks(runtime);
    if (max_tasks < 2 || task_count > max_tasks) {
        set_error(error, error_capacity, "multi-root task count exceeds llama.cpp sequence capacity");
        return -1;
    }
    if (task_count > static_cast<size_t>(std::numeric_limits<llama_seq_id>::max())) {
        set_error(error, error_capacity, "multi-root task count exceeds llama.cpp sequence-id range");
        return -1;
    }

    size_t max_prompt = 0;
    for (size_t task = 0; task < task_count; ++task) {
        const size_t prompt_begin = prompt_offsets[task];
        const size_t prompt_end = prompt_offsets[task + 1];
        const size_t candidate_begin = candidate_offsets[task];
        const size_t candidate_end = candidate_offsets[task + 1];
        if (prompt_end <= prompt_begin || prompt_end < prompt_offsets[task]
            || candidate_end < candidate_begin || candidate_end - candidate_begin < 2) {
            set_error(error, error_capacity, "multi-root offsets or candidate counts are invalid");
            return -1;
        }
        max_prompt = std::max(max_prompt, prompt_end - prompt_begin);
    }

    const size_t prompt_total = prompt_offsets[task_count];
    const size_t candidate_total = candidate_offsets[task_count];
    if (prompt_total == 0 || candidate_total == 0
        || prompt_total > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        set_error(error, error_capacity, "multi-root token buffers are invalid");
        return -1;
    }
    const size_t token_limit = tj_llama_max_multi_root_tokens(runtime);
    if (token_limit > 0 && prompt_total > token_limit) {
        set_error(error, error_capacity, "multi-root prompt tokens exceed configured batch/context capacity");
        return -1;
    }
    if (runtime->n_ctx_seq > 0 && max_prompt > static_cast<size_t>(runtime->n_ctx_seq)) {
        set_error(error, error_capacity, "one multi-root prompt exceeds effective per-sequence context capacity");
        return -1;
    }

    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose sequence memory");
        return -1;
    }
    llama_memory_clear(memory, true);
    runtime->shared_prefix_ready = false;
    runtime->shared_prefix_len = 0;
    runtime->shared_prefix_logits.clear();

    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt_total), 0, 1);

    batch.n_tokens = static_cast<int32_t>(prompt_total);
    std::vector<int32_t> root_output_indices(task_count, -1);
    size_t flat = 0;
    bool valid = true;
    std::string failure;
    for (size_t task = 0; task < task_count && valid; ++task) {
        const size_t begin = prompt_offsets[task];
        const size_t end = prompt_offsets[task + 1];
        const llama_seq_id seq = static_cast<llama_seq_id>(task);
        for (size_t local = 0; begin + local < end; ++local, ++flat) {
            const uint32_t token = prompt_tokens[begin + local];
            if (token > static_cast<uint32_t>(std::numeric_limits<llama_token>::max())) {
                failure = "multi-root prompt token id exceeds llama.cpp range";
                valid = false;
                break;
            }
            llama_pos pos = 0;
            if (!checked_pos(local, pos)) {
                failure = "multi-root prompt position exceeds llama.cpp range";
                valid = false;
                break;
            }
            batch.token[flat] = static_cast<llama_token>(token);
            batch.pos[flat] = pos;
            batch.n_seq_id[flat] = 1;
            batch.seq_id[flat][0] = seq;
            const bool wants_logits = begin + local + 1 == end;
            batch.logits[flat] = wants_logits ? 1 : 0;
            if (wants_logits) {
                root_output_indices[task] = static_cast<int32_t>(flat);
            }
        }
    }
    if (!valid) {
        llama_batch_free(batch);
        llama_memory_clear(memory, true);
        set_error(error, error_capacity, failure);
        return -1;
    }

    const int32_t decode_status = llama_decode(runtime->context, batch);
    if (decode_status != 0) {
        llama_batch_free(batch);
        llama_memory_clear(memory, true);
        set_error(error, error_capacity, "llama_decode failed while scoring independent roots");
        return -1;
    }

    bool gathered = true;
    for (size_t task = 0; task < task_count; ++task) {
        const int32_t batch_token_index = root_output_indices[task];
        if (batch_token_index < 0) {
            failure = "missing multi-root output token index";
            gathered = false;
            break;
        }
        const float * logits = llama_get_logits_ith(runtime->context, batch_token_index);
        if (logits == nullptr) {
            failure = "llama_get_logits_ith returned null for a multi-root output";
            gathered = false;
            break;
        }
        const size_t candidate_begin = candidate_offsets[task];
        const size_t candidate_count = candidate_offsets[task + 1] - candidate_begin;
        if (!gather_candidate_logits(
                runtime,
                logits,
                candidate_tokens + candidate_begin,
                candidate_count,
                out_logits + candidate_begin,
                failure)) {
            gathered = false;
            break;
        }
    }

    llama_batch_free(batch);
    // The operation owns every root sequence. Clearing after output extraction
    // guarantees no request can leak KV state into a subsequent independent call.
    llama_memory_clear(memory, true);
    if (!gathered) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}

int32_t tj_llama_shared_prefix_supported(tj_llama_runtime * runtime) {
    // Shared scoring keeps the immutable prefix in sequence 0 and evaluates
    // suffixes in sequence 1, even when suffixes are scheduled serially.
    if (runtime == nullptr || runtime->context == nullptr || runtime->n_seq_max < 2) {
        return 0;
    }
    return llama_get_memory(runtime->context) != nullptr ? 1 : 0;
}

int32_t tj_llama_parallel_suffix_supported(tj_llama_runtime * runtime) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->n_seq_max < 3) {
        return 0;
    }
    return llama_get_memory(runtime->context) != nullptr ? 1 : 0;
}

size_t tj_llama_max_parallel_tasks(tj_llama_runtime * runtime) {
    if (tj_llama_parallel_suffix_supported(runtime) == 0) {
        return 0;
    }
    return static_cast<size_t>(runtime->n_seq_max - 1u);
}

size_t tj_llama_max_parallel_suffix_tokens(tj_llama_runtime * runtime) {
    if (runtime == nullptr || !runtime->shared_prefix_ready) {
        return 0;
    }
    size_t limit = std::numeric_limits<size_t>::max();
    if (runtime->n_batch > 0) {
        limit = std::min(limit, static_cast<size_t>(runtime->n_batch));
    }
    if (runtime->n_ctx > 0) {
        const size_t n_ctx = static_cast<size_t>(runtime->n_ctx);
        if (runtime->shared_prefix_len >= n_ctx) {
            return 0;
        }
        limit = std::min(limit, n_ctx - runtime->shared_prefix_len);
    }
    return limit == std::numeric_limits<size_t>::max() ? 0 : limit;
}

int32_t tj_llama_prefill_shared(
    tj_llama_runtime * runtime,
    const uint32_t * prefix_tokens,
    size_t prefix_count,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || prefix_tokens == nullptr || prefix_count == 0) {
        set_error(error, error_capacity, "invalid shared-prefix arguments");
        return -1;
    }
    if (tj_llama_shared_prefix_supported(runtime) == 0) {
        set_error(error, error_capacity, "shared-prefix scoring requires at least two sequences and reusable memory");
        return -1;
    }
    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose reusable sequence memory");
        return -1;
    }

    std::vector<llama_token> prefix;
    std::string failure;
    if (!copy_tokens(prefix_tokens, prefix_count, prefix, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }

    llama_memory_clear(memory, true);
    runtime->shared_prefix_ready = false;
    runtime->shared_prefix_len = 0;
    runtime->shared_prefix_logits.clear();

    llama_batch batch = llama_batch_get_one(prefix.data(), static_cast<int32_t>(prefix.size()));
    const int32_t decode_status = llama_decode(runtime->context, batch);
    if (decode_status != 0) {
        set_error(error, error_capacity, "llama_decode failed while prefilling shared prefix");
        return -1;
    }

    const float * logits = llama_get_logits_ith(runtime->context, -1);
    if (logits == nullptr) {
        set_error(error, error_capacity, "llama_get_logits_ith returned null after shared prefill");
        return -1;
    }
    const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
    if (vocab_size <= 0) {
        set_error(error, error_capacity, "llama.cpp vocabulary is empty");
        return -1;
    }
    runtime->shared_prefix_logits.assign(logits, logits + vocab_size);
    runtime->shared_prefix_len = prefix_count;
    runtime->shared_prefix_ready = true;
    return 0;
}

int32_t tj_llama_score_suffix(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    size_t suffix_count,
    const uint32_t * candidate_tokens,
    size_t candidate_count,
    float * out_logits,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || !runtime->shared_prefix_ready
        || (suffix_tokens == nullptr && suffix_count != 0)
        || candidate_tokens == nullptr || candidate_count < 2 || out_logits == nullptr) {
        set_error(error, error_capacity, "invalid shared-suffix scoring arguments");
        return -1;
    }

    std::string failure;
    if (suffix_count == 0) {
        if (runtime->shared_prefix_logits.empty()) {
            set_error(error, error_capacity, "shared prefix has no retained logits");
            return -1;
        }
        if (!gather_candidate_logits(
                runtime, runtime->shared_prefix_logits.data(), candidate_tokens,
                candidate_count, out_logits, failure)) {
            set_error(error, error_capacity, failure);
            return -1;
        }
        return 0;
    }

    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose reusable sequence memory");
        return -1;
    }

    std::vector<llama_token> suffix;
    if (!copy_tokens(suffix_tokens, suffix_count, suffix, failure)) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    llama_pos start_pos = 0;
    if (!checked_pos(runtime->shared_prefix_len, start_pos)) {
        set_error(error, error_capacity, "shared prefix exceeds llama.cpp position range");
        return -1;
    }
    const size_t max_position = static_cast<size_t>(std::numeric_limits<llama_pos>::max());
    if (suffix.size() > max_position - runtime->shared_prefix_len) {
        set_error(error, error_capacity, "shared prefix plus suffix exceeds llama.cpp position range");
        return -1;
    }

    // Sequence 0 remains the immutable prefix. Sequence 1 is recreated for each
    // serial suffix so one task can never contaminate the next task's cache.
    llama_memory_seq_rm(memory, 1, -1, -1);
    llama_memory_seq_cp(memory, 0, 1, 0, -1);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(suffix.size()), 0, 1);

    batch.n_tokens = static_cast<int32_t>(suffix.size());
    for (size_t i = 0; i < suffix.size(); ++i) {
        batch.token[i] = suffix[i];
        batch.pos[i] = start_pos + static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 1;
        batch.logits[i] = (i + 1 == suffix.size()) ? 1 : 0;
    }

    const int32_t decode_status = llama_decode(runtime->context, batch);
    if (decode_status != 0) {
        llama_batch_free(batch);
        llama_memory_seq_rm(memory, 1, -1, -1);
        set_error(error, error_capacity, "llama_decode failed while scoring shared suffix");
        return -1;
    }
    const float * logits = llama_get_logits_ith(runtime->context, -1);
    if (logits == nullptr) {
        llama_batch_free(batch);
        llama_memory_seq_rm(memory, 1, -1, -1);
        set_error(error, error_capacity, "llama_get_logits_ith returned null after shared suffix");
        return -1;
    }
    const bool gathered = gather_candidate_logits(
        runtime, logits, candidate_tokens, candidate_count, out_logits, failure);
    llama_batch_free(batch);
    llama_memory_seq_rm(memory, 1, -1, -1);
    if (!gathered) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}

int32_t tj_llama_score_suffix_sequences(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    size_t suffix_count,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t candidate_count,
    float * out_scores,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || !runtime->shared_prefix_ready || (suffix_tokens == nullptr && suffix_count != 0)
        || candidate_tokens == nullptr || candidate_offsets == nullptr || candidate_count < 2
        || out_scores == nullptr) {
        set_error(error, error_capacity, "invalid shared-prefix sequence scoring arguments");
        return -1;
    }
    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose sequence memory");
        return -1;
    }
    std::string failure;
    llama_memory_seq_rm(memory, 1, -1, -1);
    llama_memory_seq_cp(memory, 0, 1, 0, -1);
    size_t base_len = runtime->shared_prefix_len;
    std::vector<float> root_logits;

    if (suffix_count == 0) {
        root_logits = runtime->shared_prefix_logits;
    } else {
        std::vector<llama_token> suffix;
        if (!copy_tokens(suffix_tokens, suffix_count, suffix, failure)) {
            llama_memory_seq_rm(memory, 1, -1, -1);
            set_error(error, error_capacity, failure);
            return -1;
        }
        if (runtime->n_ctx > 0 && base_len + suffix.size() > static_cast<size_t>(runtime->n_ctx)) {
            llama_memory_seq_rm(memory, 1, -1, -1);
            set_error(error, error_capacity, "shared prefix plus suffix exceeds context window");
            return -1;
        }
        llama_batch batch = llama_batch_init(static_cast<int32_t>(suffix.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(suffix.size());
        for (size_t i = 0; i < suffix.size(); ++i) {
            llama_pos pos = 0;
            if (!checked_pos(base_len + i, pos)) {
                llama_batch_free(batch);
                llama_memory_seq_rm(memory, 1, -1, -1);
                set_error(error, error_capacity, "shared suffix position overflow");
                return -1;
            }
            batch.token[i] = suffix[i];
            batch.pos[i] = pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 1;
            batch.logits[i] = (i + 1 == suffix.size()) ? 1 : 0;
        }
        const int32_t status = llama_decode(runtime->context, batch);
        llama_batch_free(batch);
        if (status != 0) {
            llama_memory_seq_rm(memory, 1, -1, -1);
            set_error(error, error_capacity, "llama_decode failed before shared sequence scoring");
            return -1;
        }
        const float * root_ptr = llama_get_logits_ith(runtime->context, -1);
        const int32_t vocab_size = llama_vocab_n_tokens(runtime->vocab);
        if (root_ptr == nullptr || vocab_size <= 0) {
            llama_memory_seq_rm(memory, 1, -1, -1);
            set_error(error, error_capacity, "missing shared root logits for sequence scoring");
            return -1;
        }
        root_logits.assign(root_ptr, root_ptr + static_cast<size_t>(vocab_size));
        base_len += suffix.size();
    }

    const bool ok = score_sequence_trie(
        runtime, 1, base_len, root_logits, candidate_tokens, candidate_offsets,
        candidate_count, out_scores, failure);
    llama_memory_seq_rm(memory, 1, -1, -1);
    if (!ok) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}

int32_t tj_llama_score_suffix_batch(
    tj_llama_runtime * runtime,
    const uint32_t * suffix_tokens,
    const size_t * suffix_offsets,
    const uint32_t * candidate_tokens,
    const size_t * candidate_offsets,
    size_t task_count,
    float * out_logits,
    char * error,
    size_t error_capacity) {
    if (runtime == nullptr || runtime->context == nullptr || runtime->vocab == nullptr
        || !runtime->shared_prefix_ready || suffix_offsets == nullptr
        || candidate_offsets == nullptr || candidate_tokens == nullptr || out_logits == nullptr
        || task_count < 2) {
        set_error(error, error_capacity, "invalid parallel shared-suffix arguments");
        return -1;
    }
    const size_t max_parallel = tj_llama_max_parallel_tasks(runtime);
    if (max_parallel < 2 || task_count > max_parallel) {
        set_error(error, error_capacity, "parallel suffix task count exceeds llama.cpp sequence capacity");
        return -1;
    }
    if (task_count > static_cast<size_t>(std::numeric_limits<llama_seq_id>::max() - 1)) {
        set_error(error, error_capacity, "parallel suffix task count exceeds llama.cpp sequence-id range");
        return -1;
    }
    if (suffix_offsets[0] != 0 || candidate_offsets[0] != 0) {
        set_error(error, error_capacity, "parallel suffix offsets must start at zero");
        return -1;
    }

    size_t max_suffix = 0;
    for (size_t task = 0; task < task_count; ++task) {
        if (suffix_offsets[task + 1] < suffix_offsets[task]
            || candidate_offsets[task + 1] < candidate_offsets[task]) {
            set_error(error, error_capacity, "parallel suffix offsets must be monotonic");
            return -1;
        }
        const size_t candidate_count = candidate_offsets[task + 1] - candidate_offsets[task];
        if (candidate_count < 2) {
            set_error(error, error_capacity, "each parallel suffix task requires at least two candidates");
            return -1;
        }
        max_suffix = std::max(max_suffix, suffix_offsets[task + 1] - suffix_offsets[task]);
    }

    const size_t suffix_total = suffix_offsets[task_count];
    if (suffix_total != 0 && suffix_tokens == nullptr) {
        set_error(error, error_capacity, "parallel suffix token buffer is null");
        return -1;
    }
    if (suffix_total > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        set_error(error, error_capacity, "parallel suffix token count exceeds llama.cpp batch range");
        return -1;
    }
    if (runtime->n_batch > 0 && suffix_total > static_cast<size_t>(runtime->n_batch)) {
        set_error(error, error_capacity, "parallel suffix token count exceeds configured n_batch");
        return -1;
    }
    if (runtime->n_ctx > 0) {
        const size_t n_ctx = static_cast<size_t>(runtime->n_ctx);
        if (runtime->shared_prefix_len > n_ctx || max_suffix > n_ctx - runtime->shared_prefix_len) {
            set_error(error, error_capacity, "shared prefix plus suffix exceeds configured n_ctx");
            return -1;
        }
        // The unified context must have room for the shared prefix plus all new
        // branch suffix tokens submitted in this decode call.
        if (suffix_total > n_ctx - runtime->shared_prefix_len) {
            set_error(error, error_capacity, "parallel suffix batch exceeds available context capacity");
            return -1;
        }
    }

    llama_memory_t memory = llama_get_memory(runtime->context);
    if (memory == nullptr) {
        set_error(error, error_capacity, "llama.cpp context does not expose reusable sequence memory");
        return -1;
    }

    auto cleanup_branches = [&]() {
        for (size_t task = 0; task < task_count; ++task) {
            const llama_seq_id seq = static_cast<llama_seq_id>(task + 1);
            llama_memory_seq_rm(memory, seq, -1, -1);
        }
    };

    for (size_t task = 0; task < task_count; ++task) {
        const llama_seq_id seq = static_cast<llama_seq_id>(task + 1);
        llama_memory_seq_rm(memory, seq, -1, -1);
        llama_memory_seq_cp(memory, 0, seq, 0, -1);
    }

    llama_batch batch{};
    std::vector<int32_t> suffix_output_indices(task_count, -1);
    bool batch_allocated = false;
    if (suffix_total > 0) {
        batch = llama_batch_init(static_cast<int32_t>(suffix_total), 0, 1);
        batch.n_tokens = static_cast<int32_t>(suffix_total);
        batch_allocated = true;
        size_t flat_index = 0;
        llama_pos start_pos = 0;
        if (!checked_pos(runtime->shared_prefix_len, start_pos)) {
            cleanup_branches();
            llama_batch_free(batch);
            set_error(error, error_capacity, "shared prefix exceeds llama.cpp position range");
            return -1;
        }

        for (size_t task = 0; task < task_count; ++task) {
            const size_t begin = suffix_offsets[task];
            const size_t end = suffix_offsets[task + 1];
            const llama_seq_id seq = static_cast<llama_seq_id>(task + 1);
            for (size_t local = 0; begin + local < end; ++local, ++flat_index) {
                const uint32_t token = suffix_tokens[begin + local];
                if (token > static_cast<uint32_t>(std::numeric_limits<llama_token>::max())) {
                    cleanup_branches();
                    llama_batch_free(batch);
                    set_error(error, error_capacity, "parallel suffix token id exceeds llama.cpp range");
                    return -1;
                }
                batch.token[flat_index] = static_cast<llama_token>(token);
                batch.pos[flat_index] = start_pos + static_cast<llama_pos>(local);
                batch.n_seq_id[flat_index] = 1;
                batch.seq_id[flat_index][0] = seq;
                const bool wants_logits = begin + local + 1 == end;
                batch.logits[flat_index] = wants_logits ? 1 : 0;
                if (wants_logits) {
                    suffix_output_indices[task] = static_cast<int32_t>(flat_index);
                }
            }
        }

        const int32_t decode_status = llama_decode(runtime->context, batch);
        if (decode_status != 0) {
            cleanup_branches();
            llama_batch_free(batch);
            set_error(error, error_capacity, "llama_decode failed while scoring parallel suffixes");
            return -1;
        }
    }

    std::string failure;
    bool gathered_all = true;
    for (size_t task = 0; task < task_count; ++task) {
        const size_t suffix_count = suffix_offsets[task + 1] - suffix_offsets[task];
        const size_t candidate_begin = candidate_offsets[task];
        const size_t candidate_count = candidate_offsets[task + 1] - candidate_begin;
        const float * logits = nullptr;
        if (suffix_count == 0) {
            if (runtime->shared_prefix_logits.empty()) {
                failure = "shared prefix has no retained logits";
                gathered_all = false;
                break;
            }
            logits = runtime->shared_prefix_logits.data();
        } else {
            const int32_t batch_token_index = suffix_output_indices[task];
            if (batch_token_index < 0) {
                failure = "missing parallel suffix output token index";
                gathered_all = false;
                break;
            }
            logits = llama_get_logits_ith(runtime->context, batch_token_index);
            if (logits == nullptr) {
                failure = "llama_get_logits_ith returned null for a parallel suffix";
                gathered_all = false;
                break;
            }
        }
        if (!gather_candidate_logits(
                runtime,
                logits,
                candidate_tokens + candidate_begin,
                candidate_count,
                out_logits + candidate_begin,
                failure)) {
            gathered_all = false;
            break;
        }
    }

    if (batch_allocated) {
        llama_batch_free(batch);
    }
    cleanup_branches();
    if (!gathered_all) {
        set_error(error, error_capacity, failure);
        return -1;
    }
    return 0;
}

