#include "llama.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct llama_model {};
struct llama_vocab {};
struct llama_context;
struct llama_memory_i {
    llama_context * context = nullptr;
};
struct llama_context {
    std::vector<float> logits = std::vector<float>(4096, 0.0f);
    std::vector<std::vector<float>> output_logits;
    std::vector<int32_t> output_ids;
    std::map<llama_seq_id, std::vector<llama_token>> sequences;
    llama_memory_i memory;
    uint32_t n_ctx = 2048;
    uint32_t n_batch = 512;
    uint32_t n_seq_max = 1;

    llama_context() { memory.context = this; }
};

static llama_vocab STUB_VOCAB;

llama_model_params llama_model_default_params(void) { return {0}; }
llama_context_params llama_context_default_params(void) { return {2048, 512, 1, false}; }
void llama_backend_init(void) {}
llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model(); }
void llama_model_free(llama_model * model) { delete model; }
const llama_vocab * llama_model_get_vocab(const llama_model *) { return &STUB_VOCAB; }
const char * llama_model_chat_template(const llama_model *, const char *) { return "stub-template"; }
llama_context * llama_init_from_model(llama_model *, llama_context_params params) {
    auto * context = new llama_context();
    context->n_ctx = params.n_ctx;
    context->n_batch = params.n_batch;
    context->n_seq_max = params.n_seq_max;
    return context;
}
void llama_free(llama_context * context) { delete context; }
void llama_set_n_threads(llama_context *, int32_t, int32_t) {}
uint32_t llama_n_ctx(const llama_context * context) { return context == nullptr ? 0 : context->n_ctx; }
uint32_t llama_n_ctx_seq(const llama_context * context) {
    if (context == nullptr || context->n_seq_max == 0) return 0;
    return context->n_ctx / context->n_seq_max;
}
uint32_t llama_n_batch(const llama_context * context) { return context == nullptr ? 0 : context->n_batch; }
uint32_t llama_n_seq_max(const llama_context * context) { return context == nullptr ? 0 : context->n_seq_max; }

int32_t llama_chat_apply_template(
    const char *, const llama_chat_message * chat, size_t n_msg, bool add_ass,
    char * out, int32_t capacity) {
    std::string rendered;
    for (size_t i = 0; i < n_msg; ++i) {
        rendered += "<";
        rendered += chat[i].role;
        rendered += ">";
        rendered += chat[i].content;
        rendered += "</";
        rendered += chat[i].role;
        rendered += ">";
    }
    if (add_ass) {
        rendered += "<assistant>";
    }
    if (out == nullptr || capacity <= 0) {
        return static_cast<int32_t>(rendered.size());
    }
    if (capacity <= static_cast<int32_t>(rendered.size())) {
        return static_cast<int32_t>(rendered.size());
    }
    std::memcpy(out, rendered.data(), rendered.size());
    out[rendered.size()] = '\0';
    return static_cast<int32_t>(rendered.size());
}

int32_t llama_tokenize(
    const llama_vocab *, const char * text, int32_t text_len, llama_token * out,
    int32_t capacity, bool add_special, bool) {
    const int32_t required = text_len + (add_special ? 1 : 0);
    if (out == nullptr || capacity < required) {
        return -required;
    }
    int32_t index = 0;
    if (add_special) {
        out[index++] = 1;
    }
    for (int32_t i = 0; i < text_len; ++i) {
        out[index++] = 100 + static_cast<unsigned char>(text[i]);
    }
    return required;
}

int32_t llama_detokenize(
    const llama_vocab *, const llama_token * tokens, int32_t n_tokens, char * out,
    int32_t capacity, bool, bool) {
    std::string decoded;
    for (int32_t i = 0; i < n_tokens; ++i) {
        if (tokens[i] == 1) {
            continue;
        }
        if (tokens[i] >= 100 && tokens[i] <= 355) {
            decoded.push_back(static_cast<char>(tokens[i] - 100));
        }
    }
    if (out == nullptr || capacity < static_cast<int32_t>(decoded.size())) {
        return -static_cast<int32_t>(decoded.size());
    }
    std::memcpy(out, decoded.data(), decoded.size());
    return static_cast<int32_t>(decoded.size());
}

int32_t llama_model_desc(const llama_model *, char * out, size_t capacity) {
    const std::string value = "stub-llama-model";
    if (capacity == 0) {
        return static_cast<int32_t>(value.size());
    }
    const size_t count = std::min(value.size(), capacity - 1);
    std::memcpy(out, value.data(), count);
    out[count] = '\0';
    return static_cast<int32_t>(count);
}

llama_memory_t llama_get_memory(const llama_context * context) {
    return context == nullptr ? nullptr : const_cast<llama_memory_i *>(&context->memory);
}

void llama_memory_clear(llama_memory_t memory, bool) {
    if (memory != nullptr && memory->context != nullptr) {
        memory->context->sequences.clear();
        memory->context->output_logits.clear();
        memory->context->output_ids.clear();
    }
}

bool llama_memory_seq_rm(llama_memory_t memory, llama_seq_id seq, llama_pos p0, llama_pos p1) {
    if (memory == nullptr || memory->context == nullptr) {
        return false;
    }
    auto & sequences = memory->context->sequences;
    auto found = sequences.find(seq);
    if (found == sequences.end()) {
        return true;
    }
    if (p0 < 0 && p1 < 0) {
        sequences.erase(found);
        return true;
    }
    auto & values = found->second;
    const size_t begin = p0 < 0 ? 0 : static_cast<size_t>(p0);
    const size_t end = p1 < 0 ? values.size() : std::min(values.size(), static_cast<size_t>(p1));
    if (begin < end && begin < values.size()) {
        values.erase(values.begin() + static_cast<std::ptrdiff_t>(begin), values.begin() + static_cast<std::ptrdiff_t>(end));
    }
    return true;
}

void llama_memory_seq_cp(
    llama_memory_t memory, llama_seq_id source, llama_seq_id destination,
    llama_pos p0, llama_pos p1) {
    if (memory == nullptr || memory->context == nullptr) {
        return;
    }
    const auto found = memory->context->sequences.find(source);
    if (found == memory->context->sequences.end()) {
        memory->context->sequences[destination].clear();
        return;
    }
    const auto & values = found->second;
    const size_t begin = p0 < 0 ? 0 : std::min(values.size(), static_cast<size_t>(p0));
    const size_t end = p1 < 0 ? values.size() : std::min(values.size(), static_cast<size_t>(p1));
    memory->context->sequences[destination] = std::vector<llama_token>(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() + static_cast<std::ptrdiff_t>(end));
}

llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens) {
    return {n_tokens, tokens, nullptr, nullptr, nullptr, nullptr, nullptr};
}

llama_batch llama_batch_init(int32_t n_tokens, int32_t, int32_t n_seq_max) {
    llama_batch batch{};
    batch.n_tokens = n_tokens;
    batch.token = new llama_token[static_cast<size_t>(n_tokens)]{};
    batch.pos = new llama_pos[static_cast<size_t>(n_tokens)]{};
    batch.n_seq_id = new int32_t[static_cast<size_t>(n_tokens)]{};
    batch.seq_id = new llama_seq_id *[static_cast<size_t>(n_tokens)]{};
    batch.logits = new int8_t[static_cast<size_t>(n_tokens)]{};
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.seq_id[i] = new llama_seq_id[static_cast<size_t>(n_seq_max)]{};
    }
    return batch;
}

void llama_batch_free(llama_batch batch) {
    if (batch.seq_id != nullptr) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            delete[] batch.seq_id[i];
        }
    }
    delete[] batch.token;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    delete[] batch.seq_id;
    delete[] batch.logits;
}

static std::vector<float> logits_for_seq(const llama_context * context, llama_seq_id seq) {
    const auto found = context->sequences.find(seq);
    long long checksum = 0;
    size_t length = 0;
    if (found != context->sequences.end()) {
        length = found->second.size();
        for (llama_token token : found->second) {
            checksum += token;
        }
    }
    const float offset = static_cast<float>(checksum % 997) / 997.0f;
    const float length_term = static_cast<float>(length % 31) / 31.0f;
    std::vector<float> logits(4096, -10.0f + offset);
    // Candidate rows intentionally depend differently on branch state. This
    // makes cache contamination/output-row mixups visible to the native smoke
    // tests instead of cancelling out as a constant offset.
    logits[100 + static_cast<int>('A')] = 1.0f + offset + 0.25f * length_term;
    logits[100 + static_cast<int>('B')] = 4.0f + 1.5f * offset - 0.10f * length_term;
    logits[100 + static_cast<int>('C')] = 0.5f - 0.75f * offset + 0.20f * length_term;
    return logits;
}

int32_t llama_decode(llama_context * context, llama_batch batch) {
    if (context == nullptr || batch.n_tokens <= 0 || batch.token == nullptr) {
        return -1;
    }
    context->output_logits.clear();
    context->output_ids.assign(static_cast<size_t>(batch.n_tokens), -1);
    llama_seq_id last_seq = 0;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        llama_seq_id seq = 0;
        llama_pos pos = static_cast<llama_pos>(context->sequences[0].size());
        if (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[i] > 0) {
            seq = batch.seq_id[i][0];
        }
        if (seq < 0 || static_cast<uint32_t>(seq) >= context->n_seq_max) {
            return -1;
        }
        auto & values = context->sequences[seq];
        if (batch.pos != nullptr) {
            pos = batch.pos[i];
        } else {
            pos = static_cast<llama_pos>(values.size());
        }
        if (pos < 0) {
            return -1;
        }
        const size_t index = static_cast<size_t>(pos);
        if (values.size() <= index) {
            values.resize(index + 1, 0);
        }
        values[index] = batch.token[i];
        last_seq = seq;

        if (batch.logits != nullptr && batch.logits[i] != 0) {
            context->output_ids[static_cast<size_t>(i)] =
                static_cast<int32_t>(context->output_logits.size());
            context->output_logits.push_back(logits_for_seq(context, seq));
        }
    }

    context->logits = logits_for_seq(context, last_seq);
    if (batch.logits == nullptr) {
        const int32_t last_index = batch.n_tokens - 1;
        context->output_ids[static_cast<size_t>(last_index)] = 0;
        context->output_logits.push_back(context->logits);
    }
    return 0;
}

float * llama_get_logits_ith(llama_context * context, int32_t index) {
    if (context == nullptr || context->output_logits.empty()) {
        return nullptr;
    }
    if (index < 0) {
        const int32_t output_index =
            static_cast<int32_t>(context->output_logits.size()) + index;
        if (output_index < 0
            || static_cast<size_t>(output_index) >= context->output_logits.size()) {
            return nullptr;
        }
        return context->output_logits[static_cast<size_t>(output_index)].data();
    }
    if (static_cast<size_t>(index) >= context->output_ids.size()) {
        return nullptr;
    }
    const int32_t output_index = context->output_ids[static_cast<size_t>(index)];
    if (output_index < 0
        || static_cast<size_t>(output_index) >= context->output_logits.size()) {
        return nullptr;
    }
    return context->output_logits[static_cast<size_t>(output_index)].data();
}

int32_t llama_vocab_n_tokens(const llama_vocab *) { return 4096; }
