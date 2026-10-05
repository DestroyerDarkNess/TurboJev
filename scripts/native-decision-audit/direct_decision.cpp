// Diagnostic executable only. It uses llama.cpp directly, never TurboJev scoring.
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int count = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, true);
    std::vector<llama_token> tokens(size_t(-count));
    count = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), true, true);
    if(count < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(size_t(count));
    return tokens;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(128);
    int n = llama_token_to_piece(vocab, token, buffer.data(), int(buffer.size()), 0, true);
    if(n < 0) {
        buffer.resize(size_t(-n));
        n = llama_token_to_piece(vocab, token, buffer.data(), int(buffer.size()), 0, true);
    }
    if(n < 0) throw std::runtime_error("detokenization failed");
    return std::string(buffer.data(), size_t(n));
}

void prefill(llama_context * ctx, std::vector<llama_token> & tokens) {
    if(tokens.size() + 64 >= llama_n_ctx_seq(ctx)) throw std::runtime_error("context budget too small");
    llama_memory_clear(llama_get_memory(ctx), true);
    size_t offset = 0;
    while(offset < tokens.size()) {
        const size_t count = std::min(tokens.size() - offset, size_t(llama_n_batch(ctx)));
        auto batch = llama_batch_get_one(tokens.data() + offset, int(count));
        if(llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill decode failed");
        offset += count;
    }
}

json generate(llama_context * ctx, const llama_vocab * vocab, const std::string & prefix) {
    std::string text = prefix;
    std::vector<llama_token> generated;
    std::string stop = "token_limit";
    for(int n=0; n<32; ++n) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if(!logits) throw std::runtime_error("missing generation logits");
        const auto token = llama_token(std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits);
        if(llama_vocab_is_eog(vocab, token)) { stop="eog"; break; }
        generated.push_back(token);
        text += piece(vocab, token);
        // Stop only once the complete requested JSON object is present.
        if(text.find('}') != std::string::npos) { stop="json_end"; break; }
        auto input = token;
        auto batch = llama_batch_get_one(&input, 1);
        if(llama_decode(ctx, batch) != 0) throw std::runtime_error("generation decode failed");
    }
    return {{"text",text},{"generated_token_ids",generated},{"stop_reason",stop}};
}

int main(int argc, char ** argv) {
    if(argc != 4) { std::cerr << "Usage: direct_decision model.gguf inputs.jsonl outputs.jsonl\n"; return 2; }
    try {
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers=0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if(!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=512; cp.n_seq_max=4; cp.no_perf=true;
        auto * ctx = llama_init_from_model(model, cp);
        if(!ctx) throw std::runtime_error("context load failed");
        llama_set_n_threads(ctx,4,4);
        auto * vocab = llama_model_get_vocab(model);
        auto templates = common_chat_templates_init(model, llama_model_chat_template(model,nullptr));
        std::ifstream in(argv[2]); std::ofstream out(argv[3]);
        if(!in || !out) throw std::runtime_error("input/output open failed");
        std::string line;
        while(std::getline(in,line)) {
            const auto request=json::parse(line);
            json result={{"id",request.at("id")}};
            try {
                common_chat_templates_inputs inputs;
                inputs.use_jinja=true; inputs.add_generation_prompt=true;
                inputs.enable_thinking=false; inputs.chat_template_kwargs["enable_thinking"]="false";
                inputs.now=std::chrono::system_clock::time_point{};
                common_chat_msg system; system.role="system"; system.content=request.at("system").get<std::string>();
                common_chat_msg user; user.role="user"; user.content=request.at("user").get<std::string>();
                inputs.messages={system,user};
                const auto rendered=common_chat_templates_apply(templates.get(), inputs).prompt;
                const std::string prefix=request.at("assistant_prefix");
                const auto started=Clock::now();
                auto tokens=tokenize(vocab,rendered+prefix);
                std::vector<llama_token> ids;
                for(const auto & code:request.at("codes")) {
                    auto extended=tokenize(vocab,rendered+prefix+code.get<std::string>());
                    if(extended.size()!=tokens.size()+1 || !std::equal(tokens.begin(),tokens.end(),extended.begin()))
                        throw std::runtime_error("candidate does not extend the exact prompt by one token");
                    ids.push_back(extended.back());
                }
                prefill(ctx,tokens);
                const auto * logits=llama_get_logits_ith(ctx,-1);
                std::vector<float> scores;
                for(auto id:ids) scores.push_back(logits[id]);
                const auto best=std::max_element(scores.begin(),scores.end())-scores.begin();
                std::vector<double> probabilities; double sum=0;
                for(float score:scores) { probabilities.push_back(std::exp(double(score-scores[size_t(best)]))); sum+=probabilities.back(); }
                for(auto & p:probabilities) p/=sum;
                result["rendered_chat"]=rendered;
                result["exact_prefill"]={{"prompt_token_ids",tokens},{"candidate_token_ids",ids},
                    {"candidate_logits",scores},{"probabilities",probabilities},{"restricted_choice_code",request["codes"][best]}};
                result["exact_prefill"]["generation"]=generate(ctx,vocab,prefix);
                result["exact_prefill"]["seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
                const auto free_started=Clock::now();
                auto free_tokens=tokenize(vocab,rendered);
                prefill(ctx,free_tokens);
                result["free_generation"]=generate(ctx,vocab,"");
                result["free_generation"]["seconds"]=std::chrono::duration<double>(Clock::now()-free_started).count();
            } catch(const std::exception & e) { result["error"]=e.what(); }
            out << result.dump() << std::endl;
            std::cerr << "completed " << request.at("id") << std::endl;
        }
        llama_free(ctx); llama_model_free(model); llama_backend_free();
    } catch(const std::exception & e) { std::cerr << e.what() << std::endl; return 1; }
}
