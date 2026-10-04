#ifndef TURBOJEV_TEST_CHAT_H
#define TURBOJEV_TEST_CHAT_H

#include "llama.h"
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Only for the ABI/scheduling smoke test; Jinja behavior is covered by
// chat_template_test.cpp linked to the actual pinned llama-common library.
struct common_chat_templates {};
using common_chat_templates_ptr = std::unique_ptr<common_chat_templates>;
struct common_chat_msg { std::string role; std::string content; };
struct common_chat_templates_inputs {
    std::vector<common_chat_msg> messages;
    bool use_jinja = false;
    bool add_generation_prompt = false;
    bool enable_thinking = true;
    std::map<std::string, std::string> chat_template_kwargs;
    std::chrono::system_clock::time_point now;
};
struct common_chat_params { std::string prompt; bool supports_thinking = false; std::vector<std::string> thinking_end_tags; };
inline common_chat_templates_ptr common_chat_templates_init(const llama_model *, const std::string &) {
    return std::make_unique<common_chat_templates>();
}
inline common_chat_params common_chat_templates_apply(
    const common_chat_templates *, const common_chat_templates_inputs & inputs) {
    if (!inputs.use_jinja || inputs.enable_thinking || !inputs.add_generation_prompt
        || inputs.chat_template_kwargs.at("enable_thinking") != "false") {
        throw std::runtime_error("decision adapter must request a direct-answer Jinja prompt");
    }
    std::string rendered;
    for (const auto & message : inputs.messages) {
        rendered += "<" + message.role + ">" + message.content + "</" + message.role + ">";
    }
    common_chat_params params;
    params.prompt = rendered + "<assistant>";
    return params;
}
#endif
