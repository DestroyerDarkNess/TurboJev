#ifndef TURBOJEV_DECISION_CHAT_H
#define TURBOJEV_DECISION_CHAT_H

#include "chat.h"

// Decision logits must be read at the final-answer boundary. The legacy
// llama_chat_apply_template API recognizes template families rather than
// evaluating GGUF Jinja, and cannot pass enable_thinking=false.
inline std::string tj_render_decision_chat(
    const common_chat_templates * templates,
    const char * system,
    const char * user,
    std::chrono::system_clock::time_point now) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = false;
    inputs.chat_template_kwargs["enable_thinking"] = "false";
    inputs.now = now;
    if (system != nullptr && system[0] != '\0') {
        common_chat_msg message;
        message.role = "system";
        message.content = system;
        inputs.messages.push_back(std::move(message));
    }
    common_chat_msg message;
    message.role = "user";
    message.content = user;
    inputs.messages.push_back(std::move(message));
    return common_chat_templates_apply(templates, inputs).prompt;
}

#endif
