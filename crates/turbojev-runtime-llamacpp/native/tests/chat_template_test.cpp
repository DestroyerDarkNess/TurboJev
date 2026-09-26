#include "decision_chat.h"
#include "llama.h"

#include <iostream>
#include <stdexcept>

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    // A Qwen-style generation boundary. No model download or inference needed.
    const std::string messages =
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
        "{% endfor %}";
    const std::string generation =
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}";
    const auto now = std::chrono::system_clock::time_point{};
    auto templates = common_chat_templates_init(nullptr, messages + generation);
    const auto prompt = tj_render_decision_chat(templates.get(), "Rules", "Evidence", now);
    const std::string expected = "<|im_start|>system\nRules<|im_end|>\n"
        "<|im_start|>user\nEvidence<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    require(prompt == expected, "must score after the template's closed reasoning block");
    require(prompt == tj_render_decision_chat(templates.get(), "Rules", "Evidence", now),
        "candidate probes must reuse an identical prompt prefix");

    llama_chat_message legacy_messages[] = {{"system", "Rules"}, {"user", "Evidence"}};
    char legacy[1024] = {};
    const int32_t size = llama_chat_apply_template(
        (messages + generation).c_str(), legacy_messages, 2, true, legacy, sizeof(legacy));
    require(size > 0 && std::string(legacy, static_cast<size_t>(size)) != prompt,
        "fixture must expose the legacy template simplification bug");

    templates = common_chat_templates_init(nullptr, messages +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");
    const auto plain = tj_render_decision_chat(templates.get(), nullptr, "Evidence", now);
    require(plain == "<|im_start|>user\nEvidence<|im_end|>\n<|im_start|>assistant\n",
        "non-reasoning templates must not receive invented reasoning markers");
    std::cout << "PASS: real Jinja template preserves the direct-answer boundary\n";
}
