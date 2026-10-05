#pragma once

#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>

using common_json = nlohmann::ordered_json;

// Compile the complete production shim without pretending to implement llama's
// grammar compiler. Real generation is tested by the installed-package gates.
inline std::string json_schema_to_grammar(const common_json &, bool = false) {
    throw std::runtime_error("grammar generation unavailable in ABI stub");
}
