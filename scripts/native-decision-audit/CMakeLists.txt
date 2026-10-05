cmake_minimum_required(VERSION 3.20)
project(turbojev_direct_audit LANGUAGES CXX)
if(NOT DEFINED LLAMA_SOURCE OR NOT DEFINED LLAMA_BUILD)
  message(FATAL_ERROR "Set LLAMA_SOURCE and LLAMA_BUILD to the pinned llama.cpp source and existing Windows x64 build")
endif()
add_executable(direct_decision direct_decision.cpp)
target_compile_features(direct_decision PRIVATE cxx_std_17)
target_include_directories(direct_decision PRIVATE "${LLAMA_SOURCE}/include" "${LLAMA_SOURCE}/common" "${LLAMA_SOURCE}/ggml/include" "${LLAMA_SOURCE}/vendor")
target_link_libraries(direct_decision PRIVATE
  "${LLAMA_BUILD}/llama.cpp/src/Release/llama.lib"
  "${LLAMA_BUILD}/llama.cpp/common/Release/llama-common.lib"
  "${LLAMA_BUILD}/llama.cpp/ggml/src/Release/ggml.lib"
  "${LLAMA_BUILD}/llama.cpp/ggml/src/Release/ggml-base.lib")
