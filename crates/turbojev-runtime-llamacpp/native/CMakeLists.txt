cmake_minimum_required(VERSION 3.20)
project(turbojev_llama_shim LANGUAGES C CXX)

if(NOT DEFINED LLAMA_CPP_SOURCE_DIR)
  message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR must point to a llama.cpp checkout")
endif()
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR does not contain llama.cpp")
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build the GGUF Jinja chat renderer" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "TurboJev does not launch external media tools" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if(WIN32)
  # MSVC OpenMP can retain threads executing unloaded DLL code when a Node
  # worker exits. Use ggml's native thread pool for unloadable Windows hosts.
  set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for Windows worker teardown")
endif()

option(TURBOJEV_BUILD_NATIVE_TESTS "Build model-free chat template regression tests" OFF)
if(TURBOJEV_BUILD_NATIVE_TESTS)
  enable_testing()
  add_executable(turbojev_chat_template_test tests/chat_template_test.cpp)
  target_include_directories(turbojev_chat_template_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(turbojev_chat_template_test PRIVATE llama-common llama)
  set_target_properties(turbojev_chat_template_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  add_test(NAME turbojev_chat_template COMMAND turbojev_chat_template_test)
endif()
if(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
  # ggml's BMI2 implementation can reference the 64-bit PDEP intrinsic on x86 MSVC.
  set(GGML_BMI2 OFF CACHE BOOL "Disable BMI2 for 32-bit Windows" FORCE)
endif()

add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)
set(MTMD_VIDEO OFF CACHE BOOL "TurboJev accepts host-decoded timestamped video frames" FORCE)
add_subdirectory("${LLAMA_CPP_SOURCE_DIR}/tools/mtmd" "${CMAKE_BINARY_DIR}/llama.cpp/tools/mtmd" EXCLUDE_FROM_ALL)

add_library(turbojev_llama_shim SHARED turbojev_llama_shim.cpp)
target_compile_features(turbojev_llama_shim PRIVATE cxx_std_17)
target_compile_definitions(turbojev_llama_shim PRIVATE TURBOJEV_HAS_MTMD=1)
target_include_directories(turbojev_llama_shim
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${LLAMA_CPP_SOURCE_DIR}/include" "${LLAMA_CPP_SOURCE_DIR}/vendor")
target_include_directories(turbojev_llama_shim PRIVATE "${LLAMA_CPP_SOURCE_DIR}/tools/mtmd")
target_link_libraries(turbojev_llama_shim PRIVATE llama llama-common mtmd)
set_target_properties(turbojev_llama_shim PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(MSVC)
  target_compile_options(turbojev_llama_shim PRIVATE /W4 /permissive-)
else()
  # llama.cpp's common/Jinja headers contain static helpers unused by this
  # translation unit. Keep warnings fatal for the shim while allowing that
  # single upstream-header warning on GCC/Clang.
  target_compile_options(turbojev_llama_shim PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-error=unused-function)
endif()
