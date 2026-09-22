cmake_minimum_required(VERSION 3.20)
project(turbojev_llama_shim LANGUAGES C CXX)

if(NOT DEFINED LLAMA_CPP_SOURCE_DIR)
  message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR must point to a llama.cpp checkout")
endif()
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR does not contain llama.cpp")
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if(WIN32)
  # MSVC OpenMP can retain threads executing unloaded DLL code when a Node
  # worker exits. Use ggml's native thread pool for unloadable Windows hosts.
  set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for Windows worker teardown")
endif()

add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_library(turbojev_llama_shim SHARED turbojev_llama_shim.cpp)
target_compile_features(turbojev_llama_shim PRIVATE cxx_std_17)
target_include_directories(turbojev_llama_shim
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${LLAMA_CPP_SOURCE_DIR}/include")
target_link_libraries(turbojev_llama_shim PRIVATE llama)
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
  target_compile_options(turbojev_llama_shim PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
