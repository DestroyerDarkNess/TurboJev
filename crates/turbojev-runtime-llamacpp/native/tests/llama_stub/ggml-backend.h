#ifndef TURBOJEV_TEST_GGML_BACKEND_H
#define TURBOJEV_TEST_GGML_BACKEND_H

enum ggml_backend_dev_type { GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU };
struct ggml_backend_device;
using ggml_backend_dev_t = ggml_backend_device *;
inline ggml_backend_dev_t ggml_backend_dev_by_type(ggml_backend_dev_type) { return nullptr; }

#endif
