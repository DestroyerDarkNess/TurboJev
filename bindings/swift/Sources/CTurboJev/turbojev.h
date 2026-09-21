#ifndef TURBOJEV_H
#define TURBOJEV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tj_engine tj_engine;

enum {
  TJ_OK = 0,
  TJ_E_INVALID_ARGUMENT = 1,
  TJ_E_RUNTIME = 2,
  TJ_E_THREAD_AFFINITY = 3,
  TJ_E_PANIC = 255
};

uint32_t tj_abi_version(void);
int32_t tj_engine_open_json(const char *options_json, tj_engine **out_engine);
int32_t tj_engine_evaluate_json(tj_engine *engine, const char *request_json, char **out_json);
int32_t tj_engine_evaluate_compatible_json(tj_engine *engine, const char *profile, const char *request_json, char **out_json);
int32_t tj_engine_classify_json(tj_engine *engine, const char *state, const char *options_json, char **out_json);
int32_t tj_engine_info_json(tj_engine *engine, char **out_json);
int32_t tj_engine_close(tj_engine *engine);
void tj_string_free(char *value);
const char *tj_last_error_code(void);
const char *tj_last_error_message(void);

#ifdef __cplusplus
}
#endif
#endif
