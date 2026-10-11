#include "turbojev.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int32_t status) {
    if (status != TJ_OK) {
        fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message());
        exit(1);
    }
}

int main(void) {
    tj_engine *engine = NULL;
    tj_generation_control *control = NULL;
    char *output = NULL;
    const char *request = "{\"context\":\"facts\",\"questions\":{\"count\":{\"type\":\"integer\",\"enum\":[7]}}}";

    if (tj_abi_version() == 0) return 1;
    check(tj_candidate_softmax_json("{\"logits\":{\"a\":10000,\"b\":10000},\"temperature\":1}", &output));
    if (strstr(output, "0.5") == NULL) return 1;
    tj_string_free(output);
    check(tj_compile_schema_json("{\"type\":\"object\",\"properties\":{\"label\":{\"type\":\"string\",\"enum\":[\"niño 🐈 中文\"]}}}", &output));
    tj_string_free(output);
    check(tj_engine_open_json("{\"model\":\"fixture://c-compatibility\"}", &engine));
    check(tj_engine_generate_json(engine, request, &output));
    if (strstr(output, "\"count\":7") == NULL) return 1;
    tj_string_free(output);
    check(tj_generation_control_open(0, &control));
    if (tj_engine_generate_json_controlled(engine, request, control, &output) == TJ_OK) return 1;
    if (strcmp(tj_last_error_code(), "GENERATION_TIMEOUT") != 0) return 1;
    tj_generation_control_free(control);
    check(tj_engine_close(engine));
    printf("{\"language\":\"c\",\"bits\":%u,\"abi\":%u,\"passed\":true}\n",
           (unsigned)(sizeof(void *) * 8), tj_abi_version());
    return 0;
}
