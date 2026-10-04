#include "turbojev.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_all(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f); if (n < 0) { fclose(f); return NULL; }
    rewind(f); char *p = (char *) malloc((size_t)n + 1); if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    p[n] = '\0'; fclose(f); return p;
}

static int append_text(char *out, size_t capacity, size_t *used, const char *text) {
    size_t length = strlen(text);
    if (length >= capacity - *used) return 0;
    memcpy(out + *used, text, length);
    *used += length;
    out[*used] = '\0';
    return 1;
}

static int append_json_string(char *out, size_t capacity, size_t *used, const char *value) {
    if (!append_text(out, capacity, used, "\"")) return 0;
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        char escaped[3] = {0, 0, 0};
        if (*p < 32) return 0;
        if (*p == '\\' || *p == '"') {
            escaped[0] = '\\';
            escaped[1] = (char)*p;
            if (!append_text(out, capacity, used, escaped)) return 0;
        } else {
            escaped[0] = (char)*p;
            if (!append_text(out, capacity, used, escaped)) return 0;
        }
    }
    return append_text(out, capacity, used, "\"");
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) { fprintf(stderr, "usage: %s <model> <request.json> [mmproj.gguf]\n", argv[0]); return 2; }
    char options[16384] = {0}; size_t used = 0;
    if (!append_text(options, sizeof options, &used, "{\"model\":" ) ||
        !append_json_string(options, sizeof options, &used, argv[1]) ||
        !append_text(options, sizeof options, &used, ",\"nGpuLayers\":0")) {
        fprintf(stderr, "model path is too long or contains unsupported characters\n"); return 2;
    }
    if (argc == 4 && argv[3][0] != '\0' &&
        (!append_text(options, sizeof options, &used, ",\"mmproj\":" ) ||
         !append_json_string(options, sizeof options, &used, argv[3]))) {
        fprintf(stderr, "projector path is too long or contains unsupported characters\n"); return 2;
    }
    if (!append_text(options, sizeof options, &used, "}")) { fprintf(stderr, "load options are too large\n"); return 2; }
    char *request = read_all(argv[2]); if (!request) { fprintf(stderr, "cannot read request\n"); return 2; }
    tj_engine *engine = NULL; char *out = NULL;
    int32_t rc = tj_engine_open_json(options, &engine);
    if (rc != TJ_OK) { fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message()); free(request); return 1; }
    rc = tj_engine_evaluate_json(engine, request, &out); free(request);
    if (rc != TJ_OK) { fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message()); tj_engine_close(engine); return 1; }
    if (!strstr(out, "\"answers\"")) {
        fprintf(stderr, "missing answers object: %s\n", out);
        tj_string_free(out); tj_engine_close(engine); return 1;
    }
    printf("{\"language\":\"c\",\"response\":%s}\n", out);
    tj_string_free(out); tj_engine_close(engine); return 0;
}
