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

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model> <request.json>\n", argv[0]); return 2; }
    char options[8192];
    char escaped[7100]; size_t used = 0;
    for (const unsigned char *p = (const unsigned char *)argv[1]; *p; ++p) {
        if (*p < 32 || used + 2 >= sizeof escaped) { fprintf(stderr, "unsupported model path\n"); return 2; }
        if (*p == '\\' || *p == '"') escaped[used++] = '\\';
        escaped[used++] = (char)*p;
    }
    escaped[used] = '\0';
    snprintf(options, sizeof options, "{\"model\":\"%s\",\"nGpuLayers\":0}", escaped);
    char *request = read_all(argv[2]); if (!request) { fprintf(stderr, "cannot read request\n"); return 2; }
    tj_engine *engine = NULL; char *out = NULL;
    int32_t rc = tj_engine_open_json(options, &engine);
    if (rc != TJ_OK) { fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message()); free(request); return 1; }
    rc = tj_engine_evaluate_json(engine, request, &out); free(request);
    if (rc != TJ_OK) { fprintf(stderr, "%s: %s\n", tj_last_error_code(), tj_last_error_message()); tj_engine_close(engine); return 1; }
    if (!strstr(out, "\"route\"") || !strstr(out, "\"choice\"") ||
        !strstr(out, "\"urgent\"") || !strstr(out, "\"noul\"") ||
        !strstr(out, "\"severity\"") || !strstr(out, "\"score\"")) {
        fprintf(stderr, "missing typed E2E answers: %s\n", out);
        tj_string_free(out); tj_engine_close(engine); return 1;
    }
    printf("{\"language\":\"c\",\"response\":%s}\n", out);
    tj_string_free(out); tj_engine_close(engine); return 0;
}
