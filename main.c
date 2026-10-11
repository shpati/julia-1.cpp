/* julia-c: command line front-end for the pure-C Julia-1 engine. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "julia.h"

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s MODEL_DIR [options] [request.json]\n"
            "  Reads a request (file or stdin) and prints the JSON result.\n"
            "  typed request : {\"state\": \"...\", \"questions\": {\"id\": {\"type\": \"choice|score|noul\",\n"
            "                   \"instructions\": \"...\", \"criteria\": {...} | [...]}}}\n"
            "  legacy request: [{\"state\": \"...\", \"question\": \"...\", \"options\": [...], \"type\": \"choice\"}]\n"
            "options:\n"
            "  --legacy            request is a legacy rows array\n"
            "  --serve             keep the model loaded; one JSON request per stdin line, one JSON line out\n"
            "  --max-length N      combined token limit (default 8192)\n"
            "  --head-length N     question+options token budget (default 512)\n"
            "  --no-strict         truncate instead of rejecting overflow\n"
            "  --threads N         OpenMP threads (if built with OpenMP)\n"
            "  --debug             add token ids / marker positions / raw logits to the output\n"
            "  --info              print model + tokenizer summary and exit\n"
            "  --list-tensors      list tensors found in model.safetensors and exit\n"
            "  --tokens TEXT       print token ids for TEXT (add_special_tokens=False) and exit\n",
            argv0);
}

static char *read_all(FILE *f, size_t *n) {
    size_t cap = 1 << 16, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    size_t got;
    while ((got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (cap - len < 2) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
    }
    buf[len] = 0;
    *n = len;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    JuliaOptions opt = julia_default_options();
    const char *dir = NULL, *reqfile = NULL, *tokens = NULL;
    int legacy = 0, serve = 0, info = 0, list = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--legacy")) legacy = 1;
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--no-strict")) opt.strict = 0;
        else if (!strcmp(a, "--debug")) opt.debug = 1;
        else if (!strcmp(a, "--info")) info = 1;
        else if (!strcmp(a, "--list-tensors")) list = 1;
        else if (!strcmp(a, "--max-length") && i + 1 < argc) opt.max_length = atoi(argv[++i]);
        else if (!strcmp(a, "--head-length") && i + 1 < argc) opt.head_length = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && i + 1 < argc) opt.threads = atoi(argv[++i]);
        else if (!strcmp(a, "--tokens") && i + 1 < argc) tokens = argv[++i];
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else if (a[0] == '-' && a[1]) { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 2; }
        else if (!dir) dir = a;
        else reqfile = a;
    }
    if (!dir) { usage(argv[0]); return 2; }
    char err[512] = {0};
    JuliaModel *m = julia_load(dir, &opt, err, sizeof err);
    if (!m) { fprintf(stderr, "error: %s\n", err); return 1; }
    if (info) julia_describe(m, stdout);
    if (list) julia_list_tensors(m, stdout);
    if (info || list) { julia_free(m); return 0; }
    if (tokens) {
        int32_t *ids = NULL;
        size_t n = 0;
        if (julia_tokenize(m, tokens, strlen(tokens), &ids, &n, err, sizeof err)) { fprintf(stderr, "error: %s\n", err); julia_free(m); return 1; }
        printf("[");
        for (size_t i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
        printf("]\n");
        free(ids);
        julia_free(m);
        return 0;
    }
    int rc = 0;
    if (serve) {
        size_t cap = 1 << 16, len = 0;
        char *line = (char *)malloc(cap);
        for (;;) {
            len = 0;
            int got_any = 0, c;
            while ((c = fgetc(stdin)) != EOF && c != '\n') {
                got_any = 1;
                if (len + 2 > cap) { cap *= 2; line = (char *)realloc(line, cap); }
                line[len++] = (char)c;
            }
            if (!got_any && c == EOF) break;
            line[len] = 0;
            if (len == 0) continue;
            char *res = julia_predict_json(m, line, len, legacy, err, sizeof err);
            if (res) { puts(res); free(res); }
            else {
                printf("{\"error\": \"");
                for (const char *p = err; *p; p++) {
                    if (*p == '"' || *p == '\\') putchar('\\');
                    if ((unsigned char)*p >= 0x20) putchar(*p);
                }
                printf("\"}\n");
            }
            fflush(stdout);
        }
        free(line);
    } else {
        FILE *f = reqfile && strcmp(reqfile, "-") ? fopen(reqfile, "rb") : stdin;
        if (!f) { fprintf(stderr, "error: cannot open %s\n", reqfile); julia_free(m); return 1; }
        size_t n;
        char *text = read_all(f, &n);
        if (f != stdin) fclose(f);
        char *res = text ? julia_predict_json(m, text, n, legacy, err, sizeof err) : NULL;
        if (res) { puts(res); free(res); }
        else { fprintf(stderr, "error: %s\n", text ? err : "out of memory"); rc = 1; }
        free(text);
    }
    julia_free(m);
    return rc;
}
