/* Julia-1 decision engine in pure C.
 *
 * Reimplements the inference path of https://huggingface.co/SupersonicLabs/Julia-1 :
 *   state + question + 2-20 options  ->  mmBERT/ModernBERT encoder (22 layers)
 *   -> + type embedding -> 2 pre-norm transformer layers -> per-option scorer
 *   -> softmax over options.
 * Thread safety: a JuliaModel owns scratch buffers; use one model per thread. */
#ifndef JULIA_H
#define JULIA_H

#include <stddef.h>
#include <stdint.h>

typedef struct JuliaModel JuliaModel;

typedef struct {
    int max_length;  /* combined state/question/options token limit (default 8192) */
    int head_length; /* question+options token budget (default 512, as in inference-policy.json) */
    int strict;      /* reject overflow / reserved markers instead of truncating (default 1) */
    int threads;     /* 0 = leave OpenMP default */
    int debug;       /* include token ids / markers / raw logits in JSON output */
} JuliaOptions;

JuliaOptions julia_default_options(void);

/* `dir` is a checkpoint directory laid out like the Hugging Face repo:
 *   julia_config.json, model.safetensors, encoder/config.json,
 *   tokenizer/tokenizer.json, tokenizer/tokenizer_config.json */
JuliaModel *julia_load(const char *dir, const JuliaOptions *opt, char *err, size_t errlen);
void julia_free(JuliaModel *m);

/* Prints a short description of the loaded model/tokenizer. */
void julia_describe(const JuliaModel *m, void *file /* FILE* */);
void julia_list_tensors(const JuliaModel *m, void *file /* FILE* */);

/* Tokenizer debugging: ids for `text` exactly as tokenizer(text, add_special_tokens=False). */
int julia_tokenize(const JuliaModel *m, const char *text, size_t len, int32_t **ids, size_t *n, char *err, size_t errlen);

/* Handles one request.
 *   typed  (legacy == 0): {"state": ..., "questions": {id: {type, instructions, criteria}}}
 *   legacy (legacy == 1): [{"state":..., "question":..., "options":[...], "type":...}, ...]
 * Returns a malloc'd JSON string, or NULL with `err` filled in. */
char *julia_predict_json(JuliaModel *m, const char *request, size_t len, int legacy, char *err, size_t errlen);

#endif
