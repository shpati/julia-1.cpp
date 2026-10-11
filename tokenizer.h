/* Pure-C reimplementation of the subset of HuggingFace `tokenizers` needed by
 * Julia-1 (and other SentencePiece-derived BPE tokenizers such as Gemma's):
 *
 *   added-token splitting (+ lstrip / rstrip / single_word / normalized flags)
 *   normalizers   : Replace(String), Prepend, Strip, Sequence
 *   pre-tokenizers: Split(String pattern, all five behaviours, invert),
 *                   Metaspace, WhitespaceSplit, Sequence
 *   model         : BPE (merge ranks, byte_fallback, unk/fuse_unk, ignore_merges)
 *
 * Anything else in tokenizer.json is rejected at load time with a clear error
 * instead of being silently approximated. */
#ifndef JULIA_TOKENIZER_H
#define JULIA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int32_t *v;
    size_t n, cap;
} IdVec;

int idvec_push(IdVec *a, int32_t x);
void idvec_free(IdVec *a);

typedef struct Tokenizer Tokenizer;

Tokenizer *tok_load(const char *tokenizer_json_path, char *err, size_t errlen);
void tok_free(Tokenizer *t);

/* Added-token map first, then the model vocab (same order as HF). -1 if unknown. */
int tok_token_to_id(const Tokenizer *t, const char *s, size_t n);

/* Equivalent of tokenizer(text, add_special_tokens=False)["input_ids"].
 * Appends to `out`. text must be valid UTF-8. Returns 0 on success. */
int tok_encode(const Tokenizer *t, const char *text, size_t len, IdVec *out, char *err, size_t errlen);

void tok_describe(const Tokenizer *t, FILE *f);

#endif
