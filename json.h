/* Minimal JSON DOM parser (RFC 8259) used for configs, safetensors headers,
 * tokenizer.json and request files.  Strings without escapes point into the
 * source buffer (keep it alive); strings with escapes are decoded into the arena. */
#ifndef JULIA_JSON_H
#define JULIA_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef struct Arena Arena;
Arena *arena_new(void);
void *arena_alloc(Arena *a, size_t n);
void arena_free(Arena *a);

typedef enum { J_NULL = 0, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JKey {
    const char *s;
    uint32_t n;
} JKey;

typedef struct JNode {
    uint8_t type;
    uint32_t n; /* string / number lexeme length, or array / object element count */
    union {
        const char *s;       /* J_STR, J_NUM (lexeme) */
        struct JNode *kids;  /* J_ARR, J_OBJ */
    } a;
    union {
        double num;  /* J_NUM */
        JKey *keys;  /* J_OBJ (parallel to kids) */
    } b;
} JNode;

/* Returns NULL and fills err on failure.  Nodes live in `arena`. */
JNode *json_parse(Arena *arena, const char *text, size_t len, char *err, size_t errlen);

/* Object lookup (last duplicate wins, like Python's json). NULL when absent. */
const JNode *jobj_get(const JNode *obj, const char *key);
int jstr_eq(const JNode *n, const char *lit);
static inline int jis(const JNode *n, JType t) { return n && n->type == t; }

/* UTF-8 helpers */
int utf8_decode(const unsigned char *s, size_t n, uint32_t *cp); /* bytes used, 0 if invalid */
int utf8_encode(uint32_t cp, char *out);                         /* bytes written */
int utf8_valid(const char *s, size_t n);

#endif
