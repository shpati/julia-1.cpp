#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ arena */

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used, cap;
} ArenaBlock;

struct Arena {
    ArenaBlock *head;
};

Arena *arena_new(void) {
    return (Arena *)calloc(1, sizeof(Arena));
}

void *arena_alloc(Arena *a, size_t n) {
    n = (n + 15) & ~(size_t)15;
    ArenaBlock *b = a->head;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > ((size_t)1 << 20) ? n : ((size_t)1 << 20);
        ArenaBlock *nb = (ArenaBlock *)malloc(sizeof(ArenaBlock) + cap);
        if (!nb) return NULL;
        nb->next = a->head;
        nb->used = 0;
        nb->cap = cap;
        a->head = nb;
        b = nb;
    }
    void *p = (char *)(b + 1) + b->used;
    b->used += n;
    return p;
}

void arena_free(Arena *a) {
    if (!a) return;
    ArenaBlock *b = a->head;
    while (b) {
        ArenaBlock *n = b->next;
        free(b);
        b = n;
    }
    free(a);
}

/* ------------------------------------------------------------------ utf-8 */

int utf8_decode(const unsigned char *s, size_t n, uint32_t *cp) {
    if (n == 0) return 0;
    unsigned c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if (c >= 0xC2 && c <= 0xDF) {
        if (n < 2 || (s[1] & 0xC0) != 0x80) return 0;
        *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if (c >= 0xE0 && c <= 0xEF) {
        if (n < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        uint32_t v = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (v < 0x800 || (v >= 0xD800 && v <= 0xDFFF)) return 0;
        *cp = v;
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (n < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return 0;
        uint32_t v = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (v < 0x10000 || v > 0x10FFFF) return 0;
        *cp = v;
        return 4;
    }
    return 0;
}

int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

int utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        if ((unsigned char)s[i] < 0x80) { i++; continue; }
        uint32_t cp;
        int k = utf8_decode((const unsigned char *)s + i, n - i, &cp);
        if (!k) return 0;
        i += (size_t)k;
    }
    return 1;
}

/* ----------------------------------------------------------------- parser */

typedef struct {
    const char *p, *end, *begin;
    Arena *a;
    char *err;
    size_t errlen;
    int depth;
    JNode *stack;
    size_t scnt, scap;
    JKey *kstack;
    size_t kcnt, kcap;
    int failed;
} P;

static int fail(P *p, const char *msg) {
    if (!p->failed && p->err && p->errlen) {
        snprintf(p->err, p->errlen, "JSON error at byte %zu: %s", (size_t)(p->p - p->begin), msg);
    }
    p->failed = 1;
    return 0;
}

static void skip_ws(P *p) {
    while (p->p < p->end && (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r')) p->p++;
}

static int push_node(P *p, const JNode *n) {
    if (p->scnt == p->scap) {
        size_t nc = p->scap ? p->scap * 2 : 256;
        JNode *ns = (JNode *)realloc(p->stack, nc * sizeof(JNode));
        if (!ns) return fail(p, "out of memory");
        p->stack = ns;
        p->scap = nc;
    }
    p->stack[p->scnt++] = *n;
    return 1;
}

static int push_key(P *p, JKey k) {
    if (p->kcnt == p->kcap) {
        size_t nc = p->kcap ? p->kcap * 2 : 256;
        JKey *ns = (JKey *)realloc(p->kstack, nc * sizeof(JKey));
        if (!ns) return fail(p, "out of memory");
        p->kstack = ns;
        p->kcap = nc;
    }
    p->kstack[p->kcnt++] = k;
    return 1;
}

static int hex4(const char *s, uint32_t *v) {
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        r <<= 4;
        if (c >= '0' && c <= '9') r |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') r |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') r |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    *v = r;
    return 1;
}

/* Parses a string starting at the opening quote. */
static int parse_string(P *p, const char **out, uint32_t *outn) {
    p->p++; /* opening quote */
    const char *start = p->p;
    const char *q = start;
    int esc = 0;
    while (q < p->end && *q != '"') {
        if ((unsigned char)*q < 0x20) return fail(p, "control character in string");
        if (*q == '\\') {
            esc = 1;
            q++;
            if (q >= p->end) return fail(p, "unterminated string");
        }
        q++;
    }
    if (q >= p->end) return fail(p, "unterminated string");
    if (!esc) {
        *out = start;
        *outn = (uint32_t)(q - start);
        p->p = q + 1;
        return 1;
    }
    char *buf = (char *)arena_alloc(p->a, (size_t)(q - start) + 1);
    if (!buf) return fail(p, "out of memory");
    size_t w = 0;
    const char *s = start;
    while (s < q) {
        if (*s != '\\') { buf[w++] = *s++; continue; }
        s++;
        switch (*s) {
            case '"': buf[w++] = '"'; s++; break;
            case '\\': buf[w++] = '\\'; s++; break;
            case '/': buf[w++] = '/'; s++; break;
            case 'b': buf[w++] = '\b'; s++; break;
            case 'f': buf[w++] = '\f'; s++; break;
            case 'n': buf[w++] = '\n'; s++; break;
            case 'r': buf[w++] = '\r'; s++; break;
            case 't': buf[w++] = '\t'; s++; break;
            case 'u': {
                uint32_t v;
                if (q - s < 5 || !hex4(s + 1, &v)) return fail(p, "bad \\u escape");
                s += 5;
                if (v >= 0xD800 && v <= 0xDBFF) {
                    uint32_t lo;
                    if (q - s >= 6 && s[0] == '\\' && s[1] == 'u' && hex4(s + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        s += 6;
                    } else {
                        v = 0xFFFD;
                    }
                } else if (v >= 0xDC00 && v <= 0xDFFF) {
                    v = 0xFFFD;
                }
                w += (size_t)utf8_encode(v, buf + w);
                break;
            }
            default: return fail(p, "bad escape");
        }
    }
    *out = buf;
    *outn = (uint32_t)w;
    p->p = q + 1;
    return 1;
}

static int parse_value(P *p, JNode *out);

static int parse_array(P *p, JNode *out) {
    p->p++;
    size_t base = p->scnt;
    skip_ws(p);
    if (p->p < p->end && *p->p == ']') {
        p->p++;
    } else {
        for (;;) {
            JNode tmp;
            skip_ws(p);
            if (!parse_value(p, &tmp)) return 0;
            if (!push_node(p, &tmp)) return 0;
            skip_ws(p);
            if (p->p >= p->end) return fail(p, "unterminated array");
            if (*p->p == ',') { p->p++; continue; }
            if (*p->p == ']') { p->p++; break; }
            return fail(p, "expected ',' or ']'");
        }
    }
    size_t cnt = p->scnt - base;
    out->type = J_ARR;
    out->n = (uint32_t)cnt;
    out->a.kids = cnt ? (JNode *)arena_alloc(p->a, cnt * sizeof(JNode)) : NULL;
    if (cnt && !out->a.kids) return fail(p, "out of memory");
    if (cnt) memcpy(out->a.kids, p->stack + base, cnt * sizeof(JNode));
    out->b.keys = NULL;
    p->scnt = base;
    return 1;
}

static int parse_object(P *p, JNode *out) {
    p->p++;
    size_t base = p->scnt, kbase = p->kcnt;
    skip_ws(p);
    if (p->p < p->end && *p->p == '}') {
        p->p++;
    } else {
        for (;;) {
            skip_ws(p);
            if (p->p >= p->end || *p->p != '"') return fail(p, "expected object key");
            JKey key;
            if (!parse_string(p, &key.s, &key.n)) return 0;
            skip_ws(p);
            if (p->p >= p->end || *p->p != ':') return fail(p, "expected ':'");
            p->p++;
            skip_ws(p);
            JNode tmp;
            if (!parse_value(p, &tmp)) return 0;
            if (!push_node(p, &tmp) || !push_key(p, key)) return 0;
            skip_ws(p);
            if (p->p >= p->end) return fail(p, "unterminated object");
            if (*p->p == ',') { p->p++; continue; }
            if (*p->p == '}') { p->p++; break; }
            return fail(p, "expected ',' or '}'");
        }
    }
    size_t cnt = p->scnt - base;
    out->type = J_OBJ;
    out->n = (uint32_t)cnt;
    out->a.kids = cnt ? (JNode *)arena_alloc(p->a, cnt * sizeof(JNode)) : NULL;
    out->b.keys = cnt ? (JKey *)arena_alloc(p->a, cnt * sizeof(JKey)) : NULL;
    if (cnt && (!out->a.kids || !out->b.keys)) return fail(p, "out of memory");
    if (cnt) {
        memcpy(out->a.kids, p->stack + base, cnt * sizeof(JNode));
        memcpy(out->b.keys, p->kstack + kbase, cnt * sizeof(JKey));
    }
    p->scnt = base;
    p->kcnt = kbase;
    return 1;
}

static int parse_value(P *p, JNode *out) {
    if (p->p >= p->end) return fail(p, "unexpected end of input");
    if (++p->depth > 512) return fail(p, "nesting too deep");
    int ok = 1;
    char c = *p->p;
    memset(out, 0, sizeof(*out));
    if (c == '{') ok = parse_object(p, out);
    else if (c == '[') ok = parse_array(p, out);
    else if (c == '"') {
        out->type = J_STR;
        ok = parse_string(p, &out->a.s, &out->n);
    } else if (c == 't' && p->end - p->p >= 4 && !memcmp(p->p, "true", 4)) {
        out->type = J_TRUE;
        p->p += 4;
    } else if (c == 'f' && p->end - p->p >= 5 && !memcmp(p->p, "false", 5)) {
        out->type = J_FALSE;
        p->p += 5;
    } else if (c == 'n' && p->end - p->p >= 4 && !memcmp(p->p, "null", 4)) {
        out->type = J_NULL;
        p->p += 4;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        const char *s = p->p, *q = s;
        if (*q == '-') q++;
        if (q >= p->end || !(*q >= '0' && *q <= '9')) { ok = fail(p, "bad number"); goto done; }
        if (*q == '0') q++;
        else while (q < p->end && *q >= '0' && *q <= '9') q++;
        if (q < p->end && *q == '.') {
            q++;
            if (q >= p->end || !(*q >= '0' && *q <= '9')) { ok = fail(p, "bad number"); goto done; }
            while (q < p->end && *q >= '0' && *q <= '9') q++;
        }
        if (q < p->end && (*q == 'e' || *q == 'E')) {
            q++;
            if (q < p->end && (*q == '+' || *q == '-')) q++;
            if (q >= p->end || !(*q >= '0' && *q <= '9')) { ok = fail(p, "bad number"); goto done; }
            while (q < p->end && *q >= '0' && *q <= '9') q++;
        }
        size_t len = (size_t)(q - s);
        char small[64];
        char *tmp = len < sizeof small ? small : (char *)malloc(len + 1);
        if (!tmp) { ok = fail(p, "out of memory"); goto done; }
        memcpy(tmp, s, len);
        tmp[len] = 0;
        out->type = J_NUM;
        out->a.s = s;
        out->n = (uint32_t)len;
        out->b.num = strtod(tmp, NULL);
        if (tmp != small) free(tmp);
        p->p = q;
    } else {
        ok = fail(p, "unexpected character");
    }
done:
    p->depth--;
    return ok;
}

JNode *json_parse(Arena *arena, const char *text, size_t len, char *err, size_t errlen) {
    P p;
    memset(&p, 0, sizeof p);
    p.p = text;
    p.begin = text;
    p.end = text + len;
    p.a = arena;
    p.err = err;
    p.errlen = errlen;
    /* skip UTF-8 BOM */
    if (len >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) p.p += 3;
    JNode *root = (JNode *)arena_alloc(arena, sizeof(JNode));
    int ok = root != NULL;
    if (ok) {
        skip_ws(&p);
        ok = parse_value(&p, root);
    }
    if (ok) {
        skip_ws(&p);
        if (p.p != p.end) ok = fail(&p, "trailing data");
    }
    free(p.stack);
    free(p.kstack);
    return ok ? root : NULL;
}

const JNode *jobj_get(const JNode *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    size_t kl = strlen(key);
    for (size_t i = obj->n; i > 0; i--) {
        const JKey *k = &obj->b.keys[i - 1];
        if (k->n == kl && memcmp(k->s, key, kl) == 0) return &obj->a.kids[i - 1];
    }
    return NULL;
}

int jstr_eq(const JNode *n, const char *lit) {
    size_t l = strlen(lit);
    return n && n->type == J_STR && n->n == l && memcmp(n->a.s, lit, l) == 0;
}
