#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

#include "json.h"

/* ================================================================ helpers */

int idvec_push(IdVec *a, int32_t x) {
    if (a->n == a->cap) {
        size_t nc = a->cap ? a->cap * 2 : 256;
        int32_t *nv = (int32_t *)realloc(a->v, nc * sizeof(int32_t));
        if (!nv) return -1;
        a->v = nv;
        a->cap = nc;
    }
    a->v[a->n++] = x;
    return 0;
}

void idvec_free(IdVec *a) {
    free(a->v);
    a->v = NULL;
    a->n = a->cap = 0;
}

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

static int buf_reserve(Buf *b, size_t extra) {
    if (b->n + extra + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 64;
        while (nc < b->n + extra + 1) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) return -1;
        b->p = np;
        b->cap = nc;
    }
    return 0;
}
static int buf_add(Buf *b, const char *s, size_t n) {
    if (buf_reserve(b, n)) return -1;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
    return 0;
}

static void seterr(char *err, size_t errlen, const char *fmt, const char *a) {
    if (err && errlen) snprintf(err, errlen, fmt, a ? a : "");
}

/* Unicode White_Space (same set as Rust's char::is_whitespace). */
static int is_ws(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

/* Approximation of Rust's char::is_alphanumeric() || '_' used by single_word. */
static int is_word_cp(uint32_t c) {
    if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    if (is_ws(c)) return 0;
    if (c >= 0xA0 && c <= 0xBF) return c == 0xAA || c == 0xB5 || c == 0xBA || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE);
    if (c == 0xD7 || c == 0xF7) return 0;
    if (c >= 0x2000 && c <= 0x206F) return 0;
    if (c >= 0x3000 && c <= 0x303F) return 0;
    if (c >= 0xFF00 && c <= 0xFF0F) return 0;
    return 1;
}

/* ============================================================== data types */

typedef struct {
    const char *s;
    uint32_t n;
    int32_t id;
} VEnt;

typedef struct {
    uint64_t key;
    uint32_t rank, new_id;
} MEnt;

typedef struct {
    char *s;
    uint32_t n;
    int32_t id;
    uint8_t special, normalized, lstrip, rstrip, single_word;
} AddedTok;

typedef struct {
    int *idx;       /* indices into added[], grouped by first byte, longest first */
    size_t n;
    size_t start[257];
} TokSet;

enum { NO_REPLACE, NO_PREPEND, NO_STRIP };
typedef struct {
    int kind;
    char *a, *b;
    size_t an, bn;
    int left, right;
} NormOp;

enum { B_REMOVED, B_ISOLATED, B_PREV, B_NEXT, B_CONTIG };
enum { PT_SPLIT, PT_METASPACE, PT_WSSPLIT };
enum { PS_ALWAYS, PS_FIRST, PS_NEVER };
typedef struct {
    int kind;
    char *pat;
    size_t patn;
    int behavior, invert;
    char *repl;
    size_t repln;
    int scheme, split;
} PreOp;

struct Tokenizer {
    Arena *pool;
    VEnt *vtab;
    size_t vcap, vcount;
    MEnt *mtab;
    size_t mcap, nmerges;
    int32_t unk_id;
    int fuse_unk, byte_fallback, ignore_merges;
    int32_t byte_tok[256];
    AddedTok *added;
    size_t nadded;
    TokSet plain, norm;
    NormOp *nops;
    size_t nnops;
    PreOp *pops;
    size_t npops;
    char *norm_desc, *pre_desc;
};

/* ================================================================== vocab */

static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int vocab_get(const Tokenizer *t, const char *s, size_t n) {
    if (!t->vcap) return -1;
    size_t m = t->vcap - 1, i = (size_t)fnv(s, n) & m;
    while (t->vtab[i].s) {
        if (t->vtab[i].n == n && memcmp(t->vtab[i].s, s, n) == 0) return t->vtab[i].id;
        i = (i + 1) & m;
    }
    return -1;
}

static void vocab_put(Tokenizer *t, const char *s, size_t n, int id) {
    size_t m = t->vcap - 1, i = (size_t)fnv(s, n) & m;
    while (t->vtab[i].s) {
        if (t->vtab[i].n == n && memcmp(t->vtab[i].s, s, n) == 0) {
            t->vtab[i].id = id; /* later duplicate wins */
            return;
        }
        i = (i + 1) & m;
    }
    char *copy = (char *)arena_alloc(t->pool, n + 1);
    memcpy(copy, s, n);
    copy[n] = 0;
    t->vtab[i].s = copy;
    t->vtab[i].n = (uint32_t)n;
    t->vtab[i].id = id;
    t->vcount++;
}

static int merge_get(const Tokenizer *t, int32_t a, int32_t b, uint32_t *rank, uint32_t *nid) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t m = t->mcap - 1, i = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 20) & m;
    while (t->mtab[i].key != UINT64_MAX) {
        if (t->mtab[i].key == key) {
            *rank = t->mtab[i].rank;
            *nid = t->mtab[i].new_id;
            return 1;
        }
        i = (i + 1) & m;
    }
    return 0;
}

static void merge_put(Tokenizer *t, int32_t a, int32_t b, uint32_t rank, uint32_t nid) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t m = t->mcap - 1, i = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 20) & m;
    while (t->mtab[i].key != UINT64_MAX && t->mtab[i].key != key) i = (i + 1) & m;
    t->mtab[i].key = key;
    t->mtab[i].rank = rank;
    t->mtab[i].new_id = nid;
}

/* =========================================================== BPE (per word) */

typedef struct {
    int32_t c;
    int32_t prev, next;
    int alive;
} Sym;

typedef struct {
    uint32_t rank;
    int32_t pos;
    uint32_t new_id;
} HeapEnt;

static int heap_less(const HeapEnt *a, const HeapEnt *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->pos < b->pos;
}
static void heap_push(HeapEnt *h, size_t *n, HeapEnt e) {
    size_t i = (*n)++;
    h[i] = e;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!heap_less(&h[i], &h[p])) break;
        HeapEnt tmp = h[i];
        h[i] = h[p];
        h[p] = tmp;
        i = p;
    }
}
static HeapEnt heap_pop(HeapEnt *h, size_t *n) {
    HeapEnt top = h[0];
    h[0] = h[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && heap_less(&h[l], &h[m])) m = l;
        if (r < *n && heap_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        HeapEnt tmp = h[i];
        h[i] = h[m];
        h[m] = tmp;
        i = m;
    }
    return top;
}

static int bpe_word(const Tokenizer *t, const char *w, size_t wn, IdVec *out) {
    if (wn == 0) return 0;
    if (t->ignore_merges) {
        int id = vocab_get(t, w, wn);
        if (id >= 0) return idvec_push(out, id);
    }
    size_t cap = wn + 1; /* at most one symbol per byte */
    Sym *sym = (Sym *)malloc(cap * sizeof(Sym));
    if (!sym) return -1;
    size_t ns = 0;
    int have_unk = 0;
    size_t i = 0;
    while (i < wn) {
        uint32_t cp;
        int k = 1;
        if ((unsigned char)w[i] >= 0x80) {
            k = utf8_decode((const unsigned char *)w + i, wn - i, &cp);
            if (!k) { free(sym); return -1; }
        }
        int id = vocab_get(t, w + i, (size_t)k);
        if (id >= 0) {
            if (have_unk) { sym[ns++].c = t->unk_id; have_unk = 0; }
            sym[ns++].c = id;
        } else {
            int done = 0;
            if (t->byte_fallback) {
                int all = 1;
                for (int j = 0; j < k; j++) if (t->byte_tok[(unsigned char)w[i + (size_t)j]] < 0) { all = 0; break; }
                if (all) {
                    for (int j = 0; j < k; j++) sym[ns++].c = t->byte_tok[(unsigned char)w[i + (size_t)j]];
                    done = 1; /* HF `continue`s here without flushing a pending unk */
                }
            }
            if (!done && t->unk_id >= 0) {
                if (have_unk) {
                    if (!t->fuse_unk) sym[ns++].c = t->unk_id; /* previous one flushed, new one pending */
                    /* fuse_unk: just extend the pending one */
                } else {
                    have_unk = 1;
                }
            }
        }
        i += (size_t)k;
    }
    if (have_unk) sym[ns++].c = t->unk_id;
    for (size_t s = 0; s < ns; s++) {
        sym[s].prev = (int32_t)s - 1;
        sym[s].next = s + 1 < ns ? (int32_t)s + 1 : -1;
        sym[s].alive = 1;
    }
    if (ns > 1) {
        HeapEnt *heap = (HeapEnt *)malloc((3 * ns + 4) * sizeof(HeapEnt));
        if (!heap) { free(sym); return -1; }
        size_t hn = 0;
        uint32_t rank, nid;
        for (size_t s = 0; s + 1 < ns; s++)
            if (merge_get(t, sym[s].c, sym[s + 1].c, &rank, &nid)) heap_push(heap, &hn, (HeapEnt){rank, (int32_t)s, nid});
        while (hn) {
            HeapEnt top = heap_pop(heap, &hn);
            Sym *cur = &sym[top.pos];
            if (!cur->alive || cur->next == -1) continue;
            int32_t np = cur->next;
            uint32_t r2, n2;
            if (!merge_get(t, cur->c, sym[np].c, &r2, &n2) || n2 != top.new_id) continue; /* stale entry */
            cur->c = (int32_t)top.new_id;
            cur->next = sym[np].next;
            sym[np].alive = 0;
            if (sym[np].next != -1) sym[sym[np].next].prev = top.pos;
            if (cur->prev >= 0 && merge_get(t, sym[cur->prev].c, cur->c, &r2, &n2))
                heap_push(heap, &hn, (HeapEnt){r2, cur->prev, n2});
            if (cur->next != -1 && merge_get(t, cur->c, sym[cur->next].c, &r2, &n2))
                heap_push(heap, &hn, (HeapEnt){r2, top.pos, n2});
        }
        free(heap);
    }
    int rc = 0;
    for (size_t s = 0; s < ns && !rc; s++)
        if (sym[s].alive) rc = idvec_push(out, sym[s].c);
    free(sym);
    return rc;
}

/* ============================================================ added tokens */

typedef struct {
    size_t start, end;
    int added; /* index into added[] or -1 */
} Piece;

typedef struct {
    Piece *v;
    size_t n, cap;
} PieceVec;

static int piece_push(PieceVec *p, size_t s, size_t e, int added) {
    if (p->n == p->cap) {
        size_t nc = p->cap ? p->cap * 2 : 16;
        Piece *nv = (Piece *)realloc(p->v, nc * sizeof(Piece));
        if (!nv) return -1;
        p->v = nv;
        p->cap = nc;
    }
    p->v[p->n++] = (Piece){s, e, added};
    return 0;
}

static const AddedTok *g_sort_added;
static int cmp_added(const void *x, const void *y) {
    const AddedTok *a = &g_sort_added[*(const int *)x], *b = &g_sort_added[*(const int *)y];
    unsigned char fa = (unsigned char)a->s[0], fb = (unsigned char)b->s[0];
    if (fa != fb) return fa < fb ? -1 : 1;
    if (a->n != b->n) return a->n > b->n ? -1 : 1;
    return 0;
}

static int build_set(TokSet *set, const AddedTok *added, size_t n, int want_normalized) {
    set->idx = (int *)malloc((n ? n : 1) * sizeof(int));
    if (!set->idx) return -1;
    set->n = 0;
    for (size_t i = 0; i < n; i++)
        if (added[i].n > 0 && (added[i].normalized ? 1 : 0) == want_normalized) set->idx[set->n++] = (int)i;
    g_sort_added = added;
    qsort(set->idx, set->n, sizeof(int), cmp_added);
    memset(set->start, 0, sizeof set->start);
    for (size_t k = 0; k < set->n; k++) set->start[(unsigned char)added[set->idx[k]].s[0] + 1]++;
    for (int b = 0; b < 256; b++) set->start[b + 1] += set->start[b];
    return 0;
}

/* Splits `s` on added tokens exactly like AddedVocabulary::find_matches. */
static int split_added(const Tokenizer *t, const TokSet *set, const char *s, size_t n, PieceVec *out) {
    if (n == 0) return piece_push(out, 0, 0, -1);
    size_t start_offset = 0, i = 0;
    while (i < n && set->n) {
        unsigned char b = (unsigned char)s[i];
        size_t lo = set->start[b], hi = set->start[b + 1];
        int found = -1;
        for (size_t k = lo; k < hi; k++) {
            const AddedTok *a = &t->added[set->idx[k]];
            if (a->n <= n - i && memcmp(a->s, s + i, a->n) == 0) { found = set->idx[k]; break; }
        }
        if (found < 0) { i++; continue; }
        const AddedTok *a = &t->added[found];
        size_t start = i, stop = i + a->n;
        i = stop; /* a rejected match still consumes its text */
        if (a->single_word) {
            int start_ok = 1, stop_ok = 1;
            if (start > 0) {
                size_t j = start;
                while (j > 0 && ((unsigned char)s[j - 1] & 0xC0) == 0x80) j--;
                if (j > 0) {
                    uint32_t cp;
                    if (utf8_decode((const unsigned char *)s + j - 1, start - j + 1, &cp) == 0) {
                        j--;
                    }
                }
                uint32_t cp = 0;
                size_t q = start - 1;
                while (q > 0 && ((unsigned char)s[q] & 0xC0) == 0x80) q--;
                utf8_decode((const unsigned char *)s + q, start - q, &cp);
                start_ok = !is_word_cp(cp);
            }
            if (stop < n) {
                uint32_t cp = 0;
                utf8_decode((const unsigned char *)s + stop, n - stop, &cp);
                stop_ok = !is_word_cp(cp);
            }
            if (!start_ok || !stop_ok) continue;
        }
        if (a->lstrip) {
            size_t ns = start;
            while (ns > start_offset) {
                size_t q = ns - 1;
                while (q > start_offset && ((unsigned char)s[q] & 0xC0) == 0x80) q--;
                uint32_t cp = 0;
                utf8_decode((const unsigned char *)s + q, ns - q, &cp);
                if (!is_ws(cp)) break;
                ns = q;
            }
            start = ns;
        }
        if (a->rstrip) {
            while (stop < n) {
                uint32_t cp = 0;
                int k = utf8_decode((const unsigned char *)s + stop, n - stop, &cp);
                if (!k || !is_ws(cp)) break;
                stop += (size_t)k;
            }
            i = stop;
        }
        if (start_offset < start && piece_push(out, start_offset, start, -1)) return -1;
        if (piece_push(out, start, stop, found)) return -1;
        start_offset = stop;
    }
    if (start_offset != n && piece_push(out, start_offset, n, -1)) return -1;
    return 0;
}

/* ============================================================== normalizer */

static int normalize(const Tokenizer *t, const char *s, size_t n, Buf *out) {
    Buf cur = {0}, nxt = {0};
    if (buf_add(&cur, s, n)) return -1;
    for (size_t k = 0; k < t->nnops; k++) {
        const NormOp *op = &t->nops[k];
        nxt.n = 0;
        if (buf_reserve(&nxt, 0)) goto oom;
        nxt.p[0] = 0;
        if (op->kind == NO_REPLACE) {
            size_t i = 0;
            while (i < cur.n) {
                if (op->an && cur.n - i >= op->an && memcmp(cur.p + i, op->a, op->an) == 0) {
                    if (buf_add(&nxt, op->b, op->bn)) goto oom;
                    i += op->an;
                } else {
                    if (buf_add(&nxt, cur.p + i, 1)) goto oom;
                    i++;
                }
            }
        } else if (op->kind == NO_PREPEND) {
            if (cur.n && buf_add(&nxt, op->a, op->an)) goto oom;
            if (buf_add(&nxt, cur.p, cur.n)) goto oom;
        } else { /* NO_STRIP */
            size_t b = 0, e = cur.n;
            if (op->left) {
                while (b < e) {
                    uint32_t cp = 0;
                    int kk = utf8_decode((const unsigned char *)cur.p + b, e - b, &cp);
                    if (!kk || !is_ws(cp)) break;
                    b += (size_t)kk;
                }
            }
            if (op->right) {
                while (e > b) {
                    size_t q = e - 1;
                    while (q > b && ((unsigned char)cur.p[q] & 0xC0) == 0x80) q--;
                    uint32_t cp = 0;
                    utf8_decode((const unsigned char *)cur.p + q, e - q, &cp);
                    if (!is_ws(cp)) break;
                    e = q;
                }
            }
            if (buf_add(&nxt, cur.p + b, e - b)) goto oom;
        }
        Buf tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    free(nxt.p);
    *out = cur;
    return 0;
oom:
    free(cur.p);
    free(nxt.p);
    return -1;
}

/* ============================================================ pre-tokenizer */

typedef struct {
    const char *p;
    size_t n;
} Seg;

typedef struct {
    Seg *v;
    size_t n, cap;
} SegVec;

static int seg_push(SegVec *a, const char *p, size_t n) {
    if (a->n == a->cap) {
        size_t nc = a->cap ? a->cap * 2 : 64;
        Seg *nv = (Seg *)realloc(a->v, nc * sizeof(Seg));
        if (!nv) return -1;
        a->v = nv;
        a->cap = nc;
    }
    a->v[a->n].p = p;
    a->v[a->n].n = n;
    a->n++;
    return 0;
}

typedef struct {
    size_t s, e;
    int match;
} Span;

/* Applies a SplitDelimiterBehavior to alternating match / non-match spans. */
static int apply_behavior(const char *base, Span *sp, size_t ns, int behavior, SegVec *out) {
    if (behavior == B_REMOVED) {
        for (size_t i = 0; i < ns; i++)
            if (!sp[i].match && seg_push(out, base + sp[i].s, sp[i].e - sp[i].s)) return -1;
        return 0;
    }
    if (behavior == B_ISOLATED) {
        for (size_t i = 0; i < ns; i++)
            if (seg_push(out, base + sp[i].s, sp[i].e - sp[i].s)) return -1;
        return 0;
    }
    Span *acc = (Span *)malloc((ns ? ns : 1) * sizeof(Span));
    if (!acc) return -1;
    size_t na = 0;
    if (behavior == B_PREV) {
        int prev_match = 0;
        for (size_t i = 0; i < ns; i++) {
            if (sp[i].match && !prev_match) {
                if (na) acc[na - 1].e = sp[i].e;
                else acc[na++] = sp[i];
            } else acc[na++] = sp[i];
            prev_match = sp[i].match;
        }
    } else if (behavior == B_NEXT) {
        int prev_match = 0;
        Span *rev = (Span *)malloc((ns ? ns : 1) * sizeof(Span));
        if (!rev) { free(acc); return -1; }
        size_t nr = 0;
        for (size_t i = ns; i > 0; i--) {
            Span x = sp[i - 1];
            if (x.match && !prev_match) {
                if (nr) rev[nr - 1].s = x.s;
                else rev[nr++] = x;
            } else rev[nr++] = x;
            prev_match = x.match;
        }
        for (size_t i = 0; i < nr; i++) acc[i] = rev[nr - 1 - i];
        na = nr;
        free(rev);
    } else { /* B_CONTIG */
        int prev_match = 0;
        for (size_t i = 0; i < ns; i++) {
            if (sp[i].match == prev_match) {
                if (na) acc[na - 1].e = sp[i].e;
                else acc[na++] = sp[i];
            } else acc[na++] = sp[i];
            prev_match = sp[i].match;
        }
    }
    int rc = 0;
    for (size_t i = 0; i < na && !rc; i++) rc = seg_push(out, base + acc[i].s, acc[i].e - acc[i].s);
    free(acc);
    return rc;
}

static int split_string(const char *p, size_t n, const char *pat, size_t patn, int behavior, int invert, SegVec *out) {
    if (n == 0) return 0;
    Span *sp = (Span *)malloc((2 * (n / (patn ? patn : 1)) + 3) * sizeof(Span));
    if (!sp) return -1;
    size_t ns = 0, last = 0, i = 0;
    if (patn == 0) {
        sp[ns++] = (Span){0, n, invert ? 1 : 0};
    } else {
        while (i + patn <= n) {
            if (memcmp(p + i, pat, patn) == 0) {
                if (last < i) sp[ns++] = (Span){last, i, 0};
                sp[ns++] = (Span){i, i + patn, 1};
                i += patn;
                last = i;
            } else i++;
        }
        if (last < n) sp[ns++] = (Span){last, n, 0};
        if (invert) for (size_t k = 0; k < ns; k++) sp[k].match = !sp[k].match;
    }
    int rc = apply_behavior(p, sp, ns, behavior, out);
    free(sp);
    return rc;
}

static int split_ws(const char *p, size_t n, SegVec *out) {
    size_t i = 0, start = 0;
    int in_word = 0;
    while (i < n) {
        uint32_t cp = 0;
        int k = utf8_decode((const unsigned char *)p + i, n - i, &cp);
        if (!k) return -1;
        if (is_ws(cp)) {
            if (in_word && seg_push(out, p + start, i - start)) return -1;
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            start = i;
        }
        i += (size_t)k;
    }
    if (in_word && seg_push(out, p + start, n - start)) return -1;
    return 0;
}

/* ================================================================== encode */

int tok_encode(const Tokenizer *t, const char *text, size_t len, IdVec *out, char *err, size_t errlen) {
    if (!utf8_valid(text, len)) { seterr(err, errlen, "%s", "input text is not valid UTF-8"); return -1; }
    PieceVec p1 = {0};
    int rc = -1;
    if (split_added(t, &t->plain, text, len, &p1)) goto oom;
    for (size_t a = 0; a < p1.n; a++) {
        if (p1.v[a].added >= 0) { if (idvec_push(out, t->added[p1.v[a].added].id)) goto oom; continue; }
        Buf nb = {0};
        if (normalize(t, text + p1.v[a].start, p1.v[a].end - p1.v[a].start, &nb)) goto oom;
        if (!nb.p) { if (buf_reserve(&nb, 0)) goto oom; nb.p[0] = 0; }
        PieceVec p2 = {0};
        if (t->norm.n) {
            if (split_added(t, &t->norm, nb.p, nb.n, &p2)) { free(nb.p); goto oom; }
        } else if (piece_push(&p2, 0, nb.n, -1)) { free(nb.p); goto oom; }
        for (size_t b = 0; b < p2.n; b++) {
            if (p2.v[b].added >= 0) {
                if (idvec_push(out, t->added[p2.v[b].added].id)) { free(p2.v); free(nb.p); goto oom; }
                continue;
            }
            /* pre-tokenize */
            SegVec cur = {0}, nxt = {0};
            void **garbage = NULL;
            size_t ng = 0, gcap = 0;
            int ok = seg_push(&cur, nb.p + p2.v[b].start, p2.v[b].end - p2.v[b].start) == 0;
            int first_split = 1;
            for (size_t k = 0; ok && k < t->npops; k++) {
                const PreOp *op = &t->pops[k];
                nxt.n = 0;
                for (size_t sidx = 0; ok && sidx < cur.n; sidx++) {
                    Seg sg = cur.v[sidx];
                    if (op->kind == PT_SPLIT) {
                        ok = split_string(sg.p, sg.n, op->pat, op->patn, op->behavior, op->invert, &nxt) == 0;
                    } else if (op->kind == PT_WSSPLIT) {
                        ok = split_ws(sg.p, sg.n, &nxt) == 0;
                    } else { /* metaspace */
                        Buf mb = {0};
                        for (size_t q = 0; q < sg.n; q++) {
                            if (sg.p[q] == ' ') ok = ok && buf_add(&mb, op->repl, op->repln) == 0;
                            else ok = ok && buf_add(&mb, sg.p + q, 1) == 0;
                        }
                        if (ok && op->scheme != PS_NEVER) {
                            int starts = mb.n >= op->repln && memcmp(mb.p, op->repl, op->repln) == 0;
                            if (!starts) {
                                if (op->scheme == PS_ALWAYS || (op->scheme == PS_FIRST && first_split)) {
                                    Buf pb = {0};
                                    ok = buf_add(&pb, op->repl, op->repln) == 0 && buf_add(&pb, mb.p ? mb.p : "", mb.n) == 0;
                                    free(mb.p);
                                    mb = pb;
                                    first_split = 0;
                                }
                            } else first_split = 0;
                        }
                        if (ok) {
                            if (ng == gcap) {
                                gcap = gcap ? gcap * 2 : 8;
                                garbage = (void **)realloc(garbage, gcap * sizeof(void *));
                            }
                            garbage[ng++] = mb.p;
                            if (op->split) ok = split_string(mb.p ? mb.p : "", mb.n, op->repl, op->repln, B_NEXT, 0, &nxt) == 0;
                            else ok = seg_push(&nxt, mb.p ? mb.p : "", mb.n) == 0;
                        } else free(mb.p);
                    }
                }
                SegVec tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            for (size_t sidx = 0; ok && sidx < cur.n; sidx++) ok = bpe_word(t, cur.v[sidx].p, cur.v[sidx].n, out) == 0;
            for (size_t g = 0; g < ng; g++) free(garbage[g]);
            free(garbage);
            free(cur.v);
            free(nxt.v);
            if (!ok) { free(p2.v); free(nb.p); goto oom; }
        }
        free(p2.v);
        free(nb.p);
    }
    rc = 0;
    free(p1.v);
    return rc;
oom:
    free(p1.v);
    seterr(err, errlen, "%s", "tokenization failed (out of memory or invalid text)");
    return -1;
}

/* =================================================================== load */

static char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = 0;
    *n = got;
    return buf;
}

static char *dupn(const char *s, size_t n) {
    char *r = (char *)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}

static int add_normop(Tokenizer *t, NormOp op) {
    NormOp *nn = (NormOp *)realloc(t->nops, (t->nnops + 1) * sizeof(NormOp));
    if (!nn) return -1;
    t->nops = nn;
    t->nops[t->nnops++] = op;
    return 0;
}
static int add_preop(Tokenizer *t, PreOp op) {
    PreOp *nn = (PreOp *)realloc(t->pops, (t->npops + 1) * sizeof(PreOp));
    if (!nn) return -1;
    t->pops = nn;
    t->pops[t->npops++] = op;
    return 0;
}

static int parse_normalizer(Tokenizer *t, const JNode *n, char *err, size_t errlen) {
    if (!n || n->type == J_NULL) return 0;
    const JNode *ty = jobj_get(n, "type");
    if (!jis(ty, J_STR)) { seterr(err, errlen, "%s", "normalizer without type"); return -1; }
    if (jstr_eq(ty, "Sequence")) {
        const JNode *l = jobj_get(n, "normalizers");
        if (!jis(l, J_ARR)) { seterr(err, errlen, "%s", "bad Sequence normalizer"); return -1; }
        for (uint32_t i = 0; i < l->n; i++)
            if (parse_normalizer(t, &l->a.kids[i], err, errlen)) return -1;
        return 0;
    }
    if (jstr_eq(ty, "Replace")) {
        const JNode *pat = jobj_get(n, "pattern"), *cont = jobj_get(n, "content");
        const JNode *ps = jobj_get(pat, "String");
        if (!jis(ps, J_STR) || !jis(cont, J_STR)) {
            seterr(err, errlen, "%s", "unsupported normalizer: Replace with a Regex pattern");
            return -1;
        }
        NormOp op = {NO_REPLACE, dupn(ps->a.s, ps->n), dupn(cont->a.s, cont->n), ps->n, cont->n, 0, 0};
        return add_normop(t, op);
    }
    if (jstr_eq(ty, "Prepend")) {
        const JNode *pp = jobj_get(n, "prepend");
        if (!jis(pp, J_STR)) { seterr(err, errlen, "%s", "bad Prepend normalizer"); return -1; }
        NormOp op = {NO_PREPEND, dupn(pp->a.s, pp->n), NULL, pp->n, 0, 0, 0};
        return add_normop(t, op);
    }
    if (jstr_eq(ty, "Strip")) {
        const JNode *l = jobj_get(n, "strip_left"), *r = jobj_get(n, "strip_right");
        NormOp op = {NO_STRIP, NULL, NULL, 0, 0, l && l->type == J_TRUE, r && r->type == J_TRUE};
        return add_normop(t, op);
    }
    char name[96];
    snprintf(name, sizeof name, "%.*s", (int)(ty->n < 80 ? ty->n : 80), ty->a.s);
    seterr(err, errlen, "unsupported normalizer type '%s' (supported: Replace(String), Prepend, Strip, Sequence)", name);
    return -1;
}

static int parse_behavior(const JNode *b) {
    if (jstr_eq(b, "Removed")) return B_REMOVED;
    if (jstr_eq(b, "Isolated")) return B_ISOLATED;
    if (jstr_eq(b, "MergedWithPrevious")) return B_PREV;
    if (jstr_eq(b, "MergedWithNext")) return B_NEXT;
    if (jstr_eq(b, "Contiguous")) return B_CONTIG;
    return -1;
}

static int parse_pretok(Tokenizer *t, const JNode *n, char *err, size_t errlen) {
    if (!n || n->type == J_NULL) return 0;
    const JNode *ty = jobj_get(n, "type");
    if (!jis(ty, J_STR)) { seterr(err, errlen, "%s", "pre_tokenizer without type"); return -1; }
    if (jstr_eq(ty, "Sequence")) {
        const JNode *l = jobj_get(n, "pretokenizers");
        if (!jis(l, J_ARR)) { seterr(err, errlen, "%s", "bad Sequence pre_tokenizer"); return -1; }
        for (uint32_t i = 0; i < l->n; i++)
            if (parse_pretok(t, &l->a.kids[i], err, errlen)) return -1;
        return 0;
    }
    if (jstr_eq(ty, "Split")) {
        const JNode *pat = jobj_get(n, "pattern"), *beh = jobj_get(n, "behavior"), *inv = jobj_get(n, "invert");
        const JNode *ps = jobj_get(pat, "String");
        int b = parse_behavior(beh);
        if (!jis(ps, J_STR)) {
            seterr(err, errlen, "%s", "unsupported pre_tokenizer: Split with a Regex pattern (no regex engine in this build)");
            return -1;
        }
        if (b < 0) { seterr(err, errlen, "%s", "unknown Split behavior"); return -1; }
        PreOp op;
        memset(&op, 0, sizeof op);
        op.kind = PT_SPLIT;
        op.pat = dupn(ps->a.s, ps->n);
        op.patn = ps->n;
        op.behavior = b;
        op.invert = inv && inv->type == J_TRUE;
        return add_preop(t, op);
    }
    if (jstr_eq(ty, "WhitespaceSplit")) {
        PreOp op;
        memset(&op, 0, sizeof op);
        op.kind = PT_WSSPLIT;
        return add_preop(t, op);
    }
    if (jstr_eq(ty, "Metaspace")) {
        const JNode *rep = jobj_get(n, "replacement"), *ps = jobj_get(n, "prepend_scheme"), *sp = jobj_get(n, "split");
        const JNode *apx = jobj_get(n, "add_prefix_space");
        PreOp op;
        memset(&op, 0, sizeof op);
        op.kind = PT_METASPACE;
        if (jis(rep, J_STR)) { op.repl = dupn(rep->a.s, rep->n); op.repln = rep->n; }
        else { op.repl = dupn("\xE2\x96\x81", 3); op.repln = 3; }
        op.scheme = PS_ALWAYS;
        if (jis(ps, J_STR)) {
            if (jstr_eq(ps, "first")) op.scheme = PS_FIRST;
            else if (jstr_eq(ps, "never")) op.scheme = PS_NEVER;
        } else if (apx && apx->type == J_FALSE) op.scheme = PS_NEVER;
        op.split = !(sp && sp->type == J_FALSE);
        return add_preop(t, op);
    }
    char name[96];
    snprintf(name, sizeof name, "%.*s", (int)(ty->n < 80 ? ty->n : 80), ty->a.s);
    seterr(err, errlen, "unsupported pre_tokenizer type '%s' (supported: Split(String), Metaspace, WhitespaceSplit, Sequence)", name);
    return -1;
}

static int getbool(const JNode *o, const char *k, int dflt) {
    const JNode *v = jobj_get(o, k);
    if (!v) return dflt;
    return v->type == J_TRUE;
}

Tokenizer *tok_load(const char *path, char *err, size_t errlen) {
    size_t len;
    char *text = read_file(path, &len);
    if (!text) { seterr(err, errlen, "cannot read %s", path); return NULL; }
    if (len > 0 && len < 400 && strstr(text, "git-lfs.github.com")) {
        seterr(err, errlen, "%s is a Git LFS pointer, not the real tokenizer.json (download the actual file)", path);
        free(text);
        return NULL;
    }
    Arena *tmp = arena_new();
    JNode *root = json_parse(tmp, text, len, err, errlen);
    Tokenizer *t = NULL;
    if (!root) goto fail;
    const JNode *model = jobj_get(root, "model");
    if (!jis(model, J_OBJ)) { seterr(err, errlen, "%s", "tokenizer.json has no model"); goto fail; }
    const JNode *mt = jobj_get(model, "type");
    if (mt && mt->type == J_STR && !jstr_eq(mt, "BPE")) { seterr(err, errlen, "%s", "only BPE tokenizer models are supported"); goto fail; }
    const JNode *vocab = jobj_get(model, "vocab"), *merges = jobj_get(model, "merges");
    if (!jis(vocab, J_OBJ) || !jis(merges, J_ARR)) { seterr(err, errlen, "%s", "BPE model needs vocab object and merges array"); goto fail; }
    const JNode *dropout = jobj_get(model, "dropout");
    if (dropout && dropout->type == J_NUM && dropout->b.num != 0.0) { seterr(err, errlen, "%s", "BPE dropout is not supported"); goto fail; }
    const JNode *csp = jobj_get(model, "continuing_subword_prefix"), *eow = jobj_get(model, "end_of_word_suffix");
    if ((jis(csp, J_STR) && csp->n) || (jis(eow, J_STR) && eow->n)) {
        seterr(err, errlen, "%s", "continuing_subword_prefix / end_of_word_suffix are not supported");
        goto fail;
    }

    t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->pool = arena_new();
    t->unk_id = -1;
    for (int i = 0; i < 256; i++) t->byte_tok[i] = -1;
    t->fuse_unk = getbool(model, "fuse_unk", 0);
    t->byte_fallback = getbool(model, "byte_fallback", 0);
    t->ignore_merges = getbool(model, "ignore_merges", 0);

    size_t cap = 16;
    while (cap < (size_t)vocab->n * 2 + 2) cap *= 2;
    t->vcap = cap;
    t->vtab = (VEnt *)calloc(cap, sizeof(VEnt));
    for (uint32_t i = 0; i < vocab->n; i++) {
        const JNode *v = &vocab->a.kids[i];
        if (v->type != J_NUM) { seterr(err, errlen, "%s", "vocab values must be integers"); goto fail; }
        vocab_put(t, vocab->b.keys[i].s, vocab->b.keys[i].n, (int)v->b.num);
    }
    for (int b = 0; b < 256; b++) {
        char code[8];
        int k = snprintf(code, sizeof code, "<0x%02X>", b);
        t->byte_tok[b] = vocab_get(t, code, (size_t)k);
    }
    const JNode *unk = jobj_get(model, "unk_token");
    if (jis(unk, J_STR)) {
        t->unk_id = vocab_get(t, unk->a.s, unk->n);
        if (t->unk_id < 0) { seterr(err, errlen, "%s", "unk_token is not in the vocab"); goto fail; }
    }

    t->nmerges = merges->n;
    t->mcap = 16;
    while (t->mcap < (size_t)merges->n * 2 + 2) t->mcap *= 2;
    t->mtab = (MEnt *)malloc(t->mcap * sizeof(MEnt));
    for (size_t i = 0; i < t->mcap; i++) t->mtab[i].key = UINT64_MAX;
    Buf cat = {0};
    for (uint32_t i = 0; i < merges->n; i++) {
        const JNode *m = &merges->a.kids[i];
        const char *as, *bs;
        size_t an, bn;
        if (m->type == J_STR) {
            const char *sp = memchr(m->a.s, ' ', m->n);
            if (!sp || memchr(sp + 1, ' ', m->n - (size_t)(sp + 1 - m->a.s))) { seterr(err, errlen, "%s", "malformed merge entry"); free(cat.p); goto fail; }
            as = m->a.s; an = (size_t)(sp - m->a.s);
            bs = sp + 1; bn = m->n - an - 1;
        } else if (m->type == J_ARR && m->n == 2 && m->a.kids[0].type == J_STR && m->a.kids[1].type == J_STR) {
            as = m->a.kids[0].a.s; an = m->a.kids[0].n;
            bs = m->a.kids[1].a.s; bn = m->a.kids[1].n;
        } else { seterr(err, errlen, "%s", "malformed merge entry"); free(cat.p); goto fail; }
        int ia = vocab_get(t, as, an), ib = vocab_get(t, bs, bn);
        cat.n = 0;
        buf_add(&cat, as, an);
        buf_add(&cat, bs, bn);
        int ic = vocab_get(t, cat.p, cat.n);
        if (ia < 0 || ib < 0 || ic < 0) { seterr(err, errlen, "merge #%s references a token missing from the vocab", ""); free(cat.p); goto fail; }
        merge_put(t, ia, ib, i, (uint32_t)ic);
    }
    free(cat.p);

    const JNode *added = jobj_get(root, "added_tokens");
    if (jis(added, J_ARR)) {
        t->added = (AddedTok *)calloc(added->n ? added->n : 1, sizeof(AddedTok));
        for (uint32_t i = 0; i < added->n; i++) {
            const JNode *a = &added->a.kids[i];
            const JNode *c = jobj_get(a, "content"), *id = jobj_get(a, "id");
            if (!jis(c, J_STR) || !jis(id, J_NUM)) { seterr(err, errlen, "%s", "malformed added_tokens entry"); goto fail; }
            AddedTok *d = &t->added[t->nadded++];
            d->s = dupn(c->a.s, c->n);
            d->n = c->n;
            d->id = (int32_t)id->b.num;
            d->special = (uint8_t)getbool(a, "special", 0);
            d->normalized = (uint8_t)getbool(a, "normalized", !d->special);
            d->lstrip = (uint8_t)getbool(a, "lstrip", 0);
            d->rstrip = (uint8_t)getbool(a, "rstrip", 0);
            d->single_word = (uint8_t)getbool(a, "single_word", 0);
        }
    }
    if (build_set(&t->plain, t->added, t->nadded, 0) || build_set(&t->norm, t->added, t->nadded, 1)) { seterr(err, errlen, "%s", "out of memory"); goto fail; }
    if (parse_normalizer(t, jobj_get(root, "normalizer"), err, errlen)) goto fail;
    if (parse_pretok(t, jobj_get(root, "pre_tokenizer"), err, errlen)) goto fail;

    /* human-readable pipeline summary */
    {
        Buf d = {0};
        for (size_t i = 0; i < t->nnops; i++) {
            const NormOp *o = &t->nops[i];
            if (o->kind == NO_REPLACE) { buf_add(&d, "Replace(\"", 9); buf_add(&d, o->a, o->an); buf_add(&d, "\"->\"", 4); buf_add(&d, o->b, o->bn); buf_add(&d, "\") ", 3); }
            else if (o->kind == NO_PREPEND) { buf_add(&d, "Prepend(\"", 9); buf_add(&d, o->a, o->an); buf_add(&d, "\") ", 3); }
            else buf_add(&d, "Strip ", 6);
        }
        if (!d.n) buf_add(&d, "none", 4);
        t->norm_desc = d.p;
        Buf e = {0};
        static const char *bn[] = {"Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous"};
        for (size_t i = 0; i < t->npops; i++) {
            const PreOp *o = &t->pops[i];
            if (o->kind == PT_SPLIT) { buf_add(&e, "Split(\"", 7); buf_add(&e, o->pat, o->patn); buf_add(&e, "\",", 2); buf_add(&e, bn[o->behavior], strlen(bn[o->behavior])); buf_add(&e, ") ", 2); }
            else if (o->kind == PT_WSSPLIT) buf_add(&e, "WhitespaceSplit ", 16);
            else buf_add(&e, "Metaspace ", 10);
        }
        if (!e.n) buf_add(&e, "none", 4);
        t->pre_desc = e.p;
    }
    arena_free(tmp);
    free(text);
    return t;
fail:
    if (t) tok_free(t);
    arena_free(tmp);
    free(text);
    return NULL;
}

void tok_free(Tokenizer *t) {
    if (!t) return;
    arena_free(t->pool);
    free(t->vtab);
    free(t->mtab);
    for (size_t i = 0; i < t->nadded; i++) free(t->added[i].s);
    free(t->added);
    free(t->plain.idx);
    free(t->norm.idx);
    for (size_t i = 0; i < t->nnops; i++) { free(t->nops[i].a); free(t->nops[i].b); }
    free(t->nops);
    for (size_t i = 0; i < t->npops; i++) { free(t->pops[i].pat); free(t->pops[i].repl); }
    free(t->pops);
    free(t->norm_desc);
    free(t->pre_desc);
    free(t);
}

int tok_token_to_id(const Tokenizer *t, const char *s, size_t n) {
    for (size_t i = 0; i < t->nadded; i++)
        if (t->added[i].n == n && memcmp(t->added[i].s, s, n) == 0) return t->added[i].id;
    return vocab_get(t, s, n);
}

void tok_describe(const Tokenizer *t, FILE *f) {
    fprintf(f, "tokenizer: BPE vocab=%zu merges=%zu added=%zu byte_fallback=%d fuse_unk=%d ignore_merges=%d unk_id=%d\n",
            t->vcount, t->nmerges, t->nadded, t->byte_fallback, t->fuse_unk, t->ignore_merges, t->unk_id);
    fprintf(f, "  normalizer    : %s\n  pre_tokenizer : %s\n", t->norm_desc, t->pre_desc);
}
