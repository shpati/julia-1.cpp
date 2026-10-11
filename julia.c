#include "julia.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define JULIA_HAVE_MMAP 1
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "json.h"
#include "tokenizer.h"

#define MAX_OPTIONS 20
#define MR 4
#define NR 16

/* ================================================================= helpers */

static int errf(char *err, size_t n, const char *fmt, ...) {
    if (err && n) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, n, fmt, ap);
        va_end(ap);
    }
    return -1;
}

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

static int buf_add(Buf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (nc < b->n + n + 1) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) return -1;
        b->p = np;
        b->cap = nc;
    }
    if (n) memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
    return 0;
}
static int buf_str(Buf *b, const char *s) { return buf_add(b, s, strlen(s)); }
static int buf_printf(Buf *b, const char *fmt, ...) {
    char tmp[128];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t)k >= sizeof tmp) return -1;
    return buf_add(b, tmp, (size_t)k);
}

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

/* ============================================================== safetensors */

enum { DT_F32, DT_F16, DT_BF16, DT_OTHER };

typedef struct {
    char *name;
    int dtype, ndim;
    int64_t shape[4];
    const unsigned char *data;
    size_t nbytes;
} STensor;

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void st_read(const STensor *t, size_t off, size_t n, float *dst) {
    if (t->dtype == DT_F32) {
        memcpy(dst, t->data + off * 4, n * 4);
    } else if (t->dtype == DT_BF16) {
        for (size_t i = 0; i < n; i++) {
            uint16_t u;
            memcpy(&u, t->data + (off + i) * 2, 2);
            uint32_t bits = (uint32_t)u << 16;
            memcpy(&dst[i], &bits, 4);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            uint16_t u;
            memcpy(&u, t->data + (off + i) * 2, 2);
            dst[i] = half_to_float(u);
        }
    }
}

static int64_t st_numel(const STensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* ================================================================== model */

typedef struct {
    float *wt;   /* packed [K][Np], transposed from PyTorch's [N][K] */
    float *bias; /* [N] or NULL */
    int K, N, Np;
} Lin;

typedef struct {
    float *w, *b; /* b may be NULL (bias=False), w NULL means Identity */
} Norm;

typedef struct {
    Norm attn_norm, mlp_norm;
    Lin wqkv, wo, wi, mo;
    int global;
} EncLayer;

typedef struct {
    Norm n1, n2;
    Lin q, kv, out, l1, l2;
} HeadLayer;

struct JuliaModel {
    JuliaOptions opt;
    /* encoder config */
    int H, I, heads, hd, layers, vocab, max_pos, window;
    float eps;
    double theta_global, theta_local;
    int head_layers, head_heads, head_hd;
    int attn_bias, mlp_bias, norm_bias;
    /* tokenizer */
    Tokenizer *tok;
    int32_t id_cls, id_sep, id_mask, id_pad;
    char mask_str[64];
    /* safetensors */
    unsigned char *map;
    size_t maplen;
    int mapped;
    STensor *tensors;
    size_t ntensors;
    /* weights */
    const STensor *emb;
    Norm emb_norm, final_norm;
    EncLayer *enc;
    float *type_emb;
    HeadLayer *head;
    Norm sc_norm;
    Lin sc_lin;
    float *sc_w3;
    float sc_b3;
    /* scratch */
    size_t cap_L;
    float *h, *x, *qkv, *att, *ff, *gg;
    float *rope_cos[2], *rope_sin[2];
    float sm_x[MAX_OPTIONS * 1], *sm; /* sm: small buffers, allocated in load */
};

JuliaOptions julia_default_options(void) {
    JuliaOptions o;
    o.max_length = 8192;
    o.head_length = 512;
    o.strict = 1;
    o.threads = 0;
    o.debug = 0;
    return o;
}

static const STensor *st_find(const JuliaModel *m, const char *name) {
    for (size_t i = 0; i < m->ntensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}

static int st_open(JuliaModel *m, const char *path, char *err, size_t errlen) {
    unsigned char *base = NULL;
    size_t size = 0;
#ifdef JULIA_HAVE_MMAP
    int fd = open(path, O_RDONLY);
    if (fd < 0) return errf(err, errlen, "cannot open %s", path);
    struct stat sb;
    if (fstat(fd, &sb) != 0) { close(fd); return errf(err, errlen, "cannot stat %s", path); }
    size = (size_t)sb.st_size;
    if (size < 8) { close(fd); return errf(err, errlen, "%s is too small to be a safetensors file", path); }
    base = (unsigned char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return errf(err, errlen, "mmap failed for %s", path);
    m->mapped = 1;
#else
    char *buf = read_file(path, &size);
    if (!buf) return errf(err, errlen, "cannot read %s", path);
    base = (unsigned char *)buf;
    m->mapped = 0;
#endif
    m->map = base;
    m->maplen = size;
    if (size < 8) return errf(err, errlen, "%s is too small", path);
    if (size < 200 && memchr(base, '\n', size) && strstr((const char *)base, "git-lfs"))
        return errf(err, errlen, "%s is a Git LFS pointer, not the real weights (download the actual 550 MiB file)", path);
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | base[i];
    if (hlen == 0 || hlen > size - 8 || hlen > ((uint64_t)1 << 28)) return errf(err, errlen, "%s: invalid safetensors header length", path);
    Arena *a = arena_new();
    JNode *root = json_parse(a, (const char *)base + 8, (size_t)hlen, err, errlen);
    if (!root || root->type != J_OBJ) { arena_free(a); if (root) errf(err, errlen, "bad safetensors header"); return -1; }
    const unsigned char *data = base + 8 + hlen;
    size_t datalen = size - 8 - (size_t)hlen;
    m->tensors = (STensor *)calloc(root->n ? root->n : 1, sizeof(STensor));
    for (uint32_t i = 0; i < root->n; i++) {
        const JKey *k = &root->b.keys[i];
        if (k->n == 12 && memcmp(k->s, "__metadata__", 12) == 0) continue;
        const JNode *e = &root->a.kids[i];
        const JNode *dt = jobj_get(e, "dtype"), *sh = jobj_get(e, "shape"), *of = jobj_get(e, "data_offsets");
        if (!jis(dt, J_STR) || !jis(sh, J_ARR) || !jis(of, J_ARR) || of->n != 2 || sh->n > 4) {
            arena_free(a);
            return errf(err, errlen, "malformed tensor entry in safetensors header");
        }
        STensor *t = &m->tensors[m->ntensors++];
        t->name = (char *)malloc(k->n + 1);
        memcpy(t->name, k->s, k->n);
        t->name[k->n] = 0;
        t->dtype = jstr_eq(dt, "F32") ? DT_F32 : jstr_eq(dt, "F16") ? DT_F16 : jstr_eq(dt, "BF16") ? DT_BF16 : DT_OTHER;
        t->ndim = (int)sh->n;
        for (uint32_t d = 0; d < sh->n; d++) t->shape[d] = (int64_t)sh->a.kids[d].b.num;
        uint64_t b0 = (uint64_t)of->a.kids[0].b.num, b1 = (uint64_t)of->a.kids[1].b.num;
        if (b1 < b0 || b1 > datalen) { arena_free(a); return errf(err, errlen, "tensor %s has out-of-range data offsets (truncated file?)", t->name); }
        t->data = data + b0;
        t->nbytes = (size_t)(b1 - b0);
        if (t->dtype != DT_OTHER) {
            size_t es = t->dtype == DT_F32 ? 4 : 2;
            if ((int64_t)(t->nbytes / es) != st_numel(t) || t->nbytes % es) { arena_free(a); return errf(err, errlen, "tensor %s: byte size does not match shape", t->name); }
        }
    }
    arena_free(a);
    return 0;
}

static int need_tensor(const JuliaModel *m, const char *name, int ndim, int64_t d0, int64_t d1, const STensor **out, char *err, size_t errlen) {
    const STensor *t = st_find(m, name);
    if (!t) return errf(err, errlen, "tensor '%s' not found in model.safetensors (run with --list-tensors to see what is there)", name);
    if (t->dtype == DT_OTHER) return errf(err, errlen, "tensor '%s' has an unsupported dtype (need F32/F16/BF16)", name);
    int ok = t->ndim == ndim && t->shape[0] == d0 && (ndim < 2 || t->shape[1] == d1);
    if (!ok) return errf(err, errlen, "tensor '%s' has unexpected shape", name);
    *out = t;
    return 0;
}

static int load_vec(const JuliaModel *m, const char *name, int n, float **out, char *err, size_t errlen) {
    const STensor *t;
    if (need_tensor(m, name, 1, n, 0, &t, err, errlen)) return -1;
    *out = (float *)malloc((size_t)n * sizeof(float));
    if (!*out) return errf(err, errlen, "out of memory");
    st_read(t, 0, (size_t)n, *out);
    return 0;
}

static int load_norm(const JuliaModel *m, const char *prefix, int n, int with_bias, Norm *out, char *err, size_t errlen) {
    char nm[256];
    snprintf(nm, sizeof nm, "%s.weight", prefix);
    if (load_vec(m, nm, n, &out->w, err, errlen)) return -1;
    out->b = NULL;
    if (with_bias) {
        snprintf(nm, sizeof nm, "%s.bias", prefix);
        if (load_vec(m, nm, n, &out->b, err, errlen)) return -1;
    }
    return 0;
}

/* Packs a PyTorch Linear weight [N][K] into [K][Np]. wname/bname are full tensor names. */
static int load_lin_rows(const JuliaModel *m, const STensor *wt, size_t row0, int N, int K, const STensor *bt, Lin *L, char *err, size_t errlen) {
    L->K = K;
    L->N = N;
    L->Np = (N + NR - 1) / NR * NR;
    L->wt = (float *)calloc((size_t)K * (size_t)L->Np, sizeof(float));
    float *tmp = (float *)malloc((size_t)K * sizeof(float));
    if (!L->wt || !tmp) { free(tmp); return errf(err, errlen, "out of memory"); }
    for (int n = 0; n < N; n++) {
        st_read(wt, (row0 + (size_t)n) * (size_t)K, (size_t)K, tmp);
        for (int k = 0; k < K; k++) L->wt[(size_t)k * (size_t)L->Np + (size_t)n] = tmp[k];
    }
    free(tmp);
    L->bias = NULL;
    if (bt) {
        L->bias = (float *)malloc((size_t)N * sizeof(float));
        if (!L->bias) return errf(err, errlen, "out of memory");
        st_read(bt, row0, (size_t)N, L->bias);
    }
    (void)m;
    return 0;
}

static int load_lin(const JuliaModel *m, const char *wname, const char *bname, int N, int K, Lin *L, char *err, size_t errlen) {
    const STensor *w, *b = NULL;
    if (need_tensor(m, wname, 2, N, K, &w, err, errlen)) return -1;
    if (bname && need_tensor(m, bname, 1, N, 0, &b, err, errlen)) return -1;
    return load_lin_rows(m, w, 0, N, K, b, L, err, errlen);
}

static void free_lin(Lin *l) { free(l->wt); free(l->bias); }

/* ================================================================= kernels */

/* C[m][n] (+)= sum_k A[m][k] * W[k][n] + bias[n] */
static void gemm(const float *A, int lda, int M, const Lin *W, float *C, int ldc, int accumulate) {
    const int K = W->K, N = W->N, Np = W->Np;
    const int nb = Np / NR, mb = (M + MR - 1) / MR;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int bi = 0; bi < mb; bi++) {
        for (int bj = 0; bj < nb; bj++) {
            const int m0 = bi * MR, n0 = bj * NR;
            const int mr = (M - m0 < MR) ? (M - m0) : MR;
            float acc[MR][NR];
            for (int i = 0; i < MR; i++)
                for (int j = 0; j < NR; j++) acc[i][j] = 0.f;
            const float *wp = W->wt + n0;
            if (mr == MR) {
                const float *a0 = A + (size_t)m0 * (size_t)lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
                for (int k = 0; k < K; k++) {
                    const float *w = wp + (size_t)k * (size_t)Np;
                    const float x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
                    for (int j = 0; j < NR; j++) {
                        acc[0][j] += x0 * w[j];
                        acc[1][j] += x1 * w[j];
                        acc[2][j] += x2 * w[j];
                        acc[3][j] += x3 * w[j];
                    }
                }
            } else {
                for (int k = 0; k < K; k++) {
                    const float *w = wp + (size_t)k * (size_t)Np;
                    for (int i = 0; i < mr; i++) {
                        const float x = A[(size_t)(m0 + i) * (size_t)lda + (size_t)k];
                        for (int j = 0; j < NR; j++) acc[i][j] += x * w[j];
                    }
                }
            }
            for (int i = 0; i < mr; i++) {
                float *c = C + (size_t)(m0 + i) * (size_t)ldc;
                for (int j = 0; j < NR; j++) {
                    const int n = n0 + j;
                    if (n >= N) break;
                    float v = acc[i][j] + (W->bias ? W->bias[n] : 0.f);
                    c[n] = accumulate ? c[n] + v : v;
                }
            }
        }
    }
}

static void layernorm(const float *x, float *y, int rows, int d, const float *w, const float *b, float eps) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * (size_t)d;
        float *yr = y + (size_t)r * (size_t)d;
        double s = 0, s2 = 0;
        for (int i = 0; i < d; i++) s += xr[i];
        const double mean = s / d;
        for (int i = 0; i < d; i++) {
            double t = xr[i] - mean;
            s2 += t * t;
        }
        const float inv = (float)(1.0 / sqrt(s2 / d + (double)eps));
        const float mf = (float)mean;
        for (int i = 0; i < d; i++) {
            float v = (xr[i] - mf) * inv * w[i];
            yr[i] = b ? v + b[i] : v;
        }
    }
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static inline float dot(const float *a, const float *b, int n) {
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; j++) s[j] += a[i + j] * b[i + j];
    float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; i < n; i++) t += a[i] * b[i];
    return t;
}

/* Multi-head attention. window < 0: full; else |i-j| <= window (queries are positions 0..nq-1). */
static void attention(const float *q, size_t qs, const float *k, size_t ks, const float *v, size_t vs,
                      float *out, size_t os, int nq, int L, int heads, int hd, int window) {
    const float scale = 1.0f / sqrtf((float)hd);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float *sc = (float *)malloc((size_t)L * sizeof(float));
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < nq; i++) {
                if (!sc) continue;
                int lo = 0, hi = L - 1;
                if (window >= 0) {
                    lo = i - window < 0 ? 0 : i - window;
                    hi = i + window > L - 1 ? L - 1 : i + window;
                }
                const float *qi = q + (size_t)i * qs + (size_t)h * (size_t)hd;
                float mx = -INFINITY;
                for (int j = lo; j <= hi; j++) {
                    float s = dot(qi, k + (size_t)j * ks + (size_t)h * (size_t)hd, hd) * scale;
                    sc[j - lo] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.f;
                for (int j = lo; j <= hi; j++) {
                    float e = expf(sc[j - lo] - mx);
                    sc[j - lo] = e;
                    sum += e;
                }
                const float inv = 1.0f / sum;
                float o[128];
                for (int d = 0; d < hd; d++) o[d] = 0.f;
                for (int j = lo; j <= hi; j++) {
                    const float p = sc[j - lo] * inv;
                    const float *vj = v + (size_t)j * vs + (size_t)h * (size_t)hd;
                    for (int d = 0; d < hd; d++) o[d] += p * vj[d];
                }
                float *oi = out + (size_t)i * os + (size_t)h * (size_t)hd;
                for (int d = 0; d < hd; d++) oi[d] = o[d];
            }
        }
        free(sc);
    }
}

/* rotate_half-style RoPE on the q and k slices of a fused qkv buffer. */
static void rope_apply(float *qkv, int L, int heads, int hd, int H, const float *cs, const float *sn) {
    const int half = hd / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < L; i++) {
        const float *c = cs + (size_t)i * (size_t)half, *s = sn + (size_t)i * (size_t)half;
        for (int part = 0; part < 2; part++) {
            float *base = qkv + (size_t)i * (size_t)(3 * H) + (size_t)part * (size_t)H;
            for (int h = 0; h < heads; h++) {
                float *p = base + (size_t)h * (size_t)hd;
                for (int j = 0; j < half; j++) {
                    const float x1 = p[j], x2 = p[j + half];
                    p[j] = x1 * c[j] - x2 * s[j];
                    p[j + half] = x2 * c[j] + x1 * s[j];
                }
            }
        }
    }
}

static int ensure_rope(JuliaModel *m, int L) {
    if ((size_t)L <= m->cap_L && m->rope_cos[0]) return 0;
    return -1; /* allocated together with the scratch buffers */
}

static void fill_rope(JuliaModel *m, size_t L) {
    const int half = m->hd / 2;
    const double thetas[2] = {m->theta_global, m->theta_local};
    for (int t = 0; t < 2; t++) {
        float *inv = (float *)malloc((size_t)half * sizeof(float));
        for (int j = 0; j < half; j++) {
            /* mirrors: 1.0 / (base ** (arange(0, dim, 2, float32) / dim)) in float32 */
            float e = (float)(2 * j) / (float)m->hd;
            inv[j] = 1.0f / powf((float)thetas[t], e);
        }
        for (size_t i = 0; i < L; i++)
            for (int j = 0; j < half; j++) {
                float ang = (float)i * inv[j];
                m->rope_cos[t][i * (size_t)half + (size_t)j] = cosf(ang);
                m->rope_sin[t][i * (size_t)half + (size_t)j] = sinf(ang);
            }
        free(inv);
    }
}

static int ensure_scratch(JuliaModel *m, size_t L) {
    if (L <= m->cap_L) return 0;
    size_t cap = L < 64 ? 64 : L;
    const size_t H = (size_t)m->H, I = (size_t)m->I;
    free(m->h); free(m->x); free(m->qkv); free(m->att); free(m->ff); free(m->gg);
    free(m->rope_cos[0]); free(m->rope_cos[1]); free(m->rope_sin[0]); free(m->rope_sin[1]);
    const size_t gg_w = I > 4 * H ? I : 4 * H;
    m->h = (float *)malloc(cap * H * sizeof(float));
    m->x = (float *)malloc(cap * H * sizeof(float));
    m->qkv = (float *)malloc((3 * cap + 64) * H * sizeof(float));
    m->att = (float *)malloc(cap * H * sizeof(float));
    m->ff = (float *)malloc(cap * 2 * I * sizeof(float));
    m->gg = (float *)malloc(cap * gg_w * sizeof(float));
    const size_t rl = cap * (size_t)(m->hd / 2);
    for (int t = 0; t < 2; t++) {
        m->rope_cos[t] = (float *)malloc(rl * sizeof(float));
        m->rope_sin[t] = (float *)malloc(rl * sizeof(float));
    }
    if (!m->h || !m->x || !m->qkv || !m->att || !m->ff || !m->gg || !m->rope_cos[0] || !m->rope_cos[1] || !m->rope_sin[0] || !m->rope_sin[1]) {
        m->cap_L = 0;
        return -1;
    }
    m->cap_L = cap;
    fill_rope(m, cap);
    (void)ensure_rope;
    return 0;
}

/* ================================================================== forward */

static int forward(JuliaModel *m, const int32_t *ids, int L, const int *markers, int nm, int qtype, float *scores, char *err, size_t errlen) {
    const int H = m->H, I = m->I;
    if (ensure_scratch(m, (size_t)L)) return errf(err, errlen, "out of memory for sequence of %d tokens", L);
    float *h = m->h, *x = m->x, *qkv = m->qkv, *att = m->att, *ff = m->ff, *gg = m->gg;

    /* embeddings + norm */
    for (int i = 0; i < L; i++) {
        if (ids[i] < 0 || ids[i] >= m->vocab) return errf(err, errlen, "token id %d outside the model vocabulary", ids[i]);
        st_read(m->emb, (size_t)ids[i] * (size_t)H, (size_t)H, h + (size_t)i * (size_t)H);
    }
    layernorm(h, h, L, H, m->emb_norm.w, m->emb_norm.b, m->eps);

    /* encoder */
    for (int l = 0; l < m->layers; l++) {
        const EncLayer *e = &m->enc[l];
        const float *xin = h;
        if (e->attn_norm.w) {
            layernorm(h, x, L, H, e->attn_norm.w, e->attn_norm.b, m->eps);
            xin = x;
        }
        gemm(xin, H, L, &e->wqkv, qkv, 3 * H, 0);
        const int rt = e->global ? 0 : 1;
        rope_apply(qkv, L, m->heads, m->hd, H, m->rope_cos[rt], m->rope_sin[rt]);
        attention(qkv, (size_t)3 * H, qkv + H, (size_t)3 * H, qkv + 2 * H, (size_t)3 * H, att, (size_t)H, L, L, m->heads, m->hd,
                  e->global ? -1 : m->window);
        gemm(att, H, L, &e->wo, h, H, 1);
        layernorm(h, x, L, H, e->mlp_norm.w, e->mlp_norm.b, m->eps);
        gemm(x, H, L, &e->wi, ff, 2 * I, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < L; i++) {
            const float *a = ff + (size_t)i * (size_t)(2 * I);
            float *g = gg + (size_t)i * (size_t)I;
            for (int j = 0; j < I; j++) g[j] = gelu(a[j]) * a[I + j];
        }
        gemm(gg, I, L, &e->mo, h, H, 1);
    }
    layernorm(h, h, L, H, m->final_norm.w, m->final_norm.b, m->eps);

    /* + type embedding on every position */
    {
        const float *te = m->type_emb + (size_t)qtype * (size_t)H;
        for (int i = 0; i < L; i++) {
            float *r = h + (size_t)i * (size_t)H;
            for (int j = 0; j < H; j++) r[j] += te[j];
        }
    }

    /* decision head: pre-norm transformer layers (ReLU FFN); the last one only
     * computes the marker rows, which is exact because only they are consumed. */
    float *sm = m->sm; /* small buffers: 5 * MAX_OPTIONS * 4H floats */
    const size_t SMW = (size_t)MAX_OPTIONS * (size_t)(4 * H);
    float *r = sm, *xr = sm + SMW, *qm = sm + 2 * SMW, *am = sm + 3 * SMW, *gm = sm + 4 * SMW;
    const int hh = m->head_heads, hhd = m->head_hd;
    int have_rows = 0;
    for (int l = 0; l < m->head_layers; l++) {
        const HeadLayer *hl = &m->head[l];
        layernorm(h, x, L, H, hl->n1.w, hl->n1.b, 1e-5f);
        const int last = (l == m->head_layers - 1);
        if (!last) {
            float *kv = qkv + (size_t)L * (size_t)H;
            gemm(x, H, L, &hl->q, qkv, H, 0);
            gemm(x, H, L, &hl->kv, kv, 2 * H, 0);
            attention(qkv, (size_t)H, kv, (size_t)2 * H, kv + H, (size_t)2 * H, att, (size_t)H, L, L, hh, hhd, -1);
            gemm(att, H, L, &hl->out, h, H, 1);
            layernorm(h, x, L, H, hl->n2.w, hl->n2.b, 1e-5f);
            gemm(x, H, L, &hl->l1, gg, 4 * H, 0);
            for (size_t i = 0; i < (size_t)L * (size_t)(4 * H); i++) if (gg[i] < 0.f) gg[i] = 0.f;
            gemm(gg, 4 * H, L, &hl->l2, h, H, 1);
        } else {
            float *kv = qkv + (size_t)32 * (size_t)H;
            for (int j = 0; j < nm; j++) {
                memcpy(xr + (size_t)j * (size_t)H, x + (size_t)markers[j] * (size_t)H, (size_t)H * sizeof(float));
                memcpy(r + (size_t)j * (size_t)H, h + (size_t)markers[j] * (size_t)H, (size_t)H * sizeof(float));
            }
            gemm(xr, H, nm, &hl->q, qm, H, 0);
            gemm(x, H, L, &hl->kv, kv, 2 * H, 0);
            attention(qm, (size_t)H, kv, (size_t)2 * H, kv + H, (size_t)2 * H, am, (size_t)H, nm, L, hh, hhd, -1);
            gemm(am, H, nm, &hl->out, r, H, 1);
            layernorm(r, xr, nm, H, hl->n2.w, hl->n2.b, 1e-5f);
            gemm(xr, H, nm, &hl->l1, gm, 4 * H, 0);
            for (size_t i = 0; i < (size_t)nm * (size_t)(4 * H); i++) if (gm[i] < 0.f) gm[i] = 0.f;
            gemm(gm, 4 * H, nm, &hl->l2, r, H, 1);
            have_rows = 1;
        }
    }
    if (!have_rows)
        for (int j = 0; j < nm; j++) memcpy(r + (size_t)j * (size_t)H, h + (size_t)markers[j] * (size_t)H, (size_t)H * sizeof(float));

    /* scorer: LayerNorm -> Linear -> GELU -> Linear(1) */
    layernorm(r, xr, nm, H, m->sc_norm.w, m->sc_norm.b, 1e-5f);
    gemm(xr, H, nm, &m->sc_lin, qm, H, 0);
    for (int j = 0; j < nm; j++) {
        float *row = qm + (size_t)j * (size_t)H;
        for (int d = 0; d < H; d++) row[d] = gelu(row[d]);
        scores[j] = dot(row, m->sc_w3, H) + m->sc_b3;
    }
    return 0;
}

/* =============================================================== encoding */

typedef struct {
    IdVec ids;
    int markers[MAX_OPTIONS];
    int nm, qtype;
} Enc;

static int contains_mask(const JuliaModel *m, const char *s, size_t n) {
    size_t k = strlen(m->mask_str);
    if (!k || n < k) return 0;
    for (size_t i = 0; i + k <= n; i++)
        if (s[i] == m->mask_str[0] && memcmp(s + i, m->mask_str, k) == 0) return 1;
    return 0;
}

/* text.replace(mask_token, ' ') */
static int clean_text(const JuliaModel *m, const char *s, size_t n, Buf *out) {
    size_t k = strlen(m->mask_str), i = 0;
    while (i < n) {
        if (k && i + k <= n && s[i] == m->mask_str[0] && memcmp(s + i, m->mask_str, k) == 0) {
            if (buf_add(out, " ", 1)) return -1;
            i += k;
        } else {
            size_t j = i + 1;
            while (j < n && !(k && s[j] == m->mask_str[0])) j++;
            if (buf_add(out, s + i, j - i)) return -1;
            i = j;
        }
    }
    return 0;
}

static int tokenize_clean(const JuliaModel *m, const char *prefix, const char *s, size_t n, IdVec *out, char *err, size_t errlen) {
    Buf b = {0};
    if (prefix && buf_str(&b, prefix)) return errf(err, errlen, "out of memory");
    if (clean_text(m, s, n, &b)) { free(b.p); return errf(err, errlen, "out of memory"); }
    int rc = tok_encode(m->tok, b.p ? b.p : "", b.n, out, err, errlen);
    free(b.p);
    return rc;
}

/* Port of julia.data.sequence(). */
static int encode_row(const JuliaModel *m, const char *state, size_t sn, const char *question, size_t qn, const char *type, int qtype,
                      int nopt, const char *const *opts, const size_t *olens, Enc *e, char *err, size_t errlen) {
    const int max_length = m->opt.max_length, head_length = m->opt.head_length, strict = m->opt.strict;
    memset(e, 0, sizeof *e);
    e->qtype = qtype;
    if (head_length + 4 >= max_length) return errf(err, errlen, "max_length must leave room beyond the question head");
    if (strict) {
        int bad = contains_mask(m, state, sn) || contains_mask(m, question, qn);
        for (int i = 0; i < nopt && !bad; i++) bad = contains_mask(m, opts[i], olens[i]);
        if (bad) return errf(err, errlen, "Reserved model marker in request");
    }
    IdVec head = {0}, optv[MAX_OPTIONS], statev = {0};
    memset(optv, 0, sizeof optv);
    int rc = -1;
    Buf pre = {0};
    buf_str(&pre, type);
    buf_str(&pre, " question: ");
    if (tokenize_clean(m, pre.p, question, qn, &head, err, errlen)) goto done;
    int olen[MAX_OPTIONS], total = 0;
    for (int i = 0; i < nopt; i++) {
        if (tokenize_clean(m, " ", opts[i], olens[i], &optv[i], err, errlen)) goto done;
        if (strict && optv[i].n > 48) { errf(err, errlen, "Option exceeds 48-token model contract"); goto done; }
        olen[i] = 1 + (int)(optv[i].n > 48 ? 48 : optv[i].n);
        total += olen[i];
    }
    int budget = head_length - total;
    if (budget < 16) {
        int per = (head_length - 16) / nopt;
        if (per < 4) per = 4;
        total = 0;
        for (int i = 0; i < nopt; i++) {
            if (olen[i] > per) olen[i] = per;
            total += olen[i];
        }
        budget = head_length - total;
    }
    if (strict) {
        int lossless = (int)head.n <= budget;
        for (int i = 0; i < nopt; i++) if (olen[i] != (int)optv[i].n + 1) lossless = 0;
        if (!lossless) { errf(err, errlen, "Question/options exceed lossless head budget"); goto done; }
    }
    idvec_push(&e->ids, m->id_cls);
    int keep = budget > 8 ? budget : 8;
    for (size_t i = 0; i < head.n && (int)i < keep; i++) idvec_push(&e->ids, head.v[i]);
    idvec_push(&e->ids, m->id_sep);
    for (int i = 0; i < nopt; i++) {
        e->markers[e->nm++] = (int)e->ids.n;
        idvec_push(&e->ids, m->id_mask);
        for (int j = 0; j < olen[i] - 1; j++) idvec_push(&e->ids, optv[i].v[j]);
    }
    idvec_push(&e->ids, m->id_sep);
    if (tokenize_clean(m, NULL, state, sn, &statev, err, errlen)) goto done;
    long room = (long)max_length - (long)e->ids.n - 1;
    if (room < 1) { errf(err, errlen, "Question/options exceed sequence budget; shorten descriptions"); goto done; }
    if (strict && (long)statev.n > room) { errf(err, errlen, "Game state exceeds lossless context budget"); goto done; }
    for (size_t i = 0; i < statev.n && (long)i < room; i++) idvec_push(&e->ids, statev.v[i]);
    idvec_push(&e->ids, m->id_sep);
    rc = 0;
done:
    free(pre.p);
    idvec_free(&head);
    idvec_free(&statev);
    for (int i = 0; i < nopt; i++) idvec_free(&optv[i]);
    if (rc) idvec_free(&e->ids);
    return rc;
}

/* ================================================================== loading */

static int jint(const JNode *o, const char *k, int dflt) {
    const JNode *v = jobj_get(o, k);
    return v && v->type == J_NUM ? (int)v->b.num : dflt;
}
static double jnum(const JNode *o, const char *k, double dflt) {
    const JNode *v = jobj_get(o, k);
    return v && v->type == J_NUM ? v->b.num : dflt;
}
static int jbool(const JNode *o, const char *k, int dflt) {
    const JNode *v = jobj_get(o, k);
    if (!v) return dflt;
    return v->type == J_TRUE;
}

static char *join(const char *dir, const char *rel) {
    size_t a = strlen(dir), b = strlen(rel);
    char *s = (char *)malloc(a + b + 2);
    memcpy(s, dir, a);
    size_t n = a;
    if (a && dir[a - 1] != '/') s[n++] = '/';
    memcpy(s + n, rel, b + 1);
    return s;
}

static JNode *load_json_file(Arena *a, const char *dir, const char *rel, char **keep, char *err, size_t errlen) {
    char *path = join(dir, rel);
    size_t n;
    char *text = read_file(path, &n);
    if (!text) { errf(err, errlen, "cannot read %s", path); free(path); return NULL; }
    char perr[200];
    JNode *root = json_parse(a, text, n, perr, sizeof perr);
    if (!root) errf(err, errlen, "%s: %s", path, perr);
    free(path);
    *keep = text; /* strings point into it */
    return root;
}

static void token_string(const JNode *cfg, const char *key, const char *dflt, char *out, size_t outn) {
    const JNode *v = jobj_get(cfg, key);
    if (v && v->type == J_OBJ) v = jobj_get(v, "content");
    if (v && v->type == J_STR && v->n < outn) {
        memcpy(out, v->a.s, v->n);
        out[v->n] = 0;
    } else snprintf(out, outn, "%s", dflt);
}

void julia_free(JuliaModel *m) {
    if (!m) return;
    tok_free(m->tok);
    if (m->enc) {
        for (int i = 0; i < m->layers; i++) {
            free(m->enc[i].attn_norm.w); free(m->enc[i].attn_norm.b); free(m->enc[i].mlp_norm.w); free(m->enc[i].mlp_norm.b);
            free_lin(&m->enc[i].wqkv); free_lin(&m->enc[i].wo); free_lin(&m->enc[i].wi); free_lin(&m->enc[i].mo);
        }
        free(m->enc);
    }
    if (m->head) {
        for (int i = 0; i < m->head_layers; i++) {
            HeadLayer *l = &m->head[i];
            free(l->n1.w); free(l->n1.b); free(l->n2.w); free(l->n2.b);
            free_lin(&l->q); free_lin(&l->kv); free_lin(&l->out); free_lin(&l->l1); free_lin(&l->l2);
        }
        free(m->head);
    }
    free(m->emb_norm.w); free(m->emb_norm.b); free(m->final_norm.w); free(m->final_norm.b);
    free(m->type_emb); free(m->sc_norm.w); free(m->sc_norm.b); free_lin(&m->sc_lin); free(m->sc_w3);
    for (size_t i = 0; i < m->ntensors; i++) free(m->tensors[i].name);
    free(m->tensors);
    if (m->map) {
#ifdef JULIA_HAVE_MMAP
        if (m->mapped) munmap(m->map, m->maplen);
        else free(m->map);
#else
        free(m->map);
#endif
    }
    free(m->h); free(m->x); free(m->qkv); free(m->att); free(m->ff); free(m->gg);
    for (int t = 0; t < 2; t++) { free(m->rope_cos[t]); free(m->rope_sin[t]); }
    free(m->sm);
    free(m);
}

JuliaModel *julia_load(const char *dir, const JuliaOptions *opt, char *err, size_t errlen) {
    JuliaModel *m = (JuliaModel *)calloc(1, sizeof(JuliaModel));
    if (!m) { errf(err, errlen, "out of memory"); return NULL; }
    m->opt = opt ? *opt : julia_default_options();
#ifdef _OPENMP
    if (m->opt.threads > 0) omp_set_num_threads(m->opt.threads);
#endif
    Arena *a = arena_new();
    char *t1 = NULL, *t2 = NULL, *t3 = NULL;
    JNode *jc = load_json_file(a, dir, "julia_config.json", &t1, err, errlen);
    JNode *ec = jc ? load_json_file(a, dir, "config.json", &t2, err, errlen) : NULL;
    if (!jc || !ec) goto fail;
    if (jint(jc, "format_version", 1) != 1) { errf(err, errlen, "Unsupported Julia checkpoint format"); goto fail; }
    const JNode *arch = jobj_get(jc, "architecture");
    if (arch && !jstr_eq(arch, "JuliaDecisionModel")) { errf(err, errlen, "julia_config.json: unexpected architecture"); goto fail; }
    m->head_layers = jint(jc, "head_layers", 2);

    m->H = jint(ec, "hidden_size", 384);
    m->I = jint(ec, "intermediate_size", 1152);
    m->heads = jint(ec, "num_attention_heads", 6);
    m->layers = jint(ec, "num_hidden_layers", 22);
    m->vocab = jint(ec, "vocab_size", 256000);
    m->max_pos = jint(ec, "max_position_embeddings", 8192);
    m->window = jint(ec, "local_attention", 128) / 2;
    m->eps = (float)jnum(ec, "norm_eps", jnum(ec, "layer_norm_eps", 1e-5));
    m->attn_bias = jbool(ec, "attention_bias", 0);
    m->mlp_bias = jbool(ec, "mlp_bias", 0);
    m->norm_bias = jbool(ec, "norm_bias", 0);
    if (m->H % m->heads) { errf(err, errlen, "hidden_size not divisible by num_attention_heads"); goto fail; }
    m->hd = m->H / m->heads;
    if (m->hd > 128 || (m->hd & 1)) { errf(err, errlen, "unsupported head_dim %d (need even and <= 128)", m->hd); goto fail; }
    const JNode *act = jobj_get(ec, "hidden_activation");
    if (act && !jstr_eq(act, "gelu")) { errf(err, errlen, "unsupported hidden_activation (only exact gelu)"); goto fail; }
    const JNode *pe = jobj_get(ec, "position_embedding_type");
    if (pe && !jstr_eq(pe, "sans_pos") && !jstr_eq(pe, "rope")) { errf(err, errlen, "unsupported position_embedding_type"); goto fail; }
    m->theta_global = jnum(ec, "global_rope_theta", 160000.0);
    m->theta_local = jnum(ec, "local_rope_theta", 10000.0);
    const JNode *rp = jobj_get(ec, "rope_parameters");
    if (rp) {
        const JNode *fg = jobj_get(rp, "full_attention"), *sl = jobj_get(rp, "sliding_attention");
        if (fg || sl) {
            if (fg) m->theta_global = jnum(fg, "rope_theta", m->theta_global);
            if (sl) m->theta_local = jnum(sl, "rope_theta", m->theta_local);
            if ((fg && jobj_get(fg, "rope_type") && !jstr_eq(jobj_get(fg, "rope_type"), "default")) ||
                (sl && jobj_get(sl, "rope_type") && !jstr_eq(jobj_get(sl, "rope_type"), "default"))) {
                errf(err, errlen, "unsupported rope_type (only 'default')");
                goto fail;
            }
        } else {
            m->theta_global = m->theta_local = jnum(rp, "rope_theta", m->theta_global);
        }
    }
    if (m->opt.max_length < 1 || m->opt.max_length > m->max_pos) {
        errf(err, errlen, "max_length must be an integer between 1 and %d", m->max_pos);
        goto fail;
    }
    m->enc = (EncLayer *)calloc((size_t)m->layers, sizeof(EncLayer));
    const JNode *lt = jobj_get(ec, "layer_types");
    const int every = jint(ec, "global_attn_every_n_layers", 3);
    for (int i = 0; i < m->layers; i++) {
        if (jis(lt, J_ARR) && (int)lt->n == m->layers) m->enc[i].global = jstr_eq(&lt->a.kids[i], "full_attention");
        else m->enc[i].global = every > 0 ? (i % every == 0) : 1;
    }
    m->head_heads = m->H / 64 > 1 ? m->H / 64 : 1;
    if (m->H % m->head_heads) { errf(err, errlen, "decision head: hidden size not divisible by heads"); goto fail; }
    m->head_hd = m->H / m->head_heads;
    if (m->head_hd > 128) { errf(err, errlen, "decision head: head_dim too large"); goto fail; }

    /* tokenizer */
    JNode *tc = load_json_file(a, dir, "tokenizer_config.json", &t3, err, errlen);
    if (!tc) goto fail;
    char cls[64], sep[64], pad[64];
    token_string(tc, "cls_token", "<bos>", cls, sizeof cls);
    token_string(tc, "sep_token", "<eos>", sep, sizeof sep);
    token_string(tc, "mask_token", "<mask>", m->mask_str, sizeof m->mask_str);
    token_string(tc, "pad_token", "<pad>", pad, sizeof pad);
    char *tpath = join(dir, "tokenizer.json");
    m->tok = tok_load(tpath, err, errlen);
    free(tpath);
    if (!m->tok) goto fail;
    m->id_cls = tok_token_to_id(m->tok, cls, strlen(cls));
    m->id_sep = tok_token_to_id(m->tok, sep, strlen(sep));
    m->id_mask = tok_token_to_id(m->tok, m->mask_str, strlen(m->mask_str));
    m->id_pad = tok_token_to_id(m->tok, pad, strlen(pad));
    if (m->id_cls < 0 || m->id_sep < 0 || m->id_mask < 0) { errf(err, errlen, "Tokenizer must define MASK, CLS and SEP IDs"); goto fail; }

    /* weights */
    char *wpath = join(dir, "model.safetensors");
    int rc = st_open(m, wpath, err, errlen);
    free(wpath);
    if (rc) goto fail;
    char nm[256], nb[256];
    const int H = m->H, I = m->I;
    if (need_tensor(m, "encoder.embeddings.tok_embeddings.weight", 2, m->vocab, H, &m->emb, err, errlen)) goto fail;
    if (load_norm(m, "encoder.embeddings.norm", H, m->norm_bias, &m->emb_norm, err, errlen)) goto fail;
    for (int l = 0; l < m->layers; l++) {
        EncLayer *e = &m->enc[l];
        if (l > 0) {
            snprintf(nm, sizeof nm, "encoder.layers.%d.attn_norm", l);
            if (load_norm(m, nm, H, m->norm_bias, &e->attn_norm, err, errlen)) goto fail;
        }
        snprintf(nm, sizeof nm, "encoder.layers.%d.mlp_norm", l);
        if (load_norm(m, nm, H, m->norm_bias, &e->mlp_norm, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "encoder.layers.%d.attn.Wqkv.weight", l);
        snprintf(nb, sizeof nb, "encoder.layers.%d.attn.Wqkv.bias", l);
        if (load_lin(m, nm, m->attn_bias ? nb : NULL, 3 * H, H, &e->wqkv, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "encoder.layers.%d.attn.Wo.weight", l);
        snprintf(nb, sizeof nb, "encoder.layers.%d.attn.Wo.bias", l);
        if (load_lin(m, nm, m->attn_bias ? nb : NULL, H, H, &e->wo, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "encoder.layers.%d.mlp.Wi.weight", l);
        snprintf(nb, sizeof nb, "encoder.layers.%d.mlp.Wi.bias", l);
        if (load_lin(m, nm, m->mlp_bias ? nb : NULL, 2 * I, H, &e->wi, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "encoder.layers.%d.mlp.Wo.weight", l);
        snprintf(nb, sizeof nb, "encoder.layers.%d.mlp.Wo.bias", l);
        if (load_lin(m, nm, m->mlp_bias ? nb : NULL, H, I, &e->mo, err, errlen)) goto fail;
    }
    if (load_norm(m, "encoder.final_norm", H, m->norm_bias, &m->final_norm, err, errlen)) goto fail;
    {
        const STensor *te;
        if (need_tensor(m, "type_emb.weight", 2, 3, H, &te, err, errlen)) goto fail;
        m->type_emb = (float *)malloc((size_t)3 * (size_t)H * sizeof(float));
        if (!m->type_emb) { errf(err, errlen, "out of memory"); goto fail; }
        st_read(te, 0, (size_t)3 * (size_t)H, m->type_emb);
    }
    m->head = (HeadLayer *)calloc((size_t)(m->head_layers ? m->head_layers : 1), sizeof(HeadLayer));
    for (int l = 0; l < m->head_layers; l++) {
        HeadLayer *hl = &m->head[l];
        const STensor *w, *b;
        char p[160];
        snprintf(p, sizeof p, "head.layers.%d", l);
        if (load_norm(m, (snprintf(nm, sizeof nm, "%s.norm1", p), nm), H, 1, &hl->n1, err, errlen)) goto fail;
        if (load_norm(m, (snprintf(nm, sizeof nm, "%s.norm2", p), nm), H, 1, &hl->n2, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "%s.self_attn.in_proj_weight", p);
        snprintf(nb, sizeof nb, "%s.self_attn.in_proj_bias", p);
        if (need_tensor(m, nm, 2, 3 * H, H, &w, err, errlen) || need_tensor(m, nb, 1, 3 * H, 0, &b, err, errlen)) goto fail;
        if (load_lin_rows(m, w, 0, H, H, NULL, &hl->q, err, errlen)) goto fail;
        hl->q.bias = (float *)malloc((size_t)H * sizeof(float));
        st_read(b, 0, (size_t)H, hl->q.bias);
        if (load_lin_rows(m, w, (size_t)H, 2 * H, H, NULL, &hl->kv, err, errlen)) goto fail;
        hl->kv.bias = (float *)malloc((size_t)(2 * H) * sizeof(float));
        st_read(b, (size_t)H, (size_t)(2 * H), hl->kv.bias);
        snprintf(nm, sizeof nm, "%s.self_attn.out_proj.weight", p);
        snprintf(nb, sizeof nb, "%s.self_attn.out_proj.bias", p);
        if (load_lin(m, nm, nb, H, H, &hl->out, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "%s.linear1.weight", p);
        snprintf(nb, sizeof nb, "%s.linear1.bias", p);
        if (load_lin(m, nm, nb, 4 * H, H, &hl->l1, err, errlen)) goto fail;
        snprintf(nm, sizeof nm, "%s.linear2.weight", p);
        snprintf(nb, sizeof nb, "%s.linear2.bias", p);
        if (load_lin(m, nm, nb, H, 4 * H, &hl->l2, err, errlen)) goto fail;
    }
    if (load_norm(m, "scorer.0", H, 1, &m->sc_norm, err, errlen)) goto fail;
    if (load_lin(m, "scorer.1.weight", "scorer.1.bias", H, H, &m->sc_lin, err, errlen)) goto fail;
    {
        const STensor *w, *b;
        if (need_tensor(m, "scorer.3.weight", 2, 1, H, &w, err, errlen) || need_tensor(m, "scorer.3.bias", 1, 1, 0, &b, err, errlen)) goto fail;
        m->sc_w3 = (float *)malloc((size_t)H * sizeof(float));
        st_read(w, 0, (size_t)H, m->sc_w3);
        st_read(b, 0, 1, &m->sc_b3);
    }
    m->sm = (float *)malloc((size_t)5 * MAX_OPTIONS * (size_t)(4 * H) * sizeof(float));
    if (!m->sm || !m->type_emb) { errf(err, errlen, "out of memory"); goto fail; }
    arena_free(a);
    free(t1); free(t2); free(t3);
    return m;
fail:
    arena_free(a);
    free(t1); free(t2); free(t3);
    julia_free(m);
    return NULL;
}

void julia_describe(const JuliaModel *m, void *file) {
    FILE *f = (FILE *)file;
    fprintf(f, "Julia decision model: hidden=%d inter=%d heads=%d layers=%d vocab=%d window=+-%d head_layers=%d\n", m->H, m->I, m->heads,
            m->layers, m->vocab, m->window, m->head_layers);
    fprintf(f, "rope theta: global=%g local=%g  norm_eps=%g\n", m->theta_global, m->theta_local, (double)m->eps);
    fprintf(f, "special ids: cls=%d sep=%d mask=%d pad=%d  max_length=%d head_length=%d strict=%d\n", m->id_cls, m->id_sep, m->id_mask, m->id_pad,
            m->opt.max_length, m->opt.head_length, m->opt.strict);
    tok_describe(m->tok, f);
}

void julia_list_tensors(const JuliaModel *m, void *file) {
    FILE *f = (FILE *)file;
    static const char *dn[] = {"F32", "F16", "BF16", "?"};
    for (size_t i = 0; i < m->ntensors; i++) {
        const STensor *t = &m->tensors[i];
        fprintf(f, "%s %s [", t->name, dn[t->dtype]);
        for (int d = 0; d < t->ndim; d++) fprintf(f, "%s%lld", d ? "," : "", (long long)t->shape[d]);
        fprintf(f, "]\n");
    }
}

int julia_tokenize(const JuliaModel *m, const char *text, size_t len, int32_t **ids, size_t *n, char *err, size_t errlen) {
    IdVec v = {0};
    if (tok_encode(m->tok, text, len, &v, err, errlen)) { idvec_free(&v); return -1; }
    *ids = v.v;
    *n = v.n;
    return 0;
}

/* ============================================================== JSON glue */

static int jesc(Buf *b, const char *s, size_t n) {
    if (buf_add(b, "\"", 1)) return -1;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && (unsigned char)s[j] >= 0x20 && s[j] != '"' && s[j] != '\\') j++;
        if (j > i && buf_add(b, s + i, j - i)) return -1;
        if (j >= n) break;
        unsigned char c = (unsigned char)s[j];
        int rc;
        switch (c) {
            case '"': rc = buf_str(b, "\\\""); break;
            case '\\': rc = buf_str(b, "\\\\"); break;
            case '\n': rc = buf_str(b, "\\n"); break;
            case '\r': rc = buf_str(b, "\\r"); break;
            case '\t': rc = buf_str(b, "\\t"); break;
            case '\b': rc = buf_str(b, "\\b"); break;
            case '\f': rc = buf_str(b, "\\f"); break;
            default: rc = buf_printf(b, "\\u%04x", c);
        }
        if (rc) return -1;
        i = j + 1;
    }
    return buf_add(b, "\"", 1);
}

/* repr(float) as Python prints it. */
static int py_float(Buf *b, double x) {
    if (isinf(x)) return buf_str(b, x < 0 ? "-Infinity" : "Infinity");
    if (isnan(x)) return buf_str(b, "NaN");
    if (x == 0) return buf_str(b, signbit(x) ? "-0.0" : "0.0");
    double ax = fabs(x);
    char tmp[48];
    int p;
    for (p = 1; p <= 17; p++) {
        snprintf(tmp, sizeof tmp, "%.*e", p - 1, ax);
        if (strtod(tmp, NULL) == ax) break;
    }
    char digits[24];
    int nd = 0;
    const char *c = tmp;
    for (; *c && *c != 'e'; c++) if (*c >= '0' && *c <= '9') digits[nd++] = *c;
    int e10 = atoi(c + 1);
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    int decpt = e10 + 1;
    if (x < 0 && buf_str(b, "-")) return -1;
    if (decpt > 16 || decpt < -3) {
        if (buf_add(b, digits, 1)) return -1;
        if (nd > 1 && (buf_str(b, ".") || buf_add(b, digits + 1, (size_t)nd - 1))) return -1;
        return buf_printf(b, "e%c%02d", e10 < 0 ? '-' : '+', abs(e10));
    }
    if (decpt <= 0) {
        if (buf_str(b, "0.")) return -1;
        for (int i = 0; i < -decpt; i++) if (buf_str(b, "0")) return -1;
        return buf_add(b, digits, (size_t)nd);
    }
    if (decpt >= nd) {
        if (buf_add(b, digits, (size_t)nd)) return -1;
        for (int i = 0; i < decpt - nd; i++) if (buf_str(b, "0")) return -1;
        return buf_str(b, ".0");
    }
    if (buf_add(b, digits, (size_t)decpt) || buf_str(b, ".")) return -1;
    return buf_add(b, digits + decpt, (size_t)(nd - decpt));
}

/* json.dumps(value, ensure_ascii=False) with Python's default separators. */
static int py_dumps(const JNode *n, Buf *b) {
    switch (n->type) {
        case J_NULL: return buf_str(b, "null");
        case J_TRUE: return buf_str(b, "true");
        case J_FALSE: return buf_str(b, "false");
        case J_NUM: {
            int is_float = 0;
            for (uint32_t i = 0; i < n->n; i++) if (n->a.s[i] == '.' || n->a.s[i] == 'e' || n->a.s[i] == 'E') is_float = 1;
            if (is_float) return py_float(b, n->b.num);
            if (n->n == 2 && n->a.s[0] == '-' && n->a.s[1] == '0') return buf_str(b, "0");
            return buf_add(b, n->a.s, n->n);
        }
        case J_STR: return jesc(b, n->a.s, n->n);
        case J_ARR:
            if (buf_str(b, "[")) return -1;
            for (uint32_t i = 0; i < n->n; i++) {
                if (i && buf_str(b, ", ")) return -1;
                if (py_dumps(&n->a.kids[i], b)) return -1;
            }
            return buf_str(b, "]");
        default:
            if (buf_str(b, "{")) return -1;
            for (uint32_t i = 0; i < n->n; i++) {
                if (i && buf_str(b, ", ")) return -1;
                if (jesc(b, n->b.keys[i].s, n->b.keys[i].n) || buf_str(b, ": ") || py_dumps(&n->a.kids[i], b)) return -1;
            }
            return buf_str(b, "}");
    }
}

typedef struct {
    int qtype;
    const char *type;
    int n;
    const char *opts[MAX_OPTIONS];
    size_t olens[MAX_OPTIONS];
} RowSpec;

static int type_code(const JNode *t, const char **name, int dflt_ok) {
    if (!t && dflt_ok) { *name = "choice"; return 0; }
    if (jstr_eq(t, "choice")) { *name = "choice"; return 0; }
    if (jstr_eq(t, "score")) { *name = "score"; return 1; }
    if (jstr_eq(t, "noul")) { *name = "noul"; return 2; }
    return -1;
}

/* Port of data.validate_row for an already-built row. */
static int validate_row(int line, const JNode *state, const JNode *question, const JNode *type, int type_default_ok, const JNode *const *opt_nodes,
                        int nopt, int opts_are_array, RowSpec *rs, char *err, size_t errlen) {
    if (!state || !(state->type == J_STR || state->type == J_OBJ || state->type == J_ARR) || !jis(question, J_STR))
        return errf(err, errlen, "request %d: state must be text/JSON and question must be text", line);
    if (!opts_are_array || nopt < 2 || nopt > MAX_OPTIONS) return errf(err, errlen, "request %d: options must contain 2-20 nonempty rendered descriptions", line);
    for (int i = 0; i < nopt; i++)
        if (!jis(opt_nodes[i], J_STR) || opt_nodes[i]->n == 0) return errf(err, errlen, "request %d: options must contain 2-20 nonempty rendered descriptions", line);
    rs->qtype = type_code(type, &rs->type, type_default_ok);
    if (rs->qtype < 0) return errf(err, errlen, "request %d: type must be choice, score, or noul", line);
    if (rs->qtype == 2 && nopt != 2) return errf(err, errlen, "request %d: noul options must be ordered [false, true]", line);
    rs->n = nopt;
    for (int i = 0; i < nopt; i++) { rs->opts[i] = opt_nodes[i]->a.s; rs->olens[i] = opt_nodes[i]->n; }
    return 0;
}

static int run_row(JuliaModel *m, const Buf *state_text, const JNode *question, const RowSpec *rs, float *scores, Enc *enc, char *err, size_t errlen) {
    Enc e;
    if (encode_row(m, state_text->p ? state_text->p : "", state_text->n, question->a.s, question->n, rs->type, rs->qtype, rs->n, rs->opts, rs->olens, &e, err, errlen))
        return -1;
    int rc = forward(m, e.ids.v, (int)e.ids.n, e.markers, e.nm, e.qtype, scores, err, errlen);
    if (enc) *enc = e; else idvec_free(&e.ids);
    if (rc && enc) idvec_free(&enc->ids);
    return rc;
}

static int state_to_text(const JNode *state, Buf *out) {
    if (state->type == J_STR) return buf_add(out, state->a.s, state->n);
    return py_dumps(state, out);
}

static void debug_json(Buf *out, const Enc *e, const float *scores, int n) {
    buf_str(out, ", \"debug\": {\"ids\": [");
    for (size_t i = 0; i < e->ids.n; i++) buf_printf(out, "%s%d", i ? ", " : "", e->ids.v[i]);
    buf_str(out, "], \"markers\": [");
    for (int i = 0; i < e->nm; i++) buf_printf(out, "%s%d", i ? ", " : "", e->markers[i]);
    buf_str(out, "], \"logits\": [");
    for (int i = 0; i < n; i++) buf_printf(out, "%s%.9g", i ? ", " : "", (double)scores[i]);
    buf_str(out, "]}");
}

static void display_probabilities(double *p, int n) {
    int w = 0;
    for (int i = 1; i < n; i++) if (p[i] > p[w]) w = i;
    int dominant = p[w] > 0.95;
    for (int i = 0; i < n && dominant; i++) if (i != w && !(p[i] < 0.045)) dominant = 0;
    if (dominant) {
        for (int i = 0; i < n; i++) p[i] = i == w ? 1.0 : 0.0;
        return;
    }
    double total = 0;
    for (int i = 0; i < n; i++) { if (p[i] < 0.01) p[i] = 0.0; total += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= total;
}

static void softmax_d(const float *z, int n, double *p) {
    double mx = z[0], sum = 0;
    for (int i = 1; i < n; i++) if (z[i] > mx) mx = z[i];
    for (int i = 0; i < n; i++) { p[i] = exp((double)z[i] - mx); sum += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= sum;
}

char *julia_predict_json(JuliaModel *m, const char *request, size_t len, int legacy, char *err, size_t errlen) {
    Arena *a = arena_new();
    Buf out = {0};
    Buf state_text = {0};
    char perr[256];
    JNode *root = json_parse(a, request, len, perr, sizeof perr);
    if (!root) { errf(err, errlen, "invalid request JSON: %s", perr); goto fail; }

    if (legacy) {
        if (root->type != J_ARR) { errf(err, errlen, "legacy requests must be a JSON array of rows"); goto fail; }
        buf_str(&out, "[");
        for (uint32_t r = 0; r < root->n; r++) {
            const JNode *row = &root->a.kids[r];
            if (row->type != J_OBJ) { errf(err, errlen, "request %u: request must be a JSON object", r + 1); goto fail; }
            const JNode *opts = jobj_get(row, "options");
            const JNode *on[MAX_OPTIONS];
            int nopt = opts && opts->type == J_ARR ? (int)opts->n : 0;
            for (int i = 0; i < nopt && i < MAX_OPTIONS; i++) on[i] = &opts->a.kids[i];
            RowSpec rs;
            if (validate_row((int)r + 1, jobj_get(row, "state"), jobj_get(row, "question"), jobj_get(row, "type"), 1, on, nopt, opts && opts->type == J_ARR, &rs, err, errlen)) goto fail;
            state_text.n = 0;
            if (state_text.p) state_text.p[0] = 0;
            state_to_text(jobj_get(row, "state"), &state_text);
            float z[MAX_OPTIONS];
            Enc e;
            if (run_row(m, &state_text, jobj_get(row, "question"), &rs, z, &e, err, errlen)) goto fail;
            double p[MAX_OPTIONS];
            softmax_d(z, rs.n, p);
            int best = 0;
            for (int i = 1; i < rs.n; i++) if (p[i] > p[best]) best = i;
            display_probabilities(p, rs.n);
            buf_printf(&out, "%s{\"index\": %d, \"probabilities\": [", r ? ", " : "", best);
            for (int i = 0; i < rs.n; i++) buf_printf(&out, "%s%.9g", i ? ", " : "", p[i]);
            buf_str(&out, "]");
            if (m->opt.debug) debug_json(&out, &e, z, rs.n);
            buf_str(&out, "}");
            idvec_free(&e.ids);
        }
        buf_str(&out, "]");
    } else {
        const JNode *state = jobj_get(root, "state"), *qs = jobj_get(root, "questions");
        if (root->type != J_OBJ || !qs || !state) { errf(err, errlen, "Provide legacy rows or state with questions"); goto fail; }
        if (!jis(qs, J_OBJ) || qs->n == 0) { errf(err, errlen, "questions must be a nonempty mapping"); goto fail; }
        if (!(state->type == J_STR || state->type == J_OBJ || state->type == J_ARR)) {
            errf(err, errlen, "request 1: state must be text/JSON and question must be text");
            goto fail;
        }
        if (state_to_text(state, &state_text)) { errf(err, errlen, "out of memory"); goto fail; }
        buf_str(&out, "{\"answers\": {");
        for (uint32_t qi = 0; qi < qs->n; qi++) {
            const JKey *qk = &qs->b.keys[qi];
            const JNode *q = &qs->a.kids[qi];
            if (qk->n == 0 || q->type != J_OBJ) { errf(err, errlen, "Questions require nonempty string IDs and question objects"); goto fail; }
            const JNode *kind = jobj_get(q, "type"), *crit = jobj_get(q, "criteria");
            const JNode *labels[MAX_OPTIONS + 1];
            const char *keys[MAX_OPTIONS + 1];
            size_t klens[MAX_OPTIONS + 1];
            char numkeys[MAX_OPTIONS + 1][8];
            int nl = 0, is_arr = 1;
            const char *tname = NULL;
            JNode falsetrue[2];
            if (jstr_eq(kind, "choice")) {
                if (!jis(crit, J_OBJ)) { errf(err, errlen, "Choice criteria must map nonempty IDs to descriptions"); goto fail; }
                nl = (int)crit->n;
                for (int i = 0; i < nl; i++)
                    if (crit->b.keys[i].n == 0) { errf(err, errlen, "Choice criteria must map nonempty IDs to descriptions"); goto fail; }
                if (nl > MAX_OPTIONS) { errf(err, errlen, "request %u: options must contain 2-20 nonempty rendered descriptions", qi + 1); goto fail; }
                for (int i = 0; i < nl; i++) { labels[i] = &crit->a.kids[i]; keys[i] = crit->b.keys[i].s; klens[i] = crit->b.keys[i].n; }
            } else if (jstr_eq(kind, "score")) {
                if (!jis(crit, J_ARR)) { errf(err, errlen, "Score requires an ordered rubric"); goto fail; }
                nl = (int)crit->n;
                if (nl > MAX_OPTIONS) { errf(err, errlen, "request %u: options must contain 2-20 nonempty rendered descriptions", qi + 1); goto fail; }
                for (int i = 0; i < nl; i++) {
                    labels[i] = &crit->a.kids[i];
                    klens[i] = (size_t)snprintf(numkeys[i], sizeof numkeys[i], "%d", i);
                    keys[i] = numkeys[i];
                }
            } else if (jstr_eq(kind, "noul")) {
                nl = 2;
                keys[0] = "false"; klens[0] = 5;
                keys[1] = "true"; klens[1] = 4;
                if (!crit || crit->type == J_NULL) {
                    for (int i = 0; i < 2; i++) {
                        falsetrue[i].type = J_STR;
                        falsetrue[i].a.s = keys[i];
                        falsetrue[i].n = (uint32_t)klens[i];
                        labels[i] = &falsetrue[i];
                    }
                } else {
                    const JNode *f = jobj_get(crit, "false"), *t = jobj_get(crit, "true");
                    if (!jis(crit, J_OBJ) || crit->n != 2 || !f || !t) { errf(err, errlen, "Noul criteria must map false and true to descriptions"); goto fail; }
                    labels[0] = f;
                    labels[1] = t;
                }
            } else { errf(err, errlen, "Unsupported question type"); goto fail; }
            RowSpec rs;
            if (validate_row((int)qi + 1, state, jobj_get(q, "instructions"), kind, 0, labels, nl, is_arr, &rs, err, errlen)) goto fail;
            tname = rs.type;
            float z[MAX_OPTIONS];
            Enc e;
            if (run_row(m, &state_text, jobj_get(q, "instructions"), &rs, z, &e, err, errlen)) goto fail;
            double p[MAX_OPTIONS];
            softmax_d(z, nl, p);
            int best = 0;
            double expect = 0, maxp = p[0];
            for (int i = 0; i < nl; i++) {
                if (p[i] > p[best]) best = i;
                expect += (double)i * p[i];
                if (p[i] > maxp) maxp = p[i];
            }
            if (qi) buf_str(&out, ", ");
            jesc(&out, qk->s, qk->n);
            buf_printf(&out, ": {\"type\": \"%s\", \"probabilities\": {", tname);
            for (int i = 0; i < nl; i++) {
                if (i) buf_str(&out, ", ");
                jesc(&out, keys[i], klens[i]);
                buf_printf(&out, ": %.9g", p[i]);
            }
            buf_str(&out, "}");
            if (rs.qtype == 0) { buf_str(&out, ", \"choice\": "); jesc(&out, keys[best], klens[best]); }
            else if (rs.qtype == 1) buf_printf(&out, ", \"score\": %.9g", expect);
            else buf_printf(&out, ", \"noul\": %.9g", p[1]);
            if (rs.qtype != 2) buf_printf(&out, ", \"max_probability\": %.9g", maxp);
            if (m->opt.debug) debug_json(&out, &e, z, nl);
            buf_str(&out, "}");
            idvec_free(&e.ids);
        }
        buf_str(&out, "}}");
    }
    arena_free(a);
    free(state_text.p);
    return out.p;
fail:
    arena_free(a);
    free(state_text.p);
    free(out.p);
    return NULL;
}
