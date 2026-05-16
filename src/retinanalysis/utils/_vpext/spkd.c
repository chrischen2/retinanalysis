/*
 * Victor-Purpura spike-train distance — C port of spkd_with_scr.m.
 *
 * The DP is identical to the MATLAB reference; we just allocate two
 * rolling rows instead of the full (n+1)x(m+1) score matrix since the
 * caller only ever needs the final cost.
 *
 * Bulk variants (vp_pairwise, vp_self_pairwise) parallelize across
 * pairs with POSIX threads. Each pair is independent, so we get
 * near-linear scaling with cores once the workload is large enough
 * to amortize pthread_create + pthread_join (we use a static cutoff
 * of ~16 pairs total before spinning up threads).
 *
 * Build (handled automatically by retinanalysis.utils.victor_purpura on
 * first import; this matches that compile command):
 *   cc -O3 -ffast-math -pthread -shared -fPIC -o libspkd.dylib spkd.c
 *   cc -O3 -ffast-math -pthread -shared -fPIC -o libspkd.so     spkd.c
 */

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

static inline double dmin3(double a, double b, double c) {
    double m = a < b ? a : b;
    return c < m ? c : m;
}

static inline double dabs(double x) { return x < 0 ? -x : x; }

/* Single pairwise distance. tli/tlj must be ascending; nspi/nspj are lengths. */
double vp_distance(const double *tli, int nspi,
                   const double *tlj, int nspj,
                   double cost) {
    if (nspi == 0) return (double)nspj;
    if (nspj == 0) return (double)nspi;
    if (cost == 0.0) {
        int diff = nspi - nspj;
        return (double)(diff < 0 ? -diff : diff);
    }

    /* Two rolling rows of the DP table. */
    double *prev = (double *)malloc((nspj + 1) * sizeof(double));
    double *cur  = (double *)malloc((nspj + 1) * sizeof(double));
    if (!prev || !cur) { free(prev); free(cur); return -1.0; }

    for (int j = 0; j <= nspj; j++) prev[j] = (double)j;

    for (int i = 1; i <= nspi; i++) {
        cur[0] = (double)i;
        double ti = tli[i - 1];
        for (int j = 1; j <= nspj; j++) {
            double a = prev[j]     + 1.0;
            double b = cur[j - 1]  + 1.0;
            double c = prev[j - 1] + cost * dabs(ti - tlj[j - 1]);
            cur[j] = dmin3(a, b, c);
        }
        double *tmp = prev; prev = cur; cur = tmp;
    }
    double d = prev[nspj];
    free(prev);
    free(cur);
    return d;
}

/* ------------------------------------------------------------------ */
/* Thread-count resolution                                            */
/*                                                                    */
/* SPKD_NUM_THREADS env var overrides; otherwise use sysconf-reported */
/* online CPU count, capped at 32 (diminishing returns beyond that    */
/* for typical pair counts we see).                                   */
/* ------------------------------------------------------------------ */
static int _resolve_nthreads(void) {
    const char *env = getenv("SPKD_NUM_THREADS");
    if (env && *env) {
        int n = atoi(env);
        if (n > 0) return n;
    }
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
    if (ncpu > 32) ncpu = 32;
    return (int)ncpu;
}

/* ------------------------------------------------------------------ */
/* vp_pairwise: cross-set rectangular matrix                          */
/* ------------------------------------------------------------------ */

typedef struct {
    const double *a_times;
    const int *a_lens;
    const int *a_offs;
    int nA;
    const double *b_times;
    const int *b_lens;
    const int *b_offs;
    int nB;
    double cost;
    double *out;       /* row-major, shape (nA, nB) */
    int tid;
    int nthreads;
} pairwise_args_t;

static void *pairwise_worker(void *arg) {
    pairwise_args_t *a = (pairwise_args_t *)arg;
    /* Stride rows across threads (interleaved) so cache-line contention
       between adjacent threads writing the same output row is avoided
       (each thread writes its own set of rows). */
    for (int i = a->tid; i < a->nA; i += a->nthreads) {
        int a_off = a->a_offs[i];
        for (int j = 0; j < a->nB; j++) {
            int b_off = a->b_offs[j];
            a->out[i * a->nB + j] = vp_distance(
                a->a_times + a_off, a->a_lens[i],
                a->b_times + b_off, a->b_lens[j], a->cost);
        }
    }
    return NULL;
}

void vp_pairwise(const double *a_times, const int *a_lens, int nA,
                 const double *b_times, const int *b_lens, int nB,
                 double cost, double *out) {
    /* Precompute offsets to avoid re-summing each pair. */
    int *a_offs = (int *)malloc((nA + 1) * sizeof(int));
    int *b_offs = (int *)malloc((nB + 1) * sizeof(int));
    if (!a_offs || !b_offs) { free(a_offs); free(b_offs); return; }
    a_offs[0] = 0;
    for (int i = 0; i < nA; i++) a_offs[i + 1] = a_offs[i] + a_lens[i];
    b_offs[0] = 0;
    for (int j = 0; j < nB; j++) b_offs[j + 1] = b_offs[j] + b_lens[j];

    int nthreads = _resolve_nthreads();
    int total_pairs = nA * nB;
    /* Below this many pairs the pthread setup cost dominates — run serial. */
    if (total_pairs < 16 || nthreads <= 1) {
        for (int i = 0; i < nA; i++) {
            for (int j = 0; j < nB; j++) {
                out[i * nB + j] = vp_distance(
                    a_times + a_offs[i], a_lens[i],
                    b_times + b_offs[j], b_lens[j], cost);
            }
        }
        free(a_offs); free(b_offs);
        return;
    }
    /* Cap threads at #rows so we don't have idle workers. */
    if (nthreads > nA) nthreads = nA;

    pthread_t *tids = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    pairwise_args_t *args = (pairwise_args_t *)malloc(
        nthreads * sizeof(pairwise_args_t));
    if (!tids || !args) {
        /* Allocation failure → serial fallback. */
        for (int i = 0; i < nA; i++) {
            for (int j = 0; j < nB; j++) {
                out[i * nB + j] = vp_distance(
                    a_times + a_offs[i], a_lens[i],
                    b_times + b_offs[j], b_lens[j], cost);
            }
        }
        free(tids); free(args);
        free(a_offs); free(b_offs);
        return;
    }
    for (int t = 0; t < nthreads; t++) {
        args[t].a_times = a_times;
        args[t].a_lens = a_lens;
        args[t].a_offs = a_offs;
        args[t].nA = nA;
        args[t].b_times = b_times;
        args[t].b_lens = b_lens;
        args[t].b_offs = b_offs;
        args[t].nB = nB;
        args[t].cost = cost;
        args[t].out = out;
        args[t].tid = t;
        args[t].nthreads = nthreads;
        pthread_create(&tids[t], NULL, pairwise_worker, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);

    free(tids); free(args);
    free(a_offs); free(b_offs);
}

/* ------------------------------------------------------------------ */
/* vp_self_pairwise: upper-triangle symmetric matrix                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const double *times;
    const int *lens;
    const int *offs;
    int n;
    double cost;
    double *out;       /* row-major, shape (n, n) */
    int tid;
    int nthreads;
} self_args_t;

static void *self_worker(void *arg) {
    self_args_t *a = (self_args_t *)arg;
    /* Interleave rows. For row i we compute j = i+1..n-1 (upper
       triangle) and also mirror into out[j*n+i]. Different threads
       write disjoint rows i, so writes to out[i*n+...] don't collide;
       the mirrored write to out[j*n+i] is the only one another thread
       might also touch — but only for column j > i, and another thread
       owns row j's diagonal block. We avoid that hazard by having each
       thread write *both* triangle copies for every (i, j) it owns —
       so each off-diagonal cell is written by exactly one thread. */
    for (int i = a->tid; i < a->n; i += a->nthreads) {
        a->out[i * a->n + i] = 0.0;
        for (int j = i + 1; j < a->n; j++) {
            double d = vp_distance(
                a->times + a->offs[i], a->lens[i],
                a->times + a->offs[j], a->lens[j], a->cost);
            a->out[i * a->n + j] = d;
            a->out[j * a->n + i] = d;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* vp_batch_pairs: arbitrary list of pairs across one shared train     */
/* set. The intended caller flattens *all* VP work (every within and  */
/* every cross pair across every cell + condition) into one call so   */
/* the pthread setup is amortized over the whole workload — the only */
/* shape that actually scales on this codebase, where individual      */
/* self / cross calls are routinely just 3–9 pairs.                   */
/* ------------------------------------------------------------------ */

typedef struct {
    const double *times;
    const int *lens;
    const int *offs;
    int n_trains;
    const int *pair_a;
    const int *pair_b;
    int n_pairs;
    double cost;
    double *out;
    int tid;
    int nthreads;
} batch_args_t;

static void *batch_worker(void *arg) {
    batch_args_t *a = (batch_args_t *)arg;
    for (int k = a->tid; k < a->n_pairs; k += a->nthreads) {
        int i = a->pair_a[k];
        int j = a->pair_b[k];
        a->out[k] = vp_distance(
            a->times + a->offs[i], a->lens[i],
            a->times + a->offs[j], a->lens[j],
            a->cost);
    }
    return NULL;
}

void vp_batch_pairs(const double *times, const int *lens,
                    int n_trains,
                    const int *pair_a, const int *pair_b, int n_pairs,
                    double cost, double *out) {
    if (n_pairs <= 0) return;

    int *offs = (int *)malloc((n_trains + 1) * sizeof(int));
    if (!offs) return;
    offs[0] = 0;
    for (int i = 0; i < n_trains; i++) offs[i + 1] = offs[i] + lens[i];

    int nthreads = _resolve_nthreads();
    if (n_pairs < 16 || nthreads <= 1) {
        for (int k = 0; k < n_pairs; k++) {
            int i = pair_a[k];
            int j = pair_b[k];
            out[k] = vp_distance(times + offs[i], lens[i],
                                  times + offs[j], lens[j], cost);
        }
        free(offs);
        return;
    }
    if (nthreads > n_pairs) nthreads = n_pairs;

    pthread_t *tids = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    batch_args_t *args = (batch_args_t *)malloc(
        nthreads * sizeof(batch_args_t));
    if (!tids || !args) {
        /* Allocation failure → serial fallback. */
        for (int k = 0; k < n_pairs; k++) {
            int i = pair_a[k];
            int j = pair_b[k];
            out[k] = vp_distance(times + offs[i], lens[i],
                                  times + offs[j], lens[j], cost);
        }
        free(tids); free(args); free(offs);
        return;
    }
    for (int t = 0; t < nthreads; t++) {
        args[t].times = times;
        args[t].lens = lens;
        args[t].offs = offs;
        args[t].n_trains = n_trains;
        args[t].pair_a = pair_a;
        args[t].pair_b = pair_b;
        args[t].n_pairs = n_pairs;
        args[t].cost = cost;
        args[t].out = out;
        args[t].tid = t;
        args[t].nthreads = nthreads;
        pthread_create(&tids[t], NULL, batch_worker, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);

    free(tids); free(args); free(offs);
}

void vp_self_pairwise(const double *times, const int *lens, int n,
                      double cost, double *out) {
    int *offs = (int *)malloc((n + 1) * sizeof(int));
    if (!offs) return;
    offs[0] = 0;
    for (int i = 0; i < n; i++) offs[i + 1] = offs[i] + lens[i];

    int nthreads = _resolve_nthreads();
    int n_pairs = n * (n - 1) / 2;
    if (n_pairs < 16 || nthreads <= 1) {
        for (int i = 0; i < n; i++) {
            out[i * n + i] = 0.0;
            for (int j = i + 1; j < n; j++) {
                double d = vp_distance(times + offs[i], lens[i],
                                       times + offs[j], lens[j], cost);
                out[i * n + j] = d;
                out[j * n + i] = d;
            }
        }
        free(offs);
        return;
    }
    if (nthreads > n) nthreads = n;

    pthread_t *tids = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    self_args_t *args = (self_args_t *)malloc(nthreads * sizeof(self_args_t));
    if (!tids || !args) {
        for (int i = 0; i < n; i++) {
            out[i * n + i] = 0.0;
            for (int j = i + 1; j < n; j++) {
                double d = vp_distance(times + offs[i], lens[i],
                                       times + offs[j], lens[j], cost);
                out[i * n + j] = d;
                out[j * n + i] = d;
            }
        }
        free(tids); free(args); free(offs);
        return;
    }
    for (int t = 0; t < nthreads; t++) {
        args[t].times = times;
        args[t].lens = lens;
        args[t].offs = offs;
        args[t].n = n;
        args[t].cost = cost;
        args[t].out = out;
        args[t].tid = t;
        args[t].nthreads = nthreads;
        pthread_create(&tids[t], NULL, self_worker, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);

    free(tids); free(args); free(offs);
}
