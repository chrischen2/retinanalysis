/*
 * Victor-Purpura spike-train distance — C port of spkd_with_scr.m.
 *
 * The DP is identical to the MATLAB reference; we just allocate two
 * rolling rows instead of the full (n+1)x(m+1) score matrix since the
 * caller only ever needs the final cost.
 *
 * Build:
 *   cc -O3 -ffast-math -shared -fPIC -o libspkd.dylib spkd.c   (macOS)
 *   cc -O3 -ffast-math -shared -fPIC -o libspkd.so     spkd.c   (Linux)
 *
 * Python wrapper in ../victor_purpura.py compiles this on first import.
 */

#include <stdlib.h>
#include <string.h>

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

/*
 * Bulk pairwise distances between two flattened sets of spike trains.
 *
 *   a_times[ sum(a_lens) ]    — concatenated spike times of set A
 *   a_lens [ nA ]             — per-train lengths
 *   b_times, b_lens, nB       — same for set B
 *   out    [ nA * nB ]        — row-major output (out[i*nB+j] = d(a_i, b_j))
 */
void vp_pairwise(const double *a_times, const int *a_lens, int nA,
                 const double *b_times, const int *b_lens, int nB,
                 double cost, double *out) {
    int a_off = 0;
    for (int i = 0; i < nA; i++) {
        int b_off = 0;
        for (int j = 0; j < nB; j++) {
            out[i * nB + j] = vp_distance(a_times + a_off, a_lens[i],
                                          b_times + b_off, b_lens[j], cost);
            b_off += b_lens[j];
        }
        a_off += a_lens[i];
    }
}

/*
 * Pairwise distances within a single set (upper triangle, mirrored).
 *   out [n*n] — row-major, with zero diagonal and out[i*n+j]==out[j*n+i].
 */
void vp_self_pairwise(const double *times, const int *lens, int n,
                      double cost, double *out) {
    /* Precompute offsets to avoid re-summing. */
    int *offs = (int *)malloc((n + 1) * sizeof(int));
    if (!offs) return;
    offs[0] = 0;
    for (int i = 0; i < n; i++) offs[i + 1] = offs[i] + lens[i];

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
}
