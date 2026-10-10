// 2-bit matrix-vector on the NPU's vector unit, version 5: real Q2_K-style scales. Every 16 weights of a row
// (one "pouch") carry a 4-bit scale and a 4-bit min, so a weight is scale*q - min (times the row's super-block
// d/dmin, applied later in floating point). Builds on version 4 (32 rows per vector, four tallies, l2fetch).
//
// Weight layout as version 4: one 128-byte vector per (group of 32 rows, block of 16 inputs) = one pouch per row.
// Scale layout: per group, one 128-byte vector per 4 consecutive pouches; word j holds row j's 4 pouches as
// bytes (scale in the low nibble, min in the high nibble). Per super-block (16 pouches, 256 inputs) the kernel
// writes two integer totals per row: S = sum(scale * sum(q * x)) and M = sum(min * sum(x)), so the row's
// answer for that super-block is d * xd * S - dmin * xd * M.
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HAP_perf.h"
#include "HAP_power.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "qurt.h"

#define K 2560
#define N 10240
#define ROWS 32
#define BLOCK 16                  // inputs per pouch / packed vector
#define BLOCKS (K / BLOCK)        // 160 pouches per row
#define SB 16                     // pouches per super-block
#define SBLOCKS (BLOCKS / SB)     // 10 super-blocks per row
#define GROUPS (N / ROWS)
#define MAX_THREADS 2

static int power_status[2];

static void power_up(void) {
    static int ctx;
    HAP_power_request_t r;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = 1;
    power_status[0] = HAP_power_set(&ctx, &r);
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = 1;
    r.dcvs_v3.dcvs_enable = 0;
    r.dcvs_v3.set_core_params = 1;
    r.dcvs_v3.core_params.min_corner = r.dcvs_v3.core_params.max_corner = r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.set_bus_params = 1;
    r.dcvs_v3.bus_params.min_corner = r.dcvs_v3.bus_params.max_corner = r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.set_sleep_disable = 1;
    r.dcvs_v3.sleep_disable = 1;
    power_status[1] = HAP_power_set(&ctx, &r);
}

static inline void l2fetch(const void *p, uint32_t width, uint32_t stride, uint32_t height) {
    Q6_l2fetch_AP((void *) p, Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height)));
}

static inline unsigned code(unsigned n, unsigned k) { return ((n * K + k) * 2654435761u) >> 30; }
static inline unsigned scale_of(unsigned n, unsigned pouch) { return ((n * BLOCKS + pouch) * 40503u >> 12) & 15; }
static inline unsigned min_of(unsigned n, unsigned pouch) { return ((n * BLOCKS + pouch) * 2246822519u >> 28) & 15; }

typedef struct {
    const HVX_Vector *w, *sc;     // packed weights, packed scales
    const int32_t *x4, *xsum;     // inputs four at a time; sum of each pouch's 16 inputs
    int32_t *S, *M;               // [group][super-block][32 rows]
    int g0, g1;
} job_t;

static void *worker(void *arg) {
    job_t *j = arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    const HVX_Vector three = Q6_Vb_vsplat_R(3), nib = Q6_V_vsplat_R(0x0f);
    const uint32_t wbytes = BLOCKS * 128, sbytes = BLOCKS / 4 * 128;
    l2fetch(j->w + (size_t) j->g0 * BLOCKS, wbytes, wbytes, 1);
    l2fetch(j->sc + (size_t) j->g0 * (BLOCKS / 4), sbytes, sbytes, 1);
    for (int g = j->g0; g < j->g1; g++) {
        const HVX_Vector *wg = j->w + (size_t) g * BLOCKS, *sg = j->sc + (size_t) g * (BLOCKS / 4);
        if (g + 1 < j->g1) { l2fetch(wg + BLOCKS, wbytes, wbytes, 1); l2fetch(sg + BLOCKS / 4, sbytes, sbytes, 1); }
        const int32_t *xb = j->x4, *xs = j->xsum;
        for (int s = 0; s < SBLOCKS; s++) {
            HVX_Vector S = Q6_V_vzero(), M = Q6_V_vzero();
            for (int q4 = 0; q4 < SB / 4; q4++) {          // four pouches share one scale vector
                HVX_Vector sv = sg[s * (SB / 4) + q4];
                for (int i = 0; i < 4; i++, xb += 4, xs++) {
                    HVX_Vector pk = wg[s * SB + q4 * 4 + i];
                    HVX_Vector a = Q6_Vw_vrmpy_VubRb(Q6_V_vand_VV(pk, three), xb[0]);
                    HVX_Vector b = Q6_Vw_vrmpy_VubRb(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 2), three), xb[1]);
                    a = Q6_Vw_vrmpyacc_VwVubRb(a, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 4), three), xb[2]);
                    b = Q6_Vw_vrmpyacc_VwVubRb(b, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 6), three), xb[3]);
                    HVX_Vector pouch = Q6_Vw_vadd_VwVw(a, b);                         // sum(q*x) per row
                    HVX_Vector byte = Q6_Vuw_vlsr_VuwR(sv, 8 * i);
                    HVX_Vector scale = Q6_V_vand_VV(byte, nib);
                    HVX_Vector mn = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(byte, 4), nib);
                    S = Q6_Vw_vadd_VwVw(S, Q6_Vw_vmpyie_VwVuh(pouch, scale));          // pouch total * its scale
                    M = Q6_Vw_vadd_VwVw(M, Q6_Vw_vmpyi_VwRh(mn, Q6_R_combine_RlRl(*xs, *xs)));  // min * sum(x)
                }
            }
            int32_t *dst = j->S + ((size_t) g * SBLOCKS + s) * ROWS;
            *(HVX_Vector *) dst = S;
            *(HVX_Vector *) (j->M + ((size_t) g * SBLOCKS + s) * ROWS) = M;
        }
    }
    qurt_hvx_unlock();
    return NULL;
}

static uint64_t run(int threads, job_t base) {
    pthread_t t[MAX_THREADS];
    job_t jobs[MAX_THREADS];
    uint64_t start = HAP_perf_get_time_us();
    for (int i = 0; i < threads; i++) {
        jobs[i] = base;
        jobs[i].g0 = GROUPS * i / threads;
        jobs[i].g1 = GROUPS * (i + 1) / threads;
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, 64 * 1024);
        pthread_create(&t[i], &a, worker, &jobs[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    return HAP_perf_get_time_us() - start;
}

int main(int argc, char *argv[]) {
    FILE *out = fopen("gemv2_v5_result.txt", "w");
    if (!out) return 2;
    power_up();
    size_t wbytes = (size_t) N * K / 4, sbytes = (size_t) N * BLOCKS;
    uint8_t *w = memalign(128, wbytes), *sc = memalign(128, sbytes);
    int8_t *x = memalign(128, K);
    int32_t *x4 = memalign(128, K), *xsum = memalign(128, BLOCKS * 4);
    int32_t *S = memalign(128, (size_t) GROUPS * SBLOCKS * ROWS * 4), *M = memalign(128, (size_t) GROUPS * SBLOCKS * ROWS * 4);
    if (!w || !sc || !x || !x4 || !xsum || !S || !M) { fprintf(out, "allocation failed\n"); fclose(out); return 3; }
    srand(1);
    for (int k = 0; k < K; k++) x[k] = (int8_t) (rand() % 255 - 127);
    memcpy(x4, x, K);
    for (int p = 0; p < BLOCKS; p++) { int s = 0; for (int t = 0; t < BLOCK; t++) s += x[p * BLOCK + t]; xsum[p] = s; }
    for (int g = 0; g < GROUPS; g++)
        for (int b = 0; b < BLOCKS; b++) {
            for (int jj = 0; jj < ROWS; jj++)
                for (int t = 0; t < 4; t++) {
                    uint8_t v = 0;
                    for (int p = 0; p < 4; p++) v |= code(g * ROWS + jj, b * BLOCK + 4 * p + t) << (2 * p);
                    w[((size_t) g * BLOCKS + b) * 128 + 4 * jj + t] = v;
                }
            for (int jj = 0; jj < ROWS; jj++)   // scale byte: pouch b of row jj, in word jj of scale vector b/4, byte b%4
                sc[((size_t) g * (BLOCKS / 4) + b / 4) * 128 + 4 * jj + b % 4] =
                    (uint8_t) (scale_of(g * ROWS + jj, b) | min_of(g * ROWS + jj, b) << 4);
        }
    job_t base = {(const HVX_Vector *) w, (const HVX_Vector *) sc, x4, xsum, S, M, 0, 0};
    fprintf(out, "2-bit gate matrix %dx%d with Q2_K-style scales: weights %.1f MB + scales %.1f MB, version 5\n",
            K, N, wbytes / 1e6, sbytes / 1e6);
    fprintf(out, "power requests: HVX %d, clocks %d (0 = accepted)\n", power_status[0], power_status[1]);
    for (int threads = 1; threads <= MAX_THREADS; threads++) {
        uint64_t best = ~0ull;
        for (int r = 0; r < 3; r++) { uint64_t t = run(threads, base); if (t < best) best = t; }
        fprintf(out, "%d thread%s: %llu us (%.1f GB/s of weights + scales)\n", threads, threads > 1 ? "s" : " ",
                best, (wbytes + sbytes) / (double) best / 1e3);
    }
    int bad = 0, checked = 0;
    for (int n = 0; n < N; n += 97)
        for (int s = 0; s < SBLOCKS; s++, checked++) {
            int32_t rs = 0, rm = 0;
            for (int p = s * SB; p < (s + 1) * SB; p++) {
                int32_t dot = 0;
                for (int t = 0; t < BLOCK; t++) dot += (int32_t) code(n, p * BLOCK + t) * x[p * BLOCK + t];
                rs += (int32_t) scale_of(n, p) * dot;
                rm += (int32_t) min_of(n, p) * xsum[p];
            }
            size_t at = ((size_t) (n / ROWS) * SBLOCKS + s) * ROWS + n % ROWS;
            if (rs != S[at] || rm != M[at]) bad++;
        }
    fprintf(out, "answers: %s (%d mismatches in %d checked row super-blocks)\n", bad ? "WRONG" : "match", bad, checked);
    fclose(out);
    return bad ? 1 : 0;
}
