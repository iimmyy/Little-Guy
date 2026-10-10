// 2-bit matrix-vector on the NPU's vector unit, version 4: version 3 plus l2fetch, which pulls the next group's
// weights into the cache while this group computes (HVX has no hardware prefetch). Version 3: four tallies, plus 32 output rows side by side in each vector, so no
// per-row sum, and the work split over the NPU's 4 HVX contexts. One E4B gate matrix (2560 in, 10240 out).
//
// Packed layout: for each group of 32 rows and each block of 16 inputs there is one 128-byte vector.
// Byte 4*j+t holds, in bits 2p..2p+1, the weight of row (32*group + j) for input (16*block + 4*p + t).
// So unpacking field p of the whole vector gives every row's 4 weights for inputs 4p..4p+3 of the block,
// which vrmpy multiplies against those 4 inputs (one 32-bit register) into 32 separate row totals.
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
#define ROWS 32                 // rows per vector
#define BLOCK 16                // inputs per packed vector
#define BLOCKS (K / BLOCK)      // 160
#define GROUPS (N / ROWS)       // 320
#define MAX_THREADS 4

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

static inline void l2fetch(const void *p, uint32_t width, uint32_t stride, uint32_t height) {  // as llama.cpp's hex_l2fetch
    Q6_l2fetch_AP((void *) p, Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height)));
}

static inline unsigned code(unsigned n, unsigned k) { return ((n * K + k) * 2654435761u) >> 30; }  // a fixed pseudo-random 2-bit weight

typedef struct { const HVX_Vector *w; const int32_t *x4; int32_t *y; int g0, g1; int read_only; int32_t sink; } job_t;

static void *worker(void *arg) {
    job_t *j = arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    const HVX_Vector three = Q6_Vb_vsplat_R(3);
    if (j->read_only) {  // memory road speed: touch every packed byte once
        HVX_Vector acc = Q6_V_vzero();
        for (const HVX_Vector *p = j->w + (size_t) j->g0 * BLOCKS, *e = j->w + (size_t) j->g1 * BLOCKS; p < e; p++)
            acc = Q6_Vw_vadd_VwVw(acc, *p);
        j->sink = ((int32_t *) &acc)[0];
    } else {
        const uint32_t group_bytes = BLOCKS * 128;
        l2fetch(j->w + (size_t) j->g0 * BLOCKS, group_bytes, group_bytes, 1);
        for (int g = j->g0; g < j->g1; g++) {
            const HVX_Vector *wg = j->w + (size_t) g * BLOCKS;
            if (g + 1 < j->g1) l2fetch(wg + BLOCKS, group_bytes, group_bytes, 1);  // next group, while this one computes
            // four tallies, one per bit field, so consecutive vrmpy don't wait on each other
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
            const int32_t *xb = j->x4;
#pragma unroll 4
            for (int b = 0; b < BLOCKS; b++, xb += 4) {
                HVX_Vector pk = wg[b];
                a0 = Q6_Vw_vrmpyacc_VwVubRb(a0, Q6_V_vand_VV(pk, three), xb[0]);
                a1 = Q6_Vw_vrmpyacc_VwVubRb(a1, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 2), three), xb[1]);
                a2 = Q6_Vw_vrmpyacc_VwVubRb(a2, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 4), three), xb[2]);
                a3 = Q6_Vw_vrmpyacc_VwVubRb(a3, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 6), three), xb[3]);
            }
            *(HVX_Vector *) (j->y + g * ROWS) = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1), Q6_Vw_vadd_VwVw(a2, a3));
        }
    }
    qurt_hvx_unlock();
    return NULL;
}

static uint64_t run(int threads, const HVX_Vector *w, const int32_t *x4, int32_t *y, int read_only) {
    pthread_t t[MAX_THREADS];
    job_t jobs[MAX_THREADS];
    uint64_t start = HAP_perf_get_time_us();
    for (int i = 0; i < threads; i++) {
        jobs[i] = (job_t) {w, x4, y, GROUPS * i / threads, GROUPS * (i + 1) / threads, read_only, 0};
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, 64 * 1024);
        pthread_create(&t[i], &a, worker, &jobs[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    return HAP_perf_get_time_us() - start;
}

static uint64_t best(int reps, int threads, const HVX_Vector *w, const int32_t *x4, int32_t *y, int read_only) {
    uint64_t b = ~0ull;
    for (int r = 0; r < reps; r++) { uint64_t t = run(threads, w, x4, y, read_only); if (t < b) b = t; }
    return b;
}

int main(int argc, char *argv[]) {
    FILE *out = fopen("gemv2_v4_result.txt", "w");
    if (!out) return 2;
    power_up();
    size_t bytes = (size_t) N * K / 4;
    uint8_t *w = memalign(128, bytes);
    int8_t *x = memalign(128, K);
    int32_t *y = memalign(128, N * 4);
    if (!w || !x || !y) { fprintf(out, "allocation failed\n"); fclose(out); return 3; }
    srand(1);
    for (int k = 0; k < K; k++) x[k] = (int8_t) (rand() % 255 - 127);
    for (int g = 0; g < GROUPS; g++)
        for (int b = 0; b < BLOCKS; b++)
            for (int jj = 0; jj < ROWS; jj++)
                for (int t = 0; t < 4; t++) {
                    uint8_t v = 0;
                    for (int p = 0; p < 4; p++) v |= code(g * ROWS + jj, b * BLOCK + 4 * p + t) << (2 * p);
                    w[((size_t) g * BLOCKS + b) * 128 + 4 * jj + t] = v;
                }
    int32_t *x4 = memalign(128, K);  // the inputs four at a time, as the 32-bit registers vrmpy wants
    memcpy(x4, x, K);

    fprintf(out, "2-bit gate matrix %dx%d (%.1f MB), version 4 (4 tallies + l2fetch)\n", K, N, bytes / 1e6);
    fprintf(out, "power requests: HVX %d, clocks %d (0 = accepted)\n", power_status[0], power_status[1]);
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        uint64_t tr = best(3, threads, (const HVX_Vector *) w, x4, y, 1);
        uint64_t tg = best(3, threads, (const HVX_Vector *) w, x4, y, 0);
        fprintf(out, "%d thread%s: read %llu us (%.1f GB/s)   gemv %llu us\n", threads, threads > 1 ? "s" : " ",
                tr, bytes / (double) tr / 1e3, tg);
    }
    int bad = 0;
    for (int n = 0; n < N; n += 97) {  // plain C reference on a spread of rows
        int32_t ref = 0;
        for (int k = 0; k < K; k++) ref += (int32_t) code(n, k) * x[k];
        if (ref != y[n]) bad++;
    }
    fprintf(out, "answers: %s (%d mismatches in %d checked rows)\n", bad ? "WRONG" : "match", bad, (N + 96) / 97);
    fclose(out);
    return bad ? 1 : 0;
}
