// Can the NPU's vector unit (HVX) talk with 2-bit weights? One E4B gate matrix (2560 in, 10240 out), one word.
// Times: plain memory read speed, an 8-bit matrix-vector product, and a 2-bit one that unpacks four weights per
// byte with shifts and masks before HVX's 8-bit dot product (vrmpy). Results go to gemv2_result.txt next to it.
// Runs on the cDSP through run_main_on_hexagon.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"

#define K 2560
#define N 10240
#define VB 128  // bytes per HVX vector

static FILE *out;

static void power_up(void) {
    static int ctx;
    HAP_power_request_t r;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = 1;
    HAP_power_set(&ctx, &r);
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = 1;
    r.dcvs_v3.dcvs_enable = 0;  // hold the clocks up instead of letting them wander
    r.dcvs_v3.set_core_params = 1;
    r.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.set_bus_params = 1;
    r.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    r.dcvs_v3.set_sleep_disable = 1;
    r.dcvs_v3.sleep_disable = 1;
    HAP_power_set(&ctx, &r);
}

static int32_t hsum(HVX_Vector v) {  // add the 32 words of a vector together
    for (int s = 64; s >= 4; s >>= 1) v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, s));
    int32_t w[32] __attribute__((aligned(128)));
    *(HVX_Vector *) w = v;
    return w[0];
}

// memory road speed: read every byte once
static int32_t read_all(const uint8_t *p, size_t n) {
    HVX_Vector acc = Q6_V_vzero();
    for (size_t i = 0; i < n; i += VB) acc = Q6_Vw_vadd_VwVw(acc, *(const HVX_Vector *) (p + i));
    return hsum(acc);
}

// 8-bit weights: row n is K bytes; y[n] = sum_k w[n][k] * x[k]
static void gemv8(const uint8_t *w, const int8_t *x, int32_t *y) {
    for (int n = 0; n < N; n++) {
        const HVX_Vector *wr = (const HVX_Vector *) (w + (size_t) n * K), *xv = (const HVX_Vector *) x;
        HVX_Vector acc = Q6_V_vzero();
        for (int c = 0; c < K / VB; c++) acc = Q6_Vw_vrmpyacc_VwVubVb(acc, wr[c], xv[c]);
        y[n] = hsum(acc);
    }
}

// 2-bit weights, 4 per byte: in each 128-byte packed chunk, byte j holds the weights for elements
// j, j+128, j+256, j+384 of a 512-element stretch, in bits 0-1, 2-3, 4-5, 6-7.
static void gemv2(const uint8_t *w, const int8_t *x, int32_t *y) {
    const HVX_Vector three = Q6_Vb_vsplat_R(3);
    for (int n = 0; n < N; n++) {
        const HVX_Vector *wr = (const HVX_Vector *) (w + (size_t) n * K / 4), *xv = (const HVX_Vector *) x;
        HVX_Vector acc = Q6_V_vzero();
        for (int c = 0; c < K / 512; c++) {
            HVX_Vector pk = wr[c];
            for (int p = 0; p < 4; p++) {
                HVX_Vector wv = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 2 * p), three);  // the shift stays inside each byte after the mask
                acc = Q6_Vw_vrmpyacc_VwVubVb(acc, wv, xv[c * 4 + p]);
            }
        }
        y[n] = hsum(acc);
    }
}

static uint64_t best_of(int reps, void (*f)(const uint8_t *, const int8_t *, int32_t *), const uint8_t *w, const int8_t *x, int32_t *y) {
    uint64_t best = ~0ull;
    for (int r = 0; r < reps; r++) {
        uint64_t t = HAP_perf_get_time_us();
        f(w, x, y);
        t = HAP_perf_get_time_us() - t;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char *argv[]) {
    out = fopen("gemv2_result.txt", "w");
    if (!out) return 2;
    power_up();
    uint8_t *w8 = memalign(VB, (size_t) N * K), *w2 = memalign(VB, (size_t) N * K / 4);
    int8_t *x = memalign(VB, K);
    int32_t *y8 = memalign(VB, N * 4), *y2 = memalign(VB, N * 4);
    if (!w8 || !w2 || !x || !y8 || !y2) { fprintf(out, "allocation failed (w8 %p, w2 %p)\n", w8, w2); fclose(out); return 3; }
    srand(1);
    for (int k = 0; k < K; k++) x[k] = (int8_t) (rand() % 255 - 127);
    // 2-bit codes, packed per the layout above; the 8-bit copy holds the same codes one per byte
    for (int n = 0; n < N; n++)
        for (int c = 0; c < K / 512; c++)
            for (int j = 0; j < VB; j++) {
                uint8_t b = 0;
                for (int p = 0; p < 4; p++) {
                    uint8_t q = (uint8_t) (rand() & 3);
                    b |= q << (2 * p);
                    w8[(size_t) n * K + c * 512 + p * VB + j] = q;
                }
                w2[(size_t) n * K / 4 + c * VB + j] = b;
            }

    uint64_t t0 = HAP_perf_get_time_us();
    volatile int32_t sink = read_all(w8, (size_t) N * K);
    uint64_t t_read8 = HAP_perf_get_time_us() - t0;
    t0 = HAP_perf_get_time_us();
    sink += read_all(w2, (size_t) N * K / 4);
    uint64_t t_read2 = HAP_perf_get_time_us() - t0;
    uint64_t t8 = best_of(3, gemv8, w8, x, y8);
    uint64_t t2 = best_of(3, gemv2, w2, x, y2);

    int bad = 0;
    for (int n = 0; n < N; n++) {
        if (y8[n] != y2[n]) bad++;
        if (n < 16) {  // plain C reference for a few rows
            int32_t ref = 0;
            for (int k = 0; k < K; k++) ref += w8[(size_t) n * K + k] * x[k];
            if (ref != y8[n]) bad++;
        }
    }
    fprintf(out, "one gate matrix %dx%d, one HVX thread\n", K, N);
    fprintf(out, "read 8-bit weights (%.1f MB): %llu us = %.1f GB/s\n", N * (double) K / 1e6, t_read8, N * (double) K / t_read8 / 1e3);
    fprintf(out, "read 2-bit weights (%.1f MB): %llu us = %.1f GB/s\n", N * (double) K / 4e6, t_read2, N * (double) K / 4 / t_read2 / 1e3);
    fprintf(out, "8-bit gemv: %llu us\n", t8);
    fprintf(out, "2-bit gemv (unpack + vrmpy): %llu us\n", t2);
    fprintf(out, "answers: %s (%d mismatches)\n", bad ? "WRONG" : "match", bad);
    fclose(out);
    free(w8); free(w2); free(x); free(y8); free(y2);
    return (int) (sink & 0) + (bad ? 1 : 0);
}
