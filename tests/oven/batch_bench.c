// Batch bench, one core: checking 4 words at once (the drafter's 3 guesses plus the next word). For each matrix, the
// tray recipe (gemm: 4 words packed together) against the one-word recipe four times (gemv), for the 4-bit and
// 2-bit ovens, each with the packing its path needs. Shapes are brain C's 2-bit matrices.
// Build against ggml-cpu and ggml-base; run with taskset on one big core.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"

typedef void (*mm_t)(int, float *, size_t, const void *, const void *, int, int);
typedef void (*pack_t)(const float *, void *, int64_t);
void ggml_gemm_q4_0_4x4_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q4_0_4x4_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_quantize_mat_q8_0_4x4(const float *, void *, int64_t);
void ggml_quantize_mat_q8_K_4x4(const float *, void *, int64_t);
void ggml_quantize_mat_q8_K_4x4_generic(const float *, void *, int64_t);

static struct {
    const char * name; int blk, bytes, cols, row_bytes, x4_bytes;
    mm_t gemm, gemv; pack_t pack4, row; } T[] = {
    {"q4_0 4x4", 32, 18, 4, 34, 136, ggml_gemm_q4_0_4x4_q8_0, ggml_gemv_q4_0_4x4_q8_0, ggml_quantize_mat_q8_0_4x4, NULL},
    {"q2_K 8x4", 256, 84, 8, 292, 1168, ggml_gemm_q2_K_8x4_q8_K, ggml_gemv_q2_K_8x4_q8_K, ggml_quantize_mat_q8_K_4x4, NULL},
};
static const struct { const char * name; int k, n; } MATS[] = {{"ffn_gate", 2560, 10240}, {"attn_q", 2560, 2048}};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void * weights(int type, int k, int n) {   // random codes, sane fp16 scales so no NaN leaks into the timing
    size_t blocks = (size_t) n * (k / T[type].blk), size = blocks * T[type].bytes;
    uint8_t * w = malloc(size);
    for (size_t i = 0; i < size; i++) w[i] = (uint8_t) rand();
    uint16_t * h = (uint16_t *) w;
    if (type == 0) { for (size_t b = 0; b < blocks / 4; b++) for (int j = 0; j < 4; j++) h[b * 36 + j] = 0x2000; }
    else { size_t tray = (size_t) T[type].bytes * 8 / 2; for (size_t b = 0; b < blocks / 8; b++) for (int j = 0; j < 16; j++) h[b * tray + j] = 0x2000; }
    return w;
}

int main(void) {
    const int reps = 30;
    ggml_cpu_init();   // the one-word recipe packs each word with ggml's own row quantizer
    T[0].row = (pack_t) ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    T[1].row = (pack_t) ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    float * x = malloc(sizeof(float) * 4 * 10240);
    for (int i = 0; i < 4 * 10240; i++) x[i] = (float) rand() / RAND_MAX - 0.5f;
    printf("one core, 4 words, ms per matrix (packing included)\n");
    printf("%-10s %-10s %14s %14s %16s\n", "matrix", "oven", "tray x1 (4w)", "one-word x4", "packing gen->vec");
    for (int m = 0; m < 2; m++) {
        int k = MATS[m].k, n = MATS[m].n;
        for (int t = 0; t < 2; t++) {
            void * w = weights(t, k, n);
            float * s = malloc(sizeof(float) * 4 * n);
            uint8_t * a4 = malloc((size_t) (k / T[t].blk) * T[t].x4_bytes);
            uint8_t * a1 = malloc((size_t) 4 * (k / T[t].blk) * T[t].row_bytes);
            // tray: pack 4 words together, one gemm
            T[t].pack4(x, a4, k); T[t].gemm(k, s, n, w, a4, 4, n);
            double t0 = now();
            for (int r = 0; r < reps; r++) { T[t].pack4(x, a4, k); T[t].gemm(k, s, n, w, a4, 4, n); }
            double tray = (now() - t0) / reps;
            // one-word recipe: pack each word alone, four gemvs
            size_t rb = (size_t) (k / T[t].blk) * T[t].row_bytes;
            t0 = now();
            for (int r = 0; r < reps; r++)
                for (int i = 0; i < 4; i++) { T[t].row(x + i * k, a1 + i * rb, k); T[t].gemv(k, s + i * n, n, w, a1 + i * rb, 1, n); }
            double one = (now() - t0) / reps;
            char pk[32] = "";
            if (t == 1) {   // the q8_K 4x4 packing alone, generic vs vector
                t0 = now(); for (int r = 0; r < 200; r++) ggml_quantize_mat_q8_K_4x4_generic(x, a4, k); double pg = (now() - t0) / 200;
                t0 = now(); for (int r = 0; r < 200; r++) ggml_quantize_mat_q8_K_4x4(x, a4, k); double pv = (now() - t0) / 200;
                snprintf(pk, sizeof pk, "%.3f -> %.3f", pg * 1e3, pv * 1e3);
            }
            printf("%-10s %-10s %14.3f %14.3f %16s\n", MATS[m].name, T[t].name, tray * 1e3, one * 1e3, pk);
            free(w); free(s); free(a4); free(a1);
        }
    }
    return 0;
}
