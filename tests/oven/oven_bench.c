// Oven alone, one core: how many multiply-adds per second does each Q2_K x8 kernel do?
// Same shapes as a real E4B layer: 2560 inputs, 1024 output rows, a batch of 64 words (reading) or 1 word (talking).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint16_t d[8], dmin[8]; uint8_t scales[128]; uint8_t qs[512]; } block_q2_Kx8;
typedef struct { float d[4]; int8_t qs[1024]; int16_t bsums[64]; } block_q8_Kx4;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
typedef void (*kernel_t)(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x8_q8_K_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x8_q8_K_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_quantize_mat_q8_K_4x8(const float *, void *, int64_t);   // the sheet prep for the 2-bit path
void quantize_row_q8_K(const float *, void *, int64_t);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void bench(const char * name, kernel_t k, const void * vx, const void * vy, int n, int nr, int nc, int reps) {
    float * s = malloc(sizeof(float) * nr * nc);
    k(n, s, nc, vx, vy, nr, nc);                       // warm up
    double t0 = now();
    for (int r = 0; r < reps; r++) k(n, s, nc, vx, vy, nr, nc);
    double dt = (now() - t0) / reps;
    double macs = (double) n * nr * nc;
    printf("%-34s %8.2f ms per call   %6.1f billion multiply-adds/s\n", name, dt * 1e3, macs / dt / 1e9);
    free(s);
}

int main(void) {
    const int nb = 10, n = 256 * nb, nc = 1024, trays = nc / 8;
    block_q2_Kx8 * b = malloc(sizeof(block_q2_Kx8) * trays * nb);
    for (size_t i = 0; i < sizeof(block_q2_Kx8) * trays * nb; i++) ((uint8_t *) b)[i] = (uint8_t) rand();
    for (int i = 0; i < trays * nb; i++) for (int j = 0; j < 8; j++) { b[i].d[j] = 0x2000; b[i].dmin[j] = 0x2000; }

    const int nr = 64;
    block_q8_Kx4 * a4 = malloc(sizeof(block_q8_Kx4) * (nr / 4) * nb);
    for (size_t i = 0; i < sizeof(block_q8_Kx4) * (nr / 4) * nb; i++) ((uint8_t *) a4)[i] = (uint8_t) rand();
    for (int i = 0; i < (nr / 4) * nb; i++) for (int m = 0; m < 4; m++) a4[i].d[m] = 0.01f;
    block_q8_K * a1 = malloc(sizeof(block_q8_K) * nb);
    for (size_t i = 0; i < sizeof(block_q8_K) * nb; i++) ((uint8_t *) a1)[i] = (uint8_t) rand();
    for (int i = 0; i < nb; i++) a1[i].d = 0.01f;

    printf("one core, 2560 x 1024 layer\n");
    bench("reading (64 words), our oven", ggml_gemm_q2_K_8x8_q8_K, b, a4, n, nr, nc, 10);
    bench("reading (64 words), reference", ggml_gemm_q2_K_8x8_q8_K_generic, b, a4, n, nr, nc, 2);
    bench("reading (64 words), NEW 8x4 oven", ggml_gemm_q2_K_8x4_q8_K, b, a4, n, nr, nc, 10);
    bench("talking (1 word),   our oven", ggml_gemv_q2_K_8x8_q8_K, b, a1, n, 1, nc, 200);
    bench("talking (1 word),   NEW 8x4 oven", ggml_gemv_q2_K_8x4_q8_K, b, a1, n, 1, nc, 200);
    bench("talking (1 word),   reference", ggml_gemv_q2_K_8x8_q8_K_generic, b, a1, n, 1, nc, 20);

    // sheet prep: 64 words x 2560 numbers -> the interleaved 8-bit form the oven reads
    float * x = malloc(sizeof(float) * nr * n);
    for (int i = 0; i < nr * n; i++) x[i] = (float) rand() / RAND_MAX - 0.5f;
    double t0 = now();
    for (int r = 0; r < 20; r++)
        for (int g = 0; g < nr / 4; g++) ggml_quantize_mat_q8_K_4x8(x + (size_t) g * 4 * n, (block_q8_Kx4 *) a4 + g * nb, n);
    printf("%-34s %8.2f ms per call\n", "sheet prep (64 words)", (now() - t0) / 20 * 1e3);
    t0 = now();
    for (int r = 0; r < 200; r++) quantize_row_q8_K(x, a1, n);
    printf("%-34s %8.3f ms per call\n", "sheet prep (1 word)", (now() - t0) / 200 * 1e3);
    return 0;
}
