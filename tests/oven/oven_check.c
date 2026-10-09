// Head-to-head: our NEON Q2_K x8 oven vs llama.cpp's reference recipe, on identical random trays and words.
// Every output ("bill") is compared. Build: see run.sh next to this file.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint16_t d[8], dmin[8]; uint8_t scales[128]; uint8_t qs[512]; } block_q2_Kx8;   // 784 bytes
typedef struct { float d[4]; int8_t qs[1024]; int16_t bsums[64]; } block_q8_Kx4;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;

typedef void (*kernel_t)(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x8_q8_K_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x8_q8_K_generic(int, float *, size_t, const void *, const void *, int, int);

static uint16_t fp16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float rnd(void) { return (float) rand() / RAND_MAX; }

static void fill_trays(block_q2_Kx8 * b, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 8; j++) { b[i].d[j] = fp16(0.001f + 0.01f * rnd()); b[i].dmin[j] = fp16(0.001f + 0.01f * rnd()); }
        for (int j = 0; j < 128; j++) b[i].scales[j] = (uint8_t) rand();
        for (int j = 0; j < 512; j++) b[i].qs[j] = (uint8_t) rand();
    }
}

static double compare(const char * name, kernel_t ours, kernel_t ref, const void * vx, const void * vy, int n, int nr, int nc) {
    float * s1 = calloc((size_t) nr * nc, sizeof(float));
    float * s2 = calloc((size_t) nr * nc, sizeof(float));
    ours(n, s1, nc, vx, vy, nr, nc);
    ref(n, s2, nc, vx, vy, nr, nc);
    double worst = 0, scale = 0;
    for (int i = 0; i < nr * nc; i++) {
        worst = fmax(worst, fabs((double) s1[i] - s2[i]));
        scale = fmax(scale, fabs((double) s2[i]));
    }
    printf("%-5s %d bills | biggest bill %.4f | worst difference %.2e (%.1e of the biggest)\n", name, nr * nc, scale, worst, worst / scale);
    const int last = nr * nc - 1;  // stay inside the bills
    printf("      first/last bills, ours vs reference: %.5f/%.5f  %.5f/%.5f  %.5f/%.5f\n", s1[0], s2[0], s1[1], s2[1], s1[last], s2[last]);
    free(s1); free(s2);
    return worst / scale;
}

int main(void) {
    srand(42);
    const int nb = 10, n = 256 * nb;            // 10 super-blocks per row (a 2560-wide layer, like E4B)
    const int trays = 6, nc = 8 * trays;        // 48 output rows
    block_q2_Kx8 * b = malloc(sizeof(block_q2_Kx8) * trays * nb);
    fill_trays(b, trays * nb);

    // gemm: 8 words (two groups of 4), interleaved q8_Kx4
    const int nr = 8;
    block_q8_Kx4 * a4 = malloc(sizeof(block_q8_Kx4) * (nr / 4) * nb);
    for (int i = 0; i < (nr / 4) * nb; i++) {
        for (int m = 0; m < 4; m++) a4[i].d[m] = 0.01f + 0.05f * rnd();
        for (int j = 0; j < 1024; j++) a4[i].qs[j] = (int8_t) (rand() % 255 - 127);
        for (int j = 0; j < 64; j++) a4[i].bsums[j] = (int16_t) (rand() % 4000 - 2000);
    }
    // gemv: one word, plain q8_K
    block_q8_K * a1 = malloc(sizeof(block_q8_K) * nb);
    for (int i = 0; i < nb; i++) {
        a1[i].d = 0.01f + 0.05f * rnd();
        for (int j = 0; j < 256; j++) a1[i].qs[j] = (int8_t) (rand() % 255 - 127);
        for (int j = 0; j < 16; j++) a1[i].bsums[j] = (int16_t) (rand() % 4000 - 2000);
    }

    double e1 = compare("gemm", ggml_gemm_q2_K_8x8_q8_K, ggml_gemm_q2_K_8x8_q8_K_generic, b, a4, n, nr, nc);
    double e2 = compare("gemv", ggml_gemv_q2_K_8x8_q8_K, ggml_gemv_q2_K_8x8_q8_K_generic, b, a1, n, 1, nc);
    int ok = e1 < 1e-5 && e2 < 1e-5;
    printf("%s\n", ok ? "MATCH: our oven gives the reference bills (only rounding-order differences)" : "MISMATCH: our oven is wrong somewhere");
    return ok ? 0 : 1;
}
