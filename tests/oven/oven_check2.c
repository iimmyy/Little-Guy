// Two-way check for the 4-byte-column Q2_K tray (8x4):
// 1. our NEON 8x4 kernels vs our 8x4 reference (same tray)
// 2. our 8x4 tray + kernels vs llama.cpp's ORIGINAL 8x8 tray + reference, both packed here from the same raw Q2_K
//    rows and fed the same words (quantized by llama.cpp's own 4x4 / 4x8 sheet-prep functions).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } block_q2_K;         // llama.cpp Q2_K
typedef struct { uint16_t d[8], dmin[8]; uint8_t scales[128]; uint8_t qs[512]; } block_q2_Kx8;
typedef struct { float d[4]; int8_t qs[1024]; int16_t bsums[64]; } block_q8_Kx4;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
typedef void (*kernel_t)(int, float *, size_t, const void *, const void *, int, int);
#define K(name) void name(int, float *, size_t, const void *, const void *, int, int);
K(ggml_gemm_q2_K_8x4_q8_K) K(ggml_gemm_q2_K_8x4_q8_K_generic) K(ggml_gemm_q2_K_8x8_q8_K_generic)
K(ggml_gemv_q2_K_8x4_q8_K) K(ggml_gemv_q2_K_8x4_q8_K_generic) K(ggml_gemv_q2_K_8x8_q8_K_generic)
void ggml_quantize_mat_q8_K_4x4(const float *, void *, int64_t);
void ggml_quantize_mat_q8_K_4x8(const float *, void *, int64_t);
void quantize_row_q8_K(const float *, void *, int64_t);

// same packing as repack.cpp make_block_q2_Kx8
static block_q2_Kx8 pack(const block_q2_K * in, int interleave) {
    block_q2_Kx8 out;
    for (int i = 0; i < 8; i++) { out.d[i] = in[i].d; out.dmin[i] = in[i].dmin; }
    for (int i = 0; i < 512 / interleave; i++) memcpy(&out.qs[i * interleave], &in[i % 8].qs[(i / 8) * interleave], interleave);
    for (int i = 0; i < 128; i++) out.scales[i] = in[(i % 16) / 2].scales[((i / 16) * 2) + (i % 2)];
    return out;
}

static uint16_t fp16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float frand(void) { return (float) (rand() % 100000) / 100000.0f; }

static double worst_rel(const float * a, const float * b, int count) {
    double w = 0, sc = 0;
    for (int i = 0; i < count; i++) { w = fmax(w, fabs((double) a[i] - b[i])); sc = fmax(sc, fabs((double) b[i])); }
    return w / sc;
}

int main(void) {
    srand(7);
    const int nb = 10, n = 256 * nb, trays = 6, nc = 8 * trays, nr = 8;
    block_q2_K * raw = malloc(sizeof(block_q2_K) * nc * nb);              // [row][super-block]
    for (int i = 0; i < nc * nb; i++) {
        for (int j = 0; j < 16; j++) raw[i].scales[j] = (uint8_t) rand();
        for (int j = 0; j < 64; j++) raw[i].qs[j] = (uint8_t) rand();
        raw[i].d = fp16(0.001f + 0.01f * frand()); raw[i].dmin = fp16(0.001f + 0.01f * frand());
    }
    block_q2_Kx8 * t4 = malloc(sizeof(block_q2_Kx8) * trays * nb), * t8 = malloc(sizeof(block_q2_Kx8) * trays * nb);
    for (int x = 0; x < trays; x++)
        for (int l = 0; l < nb; l++) {
            block_q2_K rows[8];
            for (int j = 0; j < 8; j++) rows[j] = raw[(x * 8 + j) * nb + l];
            t4[x * nb + l] = pack(rows, 4);
            t8[x * nb + l] = pack(rows, 8);
        }
    float * words = malloc(sizeof(float) * nr * n);
    for (int i = 0; i < nr * n; i++) words[i] = frand() * 2 - 1;
    block_q8_Kx4 * a44 = malloc(sizeof(block_q8_Kx4) * (nr / 4) * nb), * a48 = malloc(sizeof(block_q8_Kx4) * (nr / 4) * nb);
    for (int g = 0; g < nr / 4; g++) {
        ggml_quantize_mat_q8_K_4x4(words + (size_t) g * 4 * n, a44 + g * nb, n);
        ggml_quantize_mat_q8_K_4x8(words + (size_t) g * 4 * n, a48 + g * nb, n);
    }
    block_q8_K * a1 = malloc(sizeof(block_q8_K) * nb);
    quantize_row_q8_K(words, a1, n);

    float * fast = malloc(sizeof(float) * nr * nc), * ref4 = malloc(sizeof(float) * nr * nc), * ref8 = malloc(sizeof(float) * nr * nc);
    ggml_gemm_q2_K_8x4_q8_K(n, fast, nc, t4, a44, nr, nc);
    ggml_gemm_q2_K_8x4_q8_K_generic(n, ref4, nc, t4, a44, nr, nc);
    ggml_gemm_q2_K_8x8_q8_K_generic(n, ref8, nc, t8, a48, nr, nc);
    double g1 = worst_rel(fast, ref4, nr * nc), g2 = worst_rel(fast, ref8, nr * nc);
    printf("reading: ours vs our 8x4 reference %.1e | ours vs llama.cpp's 8x8 reference %.1e  (e.g. %.5f / %.5f / %.5f)\n",
           g1, g2, fast[13], ref4[13], ref8[13]);
    ggml_gemv_q2_K_8x4_q8_K(n, fast, nc, t4, a1, 1, nc);
    ggml_gemv_q2_K_8x4_q8_K_generic(n, ref4, nc, t4, a1, 1, nc);
    ggml_gemv_q2_K_8x8_q8_K_generic(n, ref8, nc, t8, a1, 1, nc);
    double v1 = worst_rel(fast, ref4, nc), v2 = worst_rel(fast, ref8, nc);
    printf("talking: ours vs our 8x4 reference %.1e | ours vs llama.cpp's 8x8 reference %.1e  (e.g. %.5f / %.5f / %.5f)\n",
           v1, v2, fast[13], ref4[13], ref8[13]);
    const int ok = g1 < 1e-5 && g2 < 1e-5 && v1 < 1e-5 && v2 < 1e-5;
    printf("%s\n", ok ? "MATCH: new tray + ovens agree with llama.cpp's original 2-bit math" : "MISMATCH");
    return ok ? 0 : 1;
}
