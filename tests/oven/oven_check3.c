// Our 8x4 ovens vs llama.cpp's PRODUCTION no-tray path (ggml_vec_dot_q2_K_q8_K, one row x one word at a time),
// across the layer shapes Gemma 4 E4B really uses, starting from the same raw Q2_K rows and the same float words.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } block_q2_K;
typedef struct { uint16_t d[8], dmin[8]; uint8_t scales[128]; uint8_t qs[512]; } block_q2_Kx8;
typedef struct { float d[4]; int8_t qs[1024]; int16_t bsums[64]; } block_q8_Kx4;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
void ggml_gemm_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_quantize_mat_q8_K_4x4(const float *, void *, int64_t);
void quantize_row_q8_K(const float *, void *, int64_t);
void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static block_q2_Kx8 pack4(const block_q2_K * in) {
    block_q2_Kx8 out;
    for (int i = 0; i < 8; i++) { out.d[i] = in[i].d; out.dmin[i] = in[i].dmin; }
    for (int i = 0; i < 128; i++) memcpy(&out.qs[i * 4], &in[i % 8].qs[(i / 8) * 4], 4);
    for (int i = 0; i < 128; i++) out.scales[i] = in[(i % 16) / 2].scales[((i / 16) * 2) + (i % 2)];
    return out;
}
static uint16_t fp16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float frand(void) { return (float) (rand() % 100000) / 100000.0f; }

static int check_file(const char * path, int n, int nr, int nc);
static int check(int n, int nr, int nc) { return check_file(NULL, n, nr, nc); }
static int check_file(const char * path, int n, int nr, int nc) {
    const int nb = n / 256, trays = nc / 8;
    block_q2_K * raw = malloc(sizeof(block_q2_K) * nc * nb);
    if (path) { FILE * f = fopen(path, "rb"); fread(raw, sizeof(block_q2_K), (size_t) nc * nb, f); fclose(f); }
    else for (int i = 0; i < nc * nb; i++) {
        for (int j = 0; j < 16; j++) raw[i].scales[j] = (uint8_t) rand();
        for (int j = 0; j < 64; j++) raw[i].qs[j] = (uint8_t) rand();
        raw[i].d = fp16(0.0005f + 0.004f * frand()); raw[i].dmin = fp16(0.0005f + 0.004f * frand());
    }
    block_q2_Kx8 * t = malloc(sizeof(block_q2_Kx8) * trays * nb);
    for (int x = 0; x < trays; x++)
        for (int l = 0; l < nb; l++) { block_q2_K rows[8]; for (int j = 0; j < 8; j++) rows[j] = raw[(x * 8 + j) * nb + l]; t[x * nb + l] = pack4(rows); }
    float * words = malloc(sizeof(float) * nr * n);
    for (int i = 0; i < nr * n; i++) words[i] = (frand() * 2 - 1) * (1 + 3 * (i % 97 == 0));   // a few larger values, like real activations

    // production: each word quantized on its own, each row dotted on its own
    float * prod = malloc(sizeof(float) * nr * nc), * ours = malloc(sizeof(float) * nr * nc);
    block_q8_K * aw = malloc(sizeof(block_q8_K) * nb);
    for (int w = 0; w < nr; w++) {
        quantize_row_q8_K(words + (size_t) w * n, aw, n);
        for (int r = 0; r < nc; r++) ggml_vec_dot_q2_K_q8_K(n, &prod[w * nc + r], 0, raw + (size_t) r * nb, 0, aw, 0, 1);
    }
    // ours: like repack.cpp forward: groups of 4 words through gemm, leftovers through gemv
    const int n4 = nr - nr % 4;
    block_q8_Kx4 * a4 = malloc(sizeof(block_q8_Kx4) * (n4 / 4 + 1) * nb);
    for (int g = 0; g < n4 / 4; g++) ggml_quantize_mat_q8_K_4x4(words + (size_t) g * 4 * n, a4 + g * nb, n);
    if (n4) ggml_gemm_q2_K_8x4_q8_K(n, ours, nc, t, a4, n4, nc);
    for (int w = n4; w < nr; w++) { quantize_row_q8_K(words + (size_t) w * n, aw, n); ggml_gemv_q2_K_8x4_q8_K(n, ours + w * nc, nc, t, aw, 1, nc); }

    double worst = 0, scale = 0; int wi = 0;
    for (int i = 0; i < nr * nc; i++) { double d = fabs((double) ours[i] - prod[i]); if (d > worst) { worst = d; wi = i; } scale = fmax(scale, fabs(prod[i])); }
    if (path) printf("%s ", path);
    printf("n=%5d words=%2d rows=%3d | worst diff %.2e (%.1e of biggest) at word %d row %d: ours %.6f vs production %.6f\n",
           n, nr, nc, worst, worst / scale, wi / nc, wi % nc, ours[wi], prod[wi]);
    return worst / scale < 1e-5;
}

int main(void) {
    srand(11);
    int ok = 1;
    ok &= check(2560, 44, 64);
    ok &= check(2560, 7, 64);
    ok &= check(10240, 12, 32);
    ok &= check(10240, 1, 32);
    ok &= check(256, 4, 8);
    ok &= check_file("blk.0.ffn_gate.weight.bin", 2560, 44, 64);
    ok &= check_file("blk.0.attn_q.weight.bin", 2560, 13, 64);
    ok &= check_file("blk.0.ffn_down.weight.bin", 10240, 12, 32);
    printf("%s\n", ok ? "ALL MATCH production" : "MISMATCH vs production");
    return !ok;
}
