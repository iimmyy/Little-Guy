// Station bench, one core: where does the 2-bit brain lose reading time against the 4-bit brain?
// Times one E4B layer (q, k, v, o, gate, up, down) for each brain, split into stations, at batch 4
// (checking the drafter's guesses) and batch 64 (reading a sentence). Trays as picked on a dotprod-only
// CPU: 4-bit = q4_0 4x4 everywhere; 2-bit VIP = q2_K 8x4, with q4_K 8x4 for attn_output and ffn_down.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*gemm_t)(int, float *, size_t, const void *, const void *, int, int);
typedef void (*pack_t)(const float *, void *, int64_t);
void ggml_gemm_q4_0_4x4_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q4_K_8x4_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_quantize_mat_q8_0_4x4(const float *, void *, int64_t);
void ggml_quantize_mat_q8_K_4x4(const float *, void *, int64_t);

enum { Q4_0, Q2_K, Q4_K };
static const struct { const char * name; int blk, bytes, cols, scales; gemm_t gemm; } T[] = {
    [Q4_0] = {"q4_0 4x4", 32, 18, 4, 0, ggml_gemm_q4_0_4x4_q8_0},  // per weight block: 32 values in 18 bytes; trays of 4 columns
    [Q2_K] = {"q2_K 8x4", 256, 84, 8, 1, ggml_gemm_q2_K_8x4_q8_K},
    [Q4_K] = {"q4_K 8x4", 256, 144, 8, 1, ggml_gemm_q4_K_8x4_q8_K},
};
// the 2-bit type of each matrix, as in qat-Q2_K-vip4-f16proj.gguf (attn_k and attn_v exist in layers 0-23 only)
static const struct { const char * name; int k, n, t2; } MATS[] = {
    {"attn_q", 2560, 2048, Q2_K}, {"attn_k", 2560, 512, Q2_K}, {"attn_v", 2560, 512, Q4_K},
    {"attn_output", 2048, 2560, Q4_K}, {"ffn_gate", 2560, 10240, Q2_K}, {"ffn_up", 2560, 10240, Q2_K},
    {"ffn_down", 10240, 2560, Q4_K},
};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// random weights with sane fp16 scales (0x2000 ~ 0.008) so no NaN or inf leaks into the timing
static void * weights(int type, int k, int n) {
    size_t blocks = (size_t) n * (k / T[type].blk), size = blocks * T[type].bytes;
    uint8_t * w = malloc(size);
    for (size_t i = 0; i < size; i++) w[i] = (uint8_t) rand();
    uint16_t * h = (uint16_t *) w;
    if (type == Q4_0) {             // block_q4_0x4: d[4] then qs[64], 72 bytes per tray block
        for (size_t b = 0; b < blocks / 4; b++) for (int j = 0; j < 4; j++) h[b * 36 + j] = 0x2000;
    } else {                        // q2_Kx8 / q4_Kx8: d[8], dmin[8] first
        size_t tray = (size_t) T[type].bytes * 8 / 2;
        for (size_t b = 0; b < blocks / 8; b++) for (int j = 0; j < 16; j++) h[b * tray + j] = 0x2000;
    }
    return w;
}

static double time_gemm(int type, int k, int n, int batch, const void * act, int reps) {
    void * w = weights(type, k, n);
    float * s = malloc(sizeof(float) * batch * n);
    T[type].gemm(k, s, n, w, act, batch, n);
    double t0 = now();
    for (int r = 0; r < reps; r++) T[type].gemm(k, s, n, w, act, batch, n);
    double dt = (now() - t0) / reps;
    free(w); free(s);
    return dt;
}

static double time_pack(pack_t pack, int qk, int bytes_per_4rows, int k, int batch, void ** out, int reps) {
    float * x = malloc(sizeof(float) * batch * k);
    for (int i = 0; i < batch * k; i++) x[i] = (float) rand() / RAND_MAX - 0.5f;
    size_t per_group = (size_t) (k / qk) * bytes_per_4rows;
    uint8_t * a = malloc(per_group * (batch / 4));
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int g = 0; g < batch / 4; g++) pack(x + (size_t) g * 4 * k, a + g * per_group, k);
    double dt = (now() - t0) / reps;
    free(x);
    *out = a;
    return dt;
}

int main(void) {
    const int batches[] = {4, 64};
    for (int bi = 0; bi < 2; bi++) {
        int batch = batches[bi], reps = batch == 4 ? 20 : 4;
        double total[2] = {0, 0}, pack_t2[2] = {0, 0};
        printf("\n=== batch %d (%s), one core, one layer\n", batch, batch == 4 ? "checking 3 drafted words" : "reading a sentence");
        printf("%-12s %-18s %10s %-18s %10s\n", "matrix", "4-bit tray", "ms", "2-bit tray", "ms");
        for (size_t m = 0; m < sizeof(MATS) / sizeof(MATS[0]); m++) {
            int k = MATS[m].k, n = MATS[m].n;
            int t2 = MATS[m].t2;
            void *a0, *aK;
            double p0 = time_pack(ggml_quantize_mat_q8_0_4x4, 32, 136, k, batch, &a0, reps);
            double pK = time_pack(ggml_quantize_mat_q8_K_4x4, 256, 1168, k, batch, &aK, reps);
            double g0 = time_gemm(Q4_0, k, n, batch, a0, reps);
            double g2 = time_gemm(t2, k, n, batch, aK, reps);
            printf("%-12s %-18s %10.3f %-18s %10.3f\n", MATS[m].name, T[Q4_0].name, g0 * 1e3, T[t2].name, g2 * 1e3);
            total[0] += g0; total[1] += g2;
            // q, k, v share one packed input; gate and up share one; o and down each pack their own
            if (m != 1 && m != 2 && m != 5) { pack_t2[0] += p0; pack_t2[1] += pK; }
            free(a0); free(aK);
        }
        printf("%-12s %-18s %10.3f %-18s %10.3f\n", "packing", "q8_0 4x4", pack_t2[0] * 1e3, "q8_K 4x4", pack_t2[1] * 1e3);
        double w0 = total[0] + pack_t2[0], w2 = total[1] + pack_t2[1];
        printf("%-12s %-18s %10.3f %-18s %10.3f   (2-bit is %+.0f%%)\n", "LAYER TOTAL", "", w0 * 1e3, "", w2 * 1e3, 100 * (w2 / w0 - 1));
    }
    return 0;
}
