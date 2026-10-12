// Send skip kernel, with the real arithmetic: is a recipe that reads only the open detectors fast enough to beat the
// oven's normal send step (ffn_down, 10240 detectors -> 2560 outputs, 4-bit) on the board?
//
//   today:  the oven's own Q4_0 4x4 gemv (dotprod), every row, the word packed to q8_0 first
//   skip:   the send stored detector by detector (each detector's 2560 outputs in Q4_0 blocks of 32); the open
//           detectors are taken 8 at a time, their contributions (activation x block scale) share one int8 scale per
//           block of 32 outputs, the 8 rows are interleaved and mixed in with the dot-product instruction, so each
//           pass over the outputs carries 8 detectors at once. Silent detectors are never read.
//
// First a correctness check (one layer, against an exact float reference), then timing over brain C's 42 layers
// at the silent shares of the per-layer gate (or every detector open, to compare the arithmetic alone).
// Build on the board:
//   gcc -O3 -march=armv8.2-a+dotprod+fp16 -pthread send_skip_kernel.c -I$HOME/llama-gpu/src/ggml/include \
//       -L$HOME/oven-ultra/lib -lggml-cpu -lggml-base -lm -o send_skip_kernel
#define _GNU_SOURCE
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define LAYERS 42
#define K 10240                       // detectors
#define N 2560                        // outputs
#define BLOCKS (N / 32)
#define ROW_BYTES (BLOCKS * 18)       // one detector's outputs in Q4_0
#define THREADS 4
#define WORDS 16

void ggml_gemv_q4_0_4x4_q8_0(int, float *, size_t, const void *, const void *, int, int);

// silent share per layer under the per-layer gate (-35 -50 -40 -50 -45 -30 -35 dB by blocks of 6 layers),
// measured on brain C (research/gate-assay/send_skip.py greedy); SILENT=x overrides with one share for all
static double gate_silent[LAYERS];
static const double BLOCK_SILENT[7] = {0.946, 0.367, 0.439, 0.175, 0.289, 0.888, 0.794};

// skip layout, one detector per row: its 80 block scales (fp16) first, then 80 x 16 bytes of 4-bit values
// (low nibble = output j, high nibble = output j + 16 of each block of 32), 1440 bytes like a Q4_0 row
#define SC(row) ((const __fp16 *) (row))
#define QS(row, b) ((const uint8_t *) (row) + 2 * BLOCKS + 16 * (b))

static uint8_t * send_rows[LAYERS];       // skip layout: [detector][scales | values]
static uint8_t * send_ggml[LAYERS];       // the oven's layout: Q4_0 4x4 interleaved
static float act[WORDS][K];
static uint32_t * open_ids[WORDS][LAYERS];
static int n_open[WORDS][LAYERS];
static float out[THREADS][N];
static int mode;                          // 0 today's gemv, 1 skip kernel
static atomic_int arrived, phase;
static volatile float sink;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void barrier(int * my_phase) {
    int p = ++*my_phase;
    if (atomic_fetch_add(&arrived, 1) == THREADS - 1) { atomic_store(&arrived, 0); atomic_store(&phase, p); }
    else while (atomic_load_explicit(&phase, memory_order_acquire) != p) { }
}

// y[N] += sum over the given detectors of a[i] * row_i, 8 detectors per pass.
// Per group: each detector's contribution per block (activation x block scale) is put on one int8 grid shared by the
// 8 detectors (vector math over all 80 blocks at once), the 8 grids are transposed to [block][detector], then per
// block the packed rows are interleaved 4 at a time (before unpacking, so one shuffle serves both nibbles) and mixed
// in with the dot-product instruction; the -8 offset of the 4-bit values comes off once per block at the end.
static void skip_accumulate(float * y, const uint8_t * rows, const uint32_t * ids, int n, const float * a) {
    float c[8][BLOCKS] __attribute__((aligned(16))), scale[BLOCKS] __attribute__((aligned(16)));
    int8_t x[8][BLOCKS] __attribute__((aligned(16)));
    uint8_t xt[BLOCKS][8] __attribute__((aligned(16)));
    int32_t corr[BLOCKS];
    for (int g = 0; g < n; g += 8) {
        const uint8_t * r[8]; float av[8];
        for (int k = 0; k < 8; k++) {
            r[k] = rows + (size_t) (g + k < n ? ids[g + k] : ids[g]) * ROW_BYTES;   // pad with a zero weight
            av[k] = g + k < n ? a[ids[g + k]] : 0.0f;
        }
        float32x4_t amax[BLOCKS / 4];
        for (int b4 = 0; b4 < BLOCKS / 4; b4++) amax[b4] = vdupq_n_f32(0);
        for (int k = 0; k < 8; k++)
            for (int b = 0; b < BLOCKS; b += 8) {
                float16x8_t h = vld1q_f16(SC(r[k]) + b);
                float32x4_t lo = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h)), av[k]);
                float32x4_t hi = vmulq_n_f32(vcvt_high_f32_f16(h), av[k]);
                vst1q_f32(c[k] + b, lo); vst1q_f32(c[k] + b + 4, hi);
                amax[b / 4] = vmaxq_f32(amax[b / 4], vabsq_f32(lo));
                amax[b / 4 + 1] = vmaxq_f32(amax[b / 4 + 1], vabsq_f32(hi));
            }
        float32x4_t inv[BLOCKS / 4];
        for (int b4 = 0; b4 < BLOCKS / 4; b4++) {
            uint32x4_t zero = vceqzq_f32(amax[b4]);
            vst1q_f32(scale + 4 * b4, vmulq_n_f32(amax[b4], 1.0f / 127));
            inv[b4] = vbslq_f32(zero, vdupq_n_f32(0), vdivq_f32(vdupq_n_f32(127), amax[b4]));
        }
        int16x8_t sum16[BLOCKS / 8];
        for (int b8 = 0; b8 < BLOCKS / 8; b8++) sum16[b8] = vdupq_n_s16(0);
        for (int k = 0; k < 8; k++)
            for (int b = 0; b < BLOCKS; b += 16) {
                int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(c[k] + b), inv[b / 4]));
                int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(c[k] + b + 4), inv[b / 4 + 1]));
                int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(c[k] + b + 8), inv[b / 4 + 2]));
                int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(c[k] + b + 12), inv[b / 4 + 3]));
                int16x8_t s0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1)), s1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                vst1q_s8(x[k] + b, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
                sum16[b / 8] = vaddq_s16(sum16[b / 8], s0); sum16[b / 8 + 1] = vaddq_s16(sum16[b / 8 + 1], s1);
            }
        for (int b8 = 0; b8 < BLOCKS / 8; b8++) {   // the -8 offset: 8 x (sum of the 8 grids) per block
            vst1q_s32(corr + 8 * b8, vshlq_n_s32(vmovl_s16(vget_low_s16(sum16[b8])), 3));
            vst1q_s32(corr + 8 * b8 + 4, vshlq_n_s32(vmovl_high_s16(sum16[b8]), 3));
        }
        for (int b = 0; b < BLOCKS; b += 16) {         // transpose the grids to [block][detector]
            int8x16_t v[8];
            for (int k = 0; k < 8; k++) v[k] = vld1q_s8(x[k] + b);
            int8x16_t t[8], u[8];
            for (int p = 0; p < 4; p++) { t[2 * p] = vzip1q_s8(v[2 * p], v[2 * p + 1]); t[2 * p + 1] = vzip2q_s8(v[2 * p], v[2 * p + 1]); }
            for (int p = 0; p < 2; p++) {
                u[4 * p + 0] = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(t[4 * p]), vreinterpretq_s16_s8(t[4 * p + 2])));
                u[4 * p + 1] = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(t[4 * p]), vreinterpretq_s16_s8(t[4 * p + 2])));
                u[4 * p + 2] = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(t[4 * p + 1]), vreinterpretq_s16_s8(t[4 * p + 3])));
                u[4 * p + 3] = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(t[4 * p + 1]), vreinterpretq_s16_s8(t[4 * p + 3])));
            }
            for (int q = 0; q < 4; q++) {   // u[q]: detectors 0-3 of blocks 4q'..; u[4+q]: detectors 4-7 of the same blocks
                int32x4_t lo = vzip1q_s32(vreinterpretq_s32_s8(u[q]), vreinterpretq_s32_s8(u[4 + q]));
                int32x4_t hi = vzip2q_s32(vreinterpretq_s32_s8(u[q]), vreinterpretq_s32_s8(u[4 + q]));
                vst1q_s32((int32_t *) xt[b + 4 * q], lo);
                vst1q_s32((int32_t *) xt[b + 4 * q + 2], hi);
            }
        }
        const uint8x16_t m4 = vdupq_n_u8(0x0F);
        for (int b = 0; b < BLOCKS; b++) {
            int32x4_t acc[8];
            for (int v = 0; v < 8; v++) acc[v] = vdupq_n_s32(0);
            for (int h = 0; h < 2; h++) {
                uint32_t packed; memcpy(&packed, xt[b] + 4 * h, 4);
                int8x16_t xs = vreinterpretq_s8_u32(vdupq_n_u32(packed));
                uint8x16_t q0 = vld1q_u8(QS(r[4 * h], b)), q1 = vld1q_u8(QS(r[4 * h + 1], b));
                uint8x16_t q2 = vld1q_u8(QS(r[4 * h + 2], b)), q3 = vld1q_u8(QS(r[4 * h + 3], b));
                uint8x16_t z01l = vzip1q_u8(q0, q1), z01h = vzip2q_u8(q0, q1), z23l = vzip1q_u8(q2, q3), z23h = vzip2q_u8(q2, q3);
                uint8x16_t w[4] = {
                    vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(z01l), vreinterpretq_u16_u8(z23l))),
                    vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(z01l), vreinterpretq_u16_u8(z23l))),
                    vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(z01h), vreinterpretq_u16_u8(z23h))),
                    vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(z01h), vreinterpretq_u16_u8(z23h)))};
                for (int i = 0; i < 4; i++) {
                    acc[i] = vdotq_s32(acc[i], vreinterpretq_s8_u8(vandq_u8(w[i], m4)), xs);       // outputs 4i..4i+3
                    acc[4 + i] = vdotq_s32(acc[4 + i], vreinterpretq_s8_u8(vshrq_n_u8(w[i], 4)), xs);   // outputs 16+4i..
                }
            }
            int32x4_t cv = vdupq_n_s32(corr[b]);
            float * yb = y + 32 * b;
            for (int v = 0; v < 8; v++) {
                float32x4_t yv = vld1q_f32(yb + 4 * v);
                vst1q_f32(yb + 4 * v, vfmaq_n_f32(yv, vcvtq_f32_s32(vsubq_s32(acc[v], cv)), scale[b]));
            }
        }
    }
}

static void * worker(void * arg) {
    int id = (int) (intptr_t) arg, my_phase = 0;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(4 + id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    static uint8_t q8[THREADS][K / 32 * 34];
    const ggml_from_float_t pack = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (int w = 0; w < WORDS; w++)
        for (int l = 0; l < LAYERS; l++) {
            if (mode == 0) {   // each thread computes its slice of the 2560 outputs from every row
                pack(act[w], q8[id], K);
                int c0 = N / THREADS * id;
                ggml_gemv_q4_0_4x4_q8_0(K, out[id] + c0, 0,
                                        send_ggml[l] + (size_t) c0 / 4 * (K / 32) * 72, q8[id], 1, N / THREADS);
            } else {           // each thread takes its share of the open detectors into its own copy of the outputs
                memset(out[id], 0, sizeof out[id]);
                int n = n_open[w][l], lo = n * id / THREADS / 8 * 8, hi = id == THREADS - 1 ? n : n * (id + 1) / THREADS / 8 * 8;
                skip_accumulate(out[id], send_rows[l], open_ids[w][l] + lo, hi - lo, act[w]);
            }
            barrier(&my_phase);
            if (mode == 1 && id == 0)   // sum the four copies (2560 adds per thread copy)
                for (int t = 1; t < THREADS; t++) for (int j = 0; j < N; j++) out[0][j] += out[t][j];
            if (mode == 1) barrier(&my_phase);
        }
    return NULL;
}

static double run(int m) {
    mode = m; atomic_store(&arrived, 0); atomic_store(&phase, 0);
    pthread_t t[THREADS];
    double t0 = now();
    for (int i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, worker, (void *) (intptr_t) i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    sink = out[0][1];
    return (now() - t0) / WORDS;
}

static void fill_layer(int l) {
    send_rows[l] = aligned_alloc(1 << 21, ((size_t) K * ROW_BYTES + (1 << 21)) & ~(size_t) ((1 << 21) - 1));
    send_ggml[l] = aligned_alloc(1 << 21, ((size_t) K * ROW_BYTES + (1 << 21)) & ~(size_t) ((1 << 21) - 1));
    for (size_t i = 0; i < (size_t) K * ROW_BYTES; i++) send_rows[l][i] = send_ggml[l][i] = (uint8_t) rand();
    for (int i = 0; i < K; i++)                       // sane scales in the skip layout
        for (int b = 0; b < BLOCKS; b++) ((__fp16 *) (send_rows[l] + (size_t) i * ROW_BYTES))[b] = (__fp16) (0.01f + 0.01f * (rand() % 8));
    for (size_t g = 0; g < (size_t) N / 4 * (K / 32); g++)   // and in the oven's layout (4 scales per 72-byte group)
        for (int j = 0; j < 4; j++) ((__fp16 *) (send_ggml[l] + g * 72))[j] = (__fp16) 0.01f;
}

static int check(void) {   // the skip kernel against an exact float reference, one layer, half the detectors open
    float * ref = calloc(N, sizeof(float)), * got = calloc(N, sizeof(float));
    uint32_t * ids = malloc(sizeof(uint32_t) * K); int n = 0;
    for (int i = 0; i < K; i++) if (rand() & 1) ids[n++] = i;
    for (int t = 0; t < n; t++) {
        const uint8_t * r = send_rows[0] + (size_t) ids[t] * ROW_BYTES;
        for (int b = 0; b < BLOCKS; b++) for (int j = 0; j < 32; j++) {
            int q = j < 16 ? (QS(r, b)[j] & 0x0F) : (QS(r, b)[j - 16] >> 4);
            ref[32 * b + j] += act[0][ids[t]] * (float) SC(r)[b] * (q - 8);
        }
    }
    skip_accumulate(got, send_rows[0], ids, n, act[0]);
    double err = 0, mag = 0;
    for (int j = 0; j < N; j++) { err += (got[j] - ref[j]) * (got[j] - ref[j]); mag += ref[j] * ref[j]; }
    double db = 10 * log10(err / mag);
    printf("correctness: skip kernel vs exact float, one layer, %d of %d detectors open: error %.1f dB  %s\n",
           n, K, db, db < -30 ? "ok" : "WRONG");
    return db < -30;
}

int main(void) {
    srand(1);
    ggml_cpu_init();
    for (int l = 0; l < LAYERS; l++) fill_layer(l);
    for (int w = 0; w < WORDS; w++) for (int i = 0; i < K; i++) act[w][i] = ((float) rand() / RAND_MAX - 0.5f) * 2;
    if (!check()) return 1;
    const char * fixed = getenv("SILENT") && *getenv("SILENT") ? getenv("SILENT") : NULL;
    long open_total = 0;
    for (int l = 0; l < LAYERS; l++) gate_silent[l] = fixed ? atof(fixed) : BLOCK_SILENT[l / 6];
    for (int w = 0; w < WORDS; w++)
        for (int l = 0; l < LAYERS; l++) {
            open_ids[w][l] = malloc(sizeof(uint32_t) * K);
            int n = 0;
            for (int i = 0; i < K; i++) if ((double) rand() / RAND_MAX >= gate_silent[l]) open_ids[w][l][n++] = i;
            n_open[w][l] = n; open_total += n;
        }
    printf("open detectors: %.1f%% (%s)\n", 100.0 * open_total / ((double) WORDS * LAYERS * K),
           fixed ? "one share for every layer" : "the per-layer gate's measured shares");
    run(0);   // warm-up
    double t_today = 1e9, t_skip = 1e9;
    for (int r = 0; r < 3; r++) { double t = run(0); if (t < t_today) t_today = t; t = run(1); if (t < t_skip) t_skip = t; }
    double mb = (double) LAYERS * K * ROW_BYTES / 1e6;
    printf("today's send (oven gemv, every row): %6.2f ms per word  %5.1f GB/s\n", t_today * 1e3, mb / 1e3 / t_today);
    printf("skip kernel (open detectors only):   %6.2f ms per word  %5.1f%% of today's time\n", t_skip * 1e3, 100 * t_skip / t_today);
    return 0;
}
