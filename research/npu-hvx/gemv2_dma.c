// 2-bit matrix-vector on the NPU, version 7: weights stream from DDR into VTCM through the DMA engine (source
// bypassing the L2 cache), double-buffered per thread, instead of l2fetch through the cache. Qualcomm's DMA path
// kept ~13 GB/s next to CPU streaming (~28 GB/s combined) where the cache path shared one ~17 GB/s lane.
// Kernel math is version 5's (Q2_K-style 4-bit scale and min per 16-weight pouch, 32 rows per vector).
// DMA descriptor layout and dmlink/dmpoll follow llama.cpp's ggml-hexagon hex-dma.h (v68 descriptor format).
//
// Usage: run_main_on_hexagon 3 gemv2_dma.so            -> benchmark, 1 and 2 threads, with answer check
//        run_main_on_hexagon 3 gemv2_dma.so SECONDS    -> loop for SECONDS, GB/s every 2 s (for the both-drivers test)
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "qurt.h"

#define K 2560
#define N 10240
#define ROWS 32
#define BLOCK 16
#define BLOCKS (K / BLOCK)
#define SB 16
#define SBLOCKS (BLOCKS / SB)
#define GROUPS (N / ROWS)
#define MAX_THREADS 2
#define WBYTES (BLOCKS * 128)           // one group's packed weights
#define SBYTES (BLOCKS / 4 * 128)       // one group's packed scales
#ifndef CHUNK
#define CHUNK 4                         // groups fetched per DMA copy
#endif
#define SLOT (CHUNK * (WBYTES + SBYTES)) // one VTCM buffer: CHUNK groups of weights, then their scales

typedef struct dma_desc_s {             // 1D DMA descriptor, v68 layout (as llama.cpp's dma_descriptor_1d)
    void *next;
    uint32_t size : 24, desc_size : 2, dst_comp : 1, src_comp : 1, dst_bypass : 1, src_bypass : 1, order : 1, done : 1;
    void *src, *dst;
} __attribute__((aligned(64))) dma_desc;

static inline void dmlink(void *cur, void *next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}
static inline unsigned dmpoll(void) { unsigned r; asm volatile(" %0 = dmpoll" : "=r"(r) : : "memory"); return r; }

typedef struct {                        // a tiny per-thread ring of descriptors
    dma_desc d[8];
    dma_desc *tail;
    int push, pop;
} dma_ring;

static void ring_init(dma_ring *q) { memset(q, 0, sizeof *q); q->tail = &q->d[7]; }  // as llama.cpp: tail starts on an unused, zeroed descriptor

static void ring_push(dma_ring *q, void *dst, const void *src, uint32_t size) {
    dma_desc *d = &q->d[q->push];
    memset(d, 0, sizeof *d);
    d->src_bypass = 1;                  // read DDR without going through the L2 cache
    d->dst_bypass = 0;
    d->src = (void *) src;
    d->dst = dst;
    d->size = size;
    dmlink(q->tail, d);
    q->tail = d;
    q->push = (q->push + 1) & 7;
}

static void ring_wait(dma_ring *q) {    // wait for the oldest pushed copy
    dma_desc *d = &q->d[q->pop];
    while (!d->done) dmpoll();
    q->pop = (q->pop + 1) & 7;
}

static int power_status[2], vtcm_status;
static FILE *dbg;
#define WSTEP(msg) do { if (dbg && g == j->g0) { fprintf(dbg, "  worker g0=%d: %s\n", j->g0, msg); fflush(dbg); } } while (0)

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

static uint8_t *vtcm_get(unsigned *size) {  // as llama.cpp's vtcm_alloc: one single-page block of all VTCM
    *size = 8u << 20;
    HAP_compute_res_query_VTCM(0, size, NULL, NULL, NULL);
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 0);  // 0: VTCM is ours as soon as acquire returns (1 would need acquire_cached first)
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, *size, *size, *size);
    unsigned rctx = HAP_compute_res_acquire(&attr, 1000000);
    if (!rctx) { vtcm_status = -1; return NULL; }
    void *p = NULL;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &p, size) != 0) { vtcm_status = -2; return NULL; }
    return p;
}

static inline unsigned code(unsigned n, unsigned k) { return ((n * K + k) * 2654435761u) >> 30; }
static inline unsigned scale_of(unsigned n, unsigned pouch) { return ((n * BLOCKS + pouch) * 40503u >> 12) & 15; }
static inline unsigned min_of(unsigned n, unsigned pouch) { return ((n * BLOCKS + pouch) * 2246822519u >> 28) & 15; }

typedef struct {
    const uint8_t *w, *sc;              // packed weights and scales in DDR
    const int32_t *x4, *xsum;
    int32_t *S, *M;
    uint8_t *vtcm;                      // this thread's two slots
    int g0, g1;
} job_t;

static void compute_group(const HVX_Vector *wg, const HVX_Vector *sg, const int32_t *x4, const int32_t *xsum,
                          int32_t *S_out, int32_t *M_out) {
    const HVX_Vector three = Q6_Vb_vsplat_R(3), nib = Q6_V_vsplat_R(0x0f);
    const int32_t *xb = x4, *xs = xsum;
    for (int s = 0; s < SBLOCKS; s++) {
        HVX_Vector S = Q6_V_vzero(), M = Q6_V_vzero();
        for (int q4 = 0; q4 < SB / 4; q4++) {
            HVX_Vector sv = sg[s * (SB / 4) + q4];
            for (int i = 0; i < 4; i++, xb += 4, xs++) {
                HVX_Vector pk = wg[s * SB + q4 * 4 + i];
                HVX_Vector a = Q6_Vw_vrmpy_VubRb(Q6_V_vand_VV(pk, three), xb[0]);
                HVX_Vector b = Q6_Vw_vrmpy_VubRb(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 2), three), xb[1]);
                a = Q6_Vw_vrmpyacc_VwVubRb(a, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 4), three), xb[2]);
                b = Q6_Vw_vrmpyacc_VwVubRb(b, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(pk, 6), three), xb[3]);
                HVX_Vector pouch = Q6_Vw_vadd_VwVw(a, b);
                HVX_Vector byte = Q6_Vuw_vlsr_VuwR(sv, 8 * i);
                S = Q6_Vw_vadd_VwVw(S, Q6_Vw_vmpyie_VwVuh(pouch, Q6_V_vand_VV(byte, nib)));
                M = Q6_Vw_vadd_VwVw(M, Q6_Vw_vmpyi_VwRh(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(byte, 4), nib), Q6_R_combine_RlRl(*xs, *xs)));
            }
        }
        *(HVX_Vector *) (S_out + s * ROWS) = S;
        *(HVX_Vector *) (M_out + s * ROWS) = M;
    }
}

static void *worker(void *arg) {
    job_t *j = arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    dma_ring *qp = memalign(64, sizeof(dma_ring));  // descriptors in heap memory the DMA engine can read, not the stack
    if (!qp) { qurt_hvx_unlock(); return NULL; }
    dma_ring q_dummy; (void) q_dummy;
    ring_init(qp);
#define q (*qp)
    int g = j->g0;
    WSTEP("ring ready");
    uint8_t *slot[2] = {j->vtcm, j->vtcm + SLOT};
    int nchunks = (j->g1 - j->g0 + CHUNK - 1) / CHUNK;
#define PUSH(c, s) do { int g_ = j->g0 + (c) * CHUNK, n_ = j->g1 - g_ < CHUNK ? j->g1 - g_ : CHUNK; \
        ring_push(&q, slot[s], j->w + (size_t) g_ * WBYTES, n_ * WBYTES); \
        ring_push(&q, slot[s] + CHUNK * WBYTES, j->sc + (size_t) g_ * SBYTES, n_ * SBYTES); } while (0)
    PUSH(0, 0);
    WSTEP("first copies pushed");
    for (int c = 0; c < nchunks; c++) {
        int cur = c & 1;
        if (c + 1 < nchunks) PUSH(c + 1, !cur);
        ring_wait(&q);
        ring_wait(&q);
        int gs = j->g0 + c * CHUNK, ge = gs + CHUNK < j->g1 ? gs + CHUNK : j->g1;
        for (g = gs; g < ge; g++) {
            int k = g - gs;
            compute_group((const HVX_Vector *) (slot[cur] + (size_t) k * WBYTES),
                          (const HVX_Vector *) (slot[cur] + CHUNK * WBYTES + (size_t) k * SBYTES), j->x4, j->xsum,
                          j->S + (size_t) g * SBLOCKS * ROWS, j->M + (size_t) g * SBLOCKS * ROWS);
        }
    }
#undef q
    free(qp);
    qurt_hvx_unlock();
    return NULL;
}

static uint64_t run(int threads, job_t base) {
    pthread_t t[MAX_THREADS];
    job_t jobs[MAX_THREADS];
    uint64_t start = HAP_perf_get_time_us();
    for (int i = 0; i < threads; i++) {
        jobs[i] = base;
        jobs[i].vtcm = base.vtcm + (size_t) i * 2 * SLOT;
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
    FILE *out = fopen("gemv2_dma_result.txt", "w");
    if (!out) return 2;
#define STEP(msg) do { fprintf(out, "step: %s\n", msg); fflush(out); } while (0)
    STEP("started");
    dbg = out;
    power_up();
    STEP("powered up");
    unsigned vsize = 0;
    uint8_t *vtcm = vtcm_get(&vsize);
    STEP("vtcm requested");
    fprintf(out, "power requests: HVX %d, clocks %d; VTCM %u KB at %p (status %d)\n", power_status[0], power_status[1],
            vsize >> 10, vtcm, vtcm_status);
    if (!vtcm || vsize < MAX_THREADS * 2 * SLOT) { fclose(out); return 4; }
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
            for (int jj = 0; jj < ROWS; jj++)
                sc[((size_t) g * (BLOCKS / 4) + b / 4) * 128 + 4 * jj + b % 4] =
                    (uint8_t) (scale_of(g * ROWS + jj, b) | min_of(g * ROWS + jj, b) << 4);
        }
    // the DMA reads DDR bypassing the cache, so push the freshly packed weights out of the cache first
    qurt_mem_cache_clean((qurt_addr_t) w, wbytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t) sc, sbytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    STEP("weights packed");
    job_t base = {w, sc, x4, xsum, S, M, vtcm, 0, 0};
    double mb = (wbytes + sbytes) / 1e6;

    if (argc > 1) {                     // loop mode for the both-drivers test
        int seconds = atoi(argv[1]), passes = 0, wp = 0;
        uint64_t begin = HAP_perf_get_time_us(), end = begin + (uint64_t) seconds * 1000000, ws = begin, wt = 0, total = 0;
        while (HAP_perf_get_time_us() < end) {
            uint64_t t = run(MAX_THREADS, base);
            total += t; passes++; wp++; wt += t;
            if (HAP_perf_get_time_us() - ws >= 2000000) {
                fprintf(out, "t=%2llu s: %.1f GB/s\n", (HAP_perf_get_time_us() - begin) / 1000000, mb * wp / wt * 1e3);
                ws = HAP_perf_get_time_us(); wp = 0; wt = 0;
            }
        }
        fprintf(out, "%d passes in %d s: average %llu us (%.1f GB/s)\n", passes, seconds, total / passes, mb * passes / total * 1e3);
    } else {
        fprintf(out, "2-bit gate matrix %dx%d with scales (%.1f MB), version 7: DMA into VTCM, double-buffered, %d groups per copy\n", K, N, mb, CHUNK);
        STEP("first run starting");
        run(1, base);
        STEP("first run done");
        for (int threads = 1; threads <= MAX_THREADS; threads++) {
            uint64_t best = ~0ull;
            for (int r = 0; r < 3; r++) { uint64_t t = run(threads, base); if (t < best) best = t; }
            fprintf(out, "%d thread%s: %llu us (%.1f GB/s)\n", threads, threads > 1 ? "s" : " ", best, mb / best * 1e3);
        }
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
