// Send skip bench: can the board read only the open detectors' rows of the FFN send (down projection) at full road
// speed, or does skipping around the matrix cost extra? Brain C's send at real size: 42 layers of 10240 detectors x
// 2560 outputs, stored detector by detector in 4-bit blocks (Q4_0: 18 bytes per 32 values, ~620 MB in all). Each
// word gets a fresh mute pattern per layer at the silent shares measured at -45 dB (research/gate-assay/send_skip.py).
// Three ways to read, same kernel: every row; only the open rows (scattered); the same number of rows in one sweep.
// The kernel only sums bytes (one dot instruction per 16 bytes) so the road, not the arithmetic, sets the time.
// Build on the board: gcc -O3 -march=armv8.2-a+dotprod -pthread send_skip_bench.c -o send_skip_bench
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LAYERS 42
#define DETECTORS 10240
#define ROW_BYTES (2560 / 32 * 18)   // one detector's 2560 outputs in Q4_0
#define THREADS 4
#define WORDS 64

static uint8_t * send[LAYERS];
static uint32_t * open_rows[WORDS][LAYERS];   // which detectors play, per word and layer
static int n_open[WORDS][LAYERS];
static int mode;                               // 0 every row, 1 open rows scattered, 2 same count in one sweep
static int pf_dist, pf_lines;                  // bookmarks: while reading an open row, prefetch the one pf_dist ahead
static uint32_t * bridged[WORDS][LAYERS];      // open rows plus short silent gaps read anyway (one longer sweep)
static int n_bridged[WORDS][LAYERS];
static atomic_int arrived, phase;
volatile uint32_t sink[THREADS];   // printed at the end, so the compiler can't drop the reads

// silent share of the send per layer at -45 dB, measured every 6th layer on brain C; filled in between
static const int MEASURED_AT[] = {0, 6, 12, 18, 24, 30, 36, 41};
static const double SILENT[] = {0.85, 0.77, 0.39, 0.26, 0.29, 0.33, 0.65, 0.46};

static double silent_share(int layer) {
    if (getenv("SILENT")) return atof(getenv("SILENT"));   // one share for every layer (efficiency curve)
    for (int i = 0; i < 7; i++)
        if (layer <= MEASURED_AT[i + 1]) {
            double t = (double) (layer - MEASURED_AT[i]) / (MEASURED_AT[i + 1] - MEASURED_AT[i]);
            return SILENT[i] + t * (SILENT[i + 1] - SILENT[i]);
        }
    return SILENT[7];
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void barrier(int * my_phase) {   // spin until all four threads arrive: one per layer, like the real graph
    int p = ++*my_phase;
    if (atomic_fetch_add(&arrived, 1) == THREADS - 1) { atomic_store(&arrived, 0); atomic_store(&phase, p); }
    else while (atomic_load_explicit(&phase, memory_order_acquire) != p) { }
}

static uint32_t read_row(const uint8_t * row) {
    uint32x4_t acc = vdupq_n_u32(0);
    const uint8x16_t ones = vdupq_n_u8(1);
    for (int i = 0; i + 16 <= ROW_BYTES; i += 16) acc = vdotq_u32(acc, vld1q_u8(row + i), ones);
    return vaddvq_u32(acc);
}

static void * worker(void * arg) {
    int id = (int) (intptr_t) arg, my_phase = 0;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(4 + id, &set);   // cpus 4-7: the four fast cores
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    uint32_t s = 0;
    for (int w = 0; w < WORDS; w++)
        for (int l = 0; l < LAYERS; l++) {
            const uint32_t * rows = mode == 3 ? bridged[w][l] : open_rows[w][l];
            int n = mode == 0 ? DETECTORS : mode == 3 ? n_bridged[w][l] : n_open[w][l];
            int lo = n * id / THREADS, hi = n * (id + 1) / THREADS;
            for (int i = lo; i < hi; i++) {
                int row = (mode == 1 || mode == 3) ? (int) rows[i] : i;
                if (pf_dist && i + pf_dist < hi) {
                    const uint8_t * ahead = send[l] + (size_t) rows[i + pf_dist] * ROW_BYTES;
                    for (int c = 0; c < pf_lines; c++) __builtin_prefetch(ahead + 64 * c, 0, 0);
                }
                s += read_row(send[l] + (size_t) row * ROW_BYTES);
            }
            barrier(&my_phase);
        }
    sink[id] = s;
    return NULL;
}

static double run(int m, int dist, int lines) {
    mode = m; pf_dist = dist; pf_lines = lines; atomic_store(&arrived, 0); atomic_store(&phase, 0);
    pthread_t t[THREADS];
    double t0 = now();
    for (int i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, worker, (void *) (intptr_t) i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    return (now() - t0) / WORDS;
}

int main(void) {
    srand(1);
    size_t layer_bytes = (size_t) DETECTORS * ROW_BYTES;
    for (int l = 0; l < LAYERS; l++) {
        send[l] = aligned_alloc(1 << 21, (layer_bytes + (1 << 21) - 1) & ~(size_t) ((1 << 21) - 1));
        for (size_t i = 0; i < layer_bytes; i++) send[l][i] = (uint8_t) rand();
    }
    long open_total = 0;
    for (int w = 0; w < WORDS; w++)
        for (int l = 0; l < LAYERS; l++) {
            open_rows[w][l] = malloc(sizeof(uint32_t) * DETECTORS);
            double silent = silent_share(l);
            int n = 0;
            for (int d = 0; d < DETECTORS; d++) if ((double) rand() / RAND_MAX >= silent) open_rows[w][l][n++] = d;
            n_open[w][l] = n; open_total += n;
        }
    int bridge = getenv("BRIDGE") ? atoi(getenv("BRIDGE")) : 1;   // read silent gaps of up to this many rows anyway
    long bridged_total = 0;
    for (int w = 0; w < WORDS; w++)
        for (int l = 0; l < LAYERS; l++) {
            bridged[w][l] = malloc(sizeof(uint32_t) * DETECTORS);
            int n = 0;
            for (int i = 0; i < n_open[w][l]; i++) {
                uint32_t r = open_rows[w][l][i];
                if (n && r - bridged[w][l][n - 1] - 1 <= (uint32_t) bridge)
                    for (uint32_t g = bridged[w][l][n - 1] + 1; g < r; g++) bridged[w][l][n++] = g;
                bridged[w][l][n++] = r;
            }
            n_bridged[w][l] = n; bridged_total += n;
        }
    double share = (double) open_total / ((double) WORDS * LAYERS * DETECTORS);
    double total_mb = (double) LAYERS * layer_bytes / 1e6;
    printf("send: %d layers, %.0f MB; open share at -45 dB: %.1f%% of rows\n", LAYERS, total_mb, 100 * share);
    run(0, 0, 0);   // warm-up: page in everything once
    double bshare = (double) bridged_total / ((double) WORDS * LAYERS * DETECTORS);
    struct { const char * label; int mode, dist, lines; } cfg[] = {
        {"every row", 0, 0, 0}, {"open rows, scattered", 1, 0, 0}, {"same count, one sweep", 2, 0, 0},
        {"scattered, bookmark 1 ahead", 1, 1, 23}, {"scattered, bookmark 2 ahead", 1, 2, 23},
        {"scattered, bookmark 4 ahead", 1, 4, 23}, {"scattered, bookmark 8 ahead", 1, 8, 23},
        {"scattered, bookmark 16 ahead", 1, 16, 23}, {"scattered, 4 ahead, first line", 1, 4, 1},
        {"scattered, 4 ahead, first 4", 1, 4, 4}, {"bridged gaps", 3, 0, 0}, {"bridged gaps, bookmark 4", 3, 4, 23},
    };
    double full = 0;
    for (size_t c = 0; c < sizeof cfg / sizeof cfg[0]; c++) {
        double best = 1e9;
        for (int r = 0; r < 3; r++) { double t = run(cfg[c].mode, cfg[c].dist, cfg[c].lines); if (t < best) best = t; }
        double frac = cfg[c].mode == 0 ? 1 : cfg[c].mode == 3 ? bshare : share;
        if (c == 0) full = best;
        printf("%-32s %7.2f ms per word  %5.1f%% of rows  %5.1f GB/s  %5.1f%% of the full read's time\n",
               cfg[c].label, best * 1e3, 100 * frac, total_mb * frac / 1e3 / best, 100 * best / full);
    }
    printf("(checksum %u)\n", sink[0] + sink[1] + sink[2] + sink[3]);
    return 0;
}
