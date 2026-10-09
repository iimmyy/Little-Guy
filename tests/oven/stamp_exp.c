// Elimination test: is the stamp loop itself slow because of HOW it feeds the dot-product unit?
// A = our oven today: copy 8 activation bytes through a general register and duplicate them (vdupq_n_s64).
// B = "point at answers": one 16-byte activation load, then vdotq_laneq_s32 picks a 4-byte group per stamp.
// Both run the same number of dot products over the same pre-decoded template. Timing only; B's sums are not
// the right answers for the current tray layout.
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

enum { NB = 10, TRAYS = 128, NR = 64 };       // 2560 inputs, 1024 output rows, 64 words
static int8_t tpl[TRAYS][NB][32][64];          // pre-decoded template: [tray][super-block][chunk*plane][4 row pairs x 16]
static int8_t act[NR / 4][NB][1024];           // q8_Kx4 activations
static volatile int32_t sink;

static void stamp_dup(void) {
    for (int x = 0; x < TRAYS; x++)
        for (int l = 0; l < NB; l++)
            for (int y = 0; y < NR / 4; y++) {
                int32x4_t isum[4][4];
                for (int m = 0; m < 4; m++) for (int p = 0; p < 4; p++) isum[m][p] = vdupq_n_s32(0);
                for (int kp = 0; kp < 32; kp++) {
                    const int k = kp >> 2, plane = kp & 3;
                    const int8_t * aq = act[y][l] + (k >> 2) * 512 + plane * 128 + (k & 3) * 32;
                    const int8_t * w  = tpl[x][l][kp];
                    const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
                    for (int m = 0; m < 4; m++) {
                        int64_t a8; memcpy(&a8, aq + m * 8, 8);
                        const int8x16_t av = vreinterpretq_s8_s64(vdupq_n_s64(a8));
                        isum[m][0] = vdotq_s32(isum[m][0], w0, av); isum[m][1] = vdotq_s32(isum[m][1], w1, av);
                        isum[m][2] = vdotq_s32(isum[m][2], w2, av); isum[m][3] = vdotq_s32(isum[m][3], w3, av);
                    }
                }
                for (int m = 0; m < 4; m++) sink += vaddvq_s32(vaddq_s32(vaddq_s32(isum[m][0], isum[m][1]), vaddq_s32(isum[m][2], isum[m][3])));
            }
}

static void stamp_lane(void) {
    for (int x = 0; x < TRAYS; x++)
        for (int l = 0; l < NB; l++)
            for (int y = 0; y < NR / 4; y++) {
                int32x4_t isum[4][4];
                for (int m = 0; m < 4; m++) for (int p = 0; p < 4; p++) isum[m][p] = vdupq_n_s32(0);
                for (int kp = 0; kp < 32; kp++) {
                    const int k = kp >> 2, plane = kp & 3;
                    const int8_t * aq = act[y][l] + (k >> 2) * 512 + plane * 128 + (k & 3) * 32;
                    const int8_t * w  = tpl[x][l][kp];
                    const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
                    for (int m = 0; m < 4; m++) {
                        const int8x16_t av = vld1q_s8(aq + (m & 1) * 16);   // 16 bytes = 4 groups of 4 answers
                        isum[m][0] = vdotq_laneq_s32(isum[m][0], w0, av, 0); isum[m][1] = vdotq_laneq_s32(isum[m][1], w1, av, 1);
                        isum[m][2] = vdotq_laneq_s32(isum[m][2], w2, av, 2); isum[m][3] = vdotq_laneq_s32(isum[m][3], w3, av, 3);
                    }
                }
                for (int m = 0; m < 4; m++) sink += vaddvq_s32(vaddq_s32(vaddq_s32(isum[m][0], isum[m][1]), vaddq_s32(isum[m][2], isum[m][3])));
            }
}

static void run(const char * name, void (*f)(void)) {
    f();
    const double t0 = now();
    for (int r = 0; r < 5; r++) f();
    const double dt = (now() - t0) / 5;
    printf("%-36s %7.2f ms per layer   %6.1f billion multiply-adds/s\n", name, dt * 1e3, 2560.0 * 1024 * NR / dt / 1e9);
}

int main(void) {
    for (size_t i = 0; i < sizeof(tpl); i++) ((int8_t *) tpl)[i] = (int8_t) (rand() % 46);
    for (size_t i = 0; i < sizeof(act); i++) ((int8_t *) act)[i] = (int8_t) (rand() % 255 - 127);
    printf("one core, stamping only (template already decoded), 2560 x 1024 layer, 64 words\n");
    run("A: copy-and-duplicate (our oven)", stamp_dup);
    run("B: point-at-answers (lane-indexed)", stamp_lane);
    return 0;
}
