// DMA probe: which copies can this NPU's DMA engine do from our unsigned PD?
//   A: VTCM -> VTCM     B: heap (DDR) -> VTCM after a cache clean     C: same with the source not bypassing L2
// Each step is written to dma_probe.txt before it runs, so a crash shows which one died.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HAP_compute_res.h"
#include "qurt.h"

typedef struct dma_desc_s {
    void *next;
    unsigned size : 24, desc_size : 2, dst_comp : 1, src_comp : 1, dst_bypass : 1, src_bypass : 1, order : 1, done : 1;
    void *src, *dst;
} __attribute__((aligned(64))) dma_desc;

static inline void dmlink(void *cur, void *next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}
static inline unsigned dmpoll(void) { unsigned r; asm volatile(" %0 = dmpoll" : "=r"(r) : : "memory"); return r; }

static FILE *out;
static dma_desc *tail;

static int copy(const char *name, void *dst, const void *src, unsigned size, int src_bypass) {
    fprintf(out, "%s: starting\n", name); fflush(out);
    dma_desc *d = memalign(64, sizeof *d);
    memset(d, 0, sizeof *d);
    d->src = (void *) src; d->dst = dst; d->size = size; d->src_bypass = src_bypass;
    dmlink(tail, d);
    tail = d;
    unsigned spins = 0;
    while (!d->done && spins < 100000000) { dmpoll(); spins++; }
    int ok = d->done && memcmp(dst, src, size) == 0;
    fprintf(out, "%s: done=%u, data %s, spins %u\n", name, d->done, ok ? "matches" : "DIFFERS", spins); fflush(out);
    return ok;
}

int main(int argc, char *argv[]) {
    out = fopen("dma_probe.txt", "w");
    if (!out) return 2;
    unsigned size = 8u << 20;
    HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL);
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 0);  // 0: VTCM is ours as soon as acquire returns (1 would need acquire_cached first)
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size);
    unsigned rctx = HAP_compute_res_acquire(&attr, 1000000);
    void *vtcm = NULL;
    HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm, &size);
    fprintf(out, "VTCM %u KB at %p (rctx %u)\n", size >> 10, vtcm, rctx); fflush(out);
    if (!vtcm) return 3;
    tail = memalign(64, sizeof(dma_desc));
    memset(tail, 0, sizeof *tail);

    uint8_t *va = vtcm, *vb = (uint8_t *) vtcm + 65536;
    for (int i = 0; i < 4096; i++) va[i] = (uint8_t) (i * 7 + 1);
    copy("A vtcm->vtcm", vb, va, 4096, 0);

    uint8_t *heap = memalign(128, 65536);
    for (int i = 0; i < 65536; i++) heap[i] = (uint8_t) (i * 13 + 5);
    qurt_mem_cache_clean((qurt_addr_t) heap, 65536, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    copy("C heap->vtcm, through L2", vb, heap, 4096, 0);
    copy("B heap->vtcm, bypassing L2", vb, heap + 4096, 4096, 1);
    fprintf(out, "all done\n");
    fclose(out);
    return 0;
}
