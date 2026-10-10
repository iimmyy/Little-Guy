// First contact: which way can a program on the NPU report back to the board?
#include <stdio.h>
#include "HAP_farf.h"
#include "HAP_perf.h"

int main(int argc, char *argv[]) {
    unsigned long long t = HAP_perf_get_time_us();
    printf("hello from the NPU via printf, t=%llu us\n", t);
    FARF(ALWAYS, "hello from the NPU via FARF, t=%llu us", t);
    FILE *f = fopen("npu_hello.txt", "w");
    if (f) { fprintf(f, "hello from the NPU via a file, t=%llu us\n", t); fclose(f); }
    return f ? 42 : 7;  // 42 = the file opened, 7 = it didn't
}
