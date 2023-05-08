#include "elementwise.h"

#include <stdlib.h>
#include <stdio.h>

#include "snrt.h"
#include "omp.h"
#include "dm.h"

DTYPE x[DATA_SIZE] = {0};
DTYPE y[DATA_SIZE] = {0};

int __attribute__((noinline)) main1() {
    for (int i = 0; i < DATA_SIZE; i++) {
        x[i] = (i % 2) ? i : -i;
    }

    unsigned long t1; asm volatile ("csrr %0, mcycle" : "=r"(t1));
    OP_IMPL(y, x, DATA_SIZE);
    unsigned long t2; asm volatile ("csrr %0, mcycle" : "=r"(t2));
    printf("Cycles: %lu\n", t2 - t1);

    int err = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (y[i] != i) {
            err = 1;
        }
    }

    printf("Err: %d\n", err);

    return err;
}

int main() {
    unsigned core_idx = snrt_cluster_core_idx();
    __snrt_omp_bootstrap(core_idx);
    // DM core doesn't have FPU.
    // This indirection is required to prevent touching
    // floating point items before __snrt_omp_bootstrap() call.
    int err = main1();
    __snrt_omp_destroy(core_idx);
    return err;
}