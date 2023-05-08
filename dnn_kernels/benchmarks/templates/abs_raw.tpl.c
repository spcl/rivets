#include "elementwise.h"

#include <math.h>
#include "printf.h"
#include "snrt.h"

DTYPE gx[DATA_SIZE];
DTYPE gy[DATA_SIZE];

int main() {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = snrt_cluster_core_num();

    snrt_cluster_hw_barrier();
    DTYPE* x = gx;
    DTYPE* y = gy;

    if (tid == 0) {
        printf("Execution started!\n");
        for (int i = 0; i < DATA_SIZE; i++) {
            x[i] = (i % 2) ? i : -i;
        }
    }

    unsigned long t1; asm volatile ("csrr %0, mcycle" : "=r"(t1));
    OP_IMPL(y, x, DATA_SIZE);
    unsigned long t2; asm volatile ("csrr %0, mcycle" : "=r"(t2));
    
    if (tid == 0) {
        printf("Cycles: %lu\n", t2 - t1);
        printf("Verifying result...\n");
        int err = 0;
        for (int i = 0; i < DATA_SIZE; i++) {
            if (y[i] != i) {
                err = 1;
            }
        }
        printf("Verifying result... Done\n");
        printf("Err: %d\n", err);
        return err;
    }

    return 0;
}
