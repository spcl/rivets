#include "matmul.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "snrt.h"
#include "omp.h"
#include "dm.h"

DTYPE ref_dst[P_B * P_M * P_N] = {0};
DTYPE dst[P_B * P_M * P_N] = {0};
DTYPE src[P_B * P_M * P_K] = {0};
DTYPE weight[P_B * P_K * P_N] = {0};
DTYPE bias[P_B * P_M * P_N] = {0};


int __attribute__((noinline)) main1() {
    printf("Execution started!\n");

    for (size_t i = 0; i < P_B * P_M * P_K; i++) {
        src[i] = (double)(i % 5) - 2;
    }
    for (size_t i = 0; i < P_B * P_K * P_N; i++) {
        weight[i] = (double)(i % 5) - 2;
    }
    for (size_t i = 0; i < P_B * P_M * P_N; i++) {
        bias[i] = (double)(i % 5) - 2;
    }

    unsigned long t1; asm volatile ("csrr %0, mcycle" : "=r"(t1));
    OP_IMPL(
        dst, src, weight, bias,
        P_B, P_M, P_K, P_N,
        P_M * P_N, P_N, 1, // stride dst
        P_M * P_K, P_K, 1, // stride src
        P_K * P_N, P_N, 1, // stride weight
        P_M * P_N, P_N, 1  // stride bias
    );
    unsigned long t2; asm volatile ("csrr %0, mcycle" : "=r"(t2));
    printf("Cycles: %lu (%lu%% of single-core peak)\n", t2 - t1, 100 * P_B * P_M * P_K * P_N / (t2 - t1));
    
    printf("Running reference implementation...\n");
    matmul_fp64(ref_dst, src, weight, bias,
        P_B, P_M, P_K, P_N,
        P_M * P_N, P_N, 1, // stride dst
        P_M * P_K, P_K, 1, // stride src
        P_K * P_N, P_N, 1, // stride weight
        P_M * P_N, P_N, 1  // stride bias
    );
    printf("Running reference implementation... Done\n");

    int err = 0;

    printf("Verifying result...\n");
    for (size_t i = 0; i < P_B * P_M * P_N; i++) {
        if (fabs(ref_dst[i] - dst[i]) > 0.0001) {
            printf("Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
            err = 1;
        }
    }
    printf("Verifying result... Done\n");
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