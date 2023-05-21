#include "matmul.h"

// #define P_B 8
// #define P_M 64
// #define P_K 32
// #define P_N 32
// #define OP_IMPL matmul_raw_fp64_sdma_ssr_frep_omp
// #define DTYPE double

#define P_B ARG0
#define P_M ARG1
#define P_K ARG2
#define P_N ARG3
#define OP_IMPL ARG4
#define DTYPE ARG5

#include "printf.h"
// #include <stdio.h>
#include <math.h>

#include "snrt.h"
// #include "omp.h"
// #include "dm.h"

DTYPE ref_dst[P_B * P_M * P_N] = {0};
DTYPE dst[P_B * P_M * P_N] = {0};
DTYPE src[P_B * P_M * P_K] = {0};
DTYPE weight[P_B * P_K * P_N] = {0};
DTYPE bias[P_B * P_M * P_N] = {0};


int main() {
    // WARNING: don't print floating point values otherwise DM core will crash
    // even when it is not printing itself!
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = snrt_cluster_core_num();

    if (tid == 0) {
        printf("Execution started!\n");
    }

    if (tid == 0) {
        for (size_t i = 0; i < P_B * P_M * P_K; i++) {
            src[i] = (double)(i % 5) - 2;
        }
        for (size_t i = 0; i < P_B * P_K * P_N; i++) {
            weight[i] = (double)(i % 5) - 2;
        }
        for (size_t i = 0; i < P_B * P_M * P_N; i++) {
            bias[i] = (double)(i % 5) - 2;
        }
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

    if (tid == 0) {
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
                //printf("Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
                err = 1;
            }
        }
        printf("Verifying result... Done\n");
        printf("Err: %d\n", err);

        return err;
    }
    return 0;
}
