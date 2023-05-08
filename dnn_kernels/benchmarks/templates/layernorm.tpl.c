#include "layernorm.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "snrt.h"
#include "omp.h"
#include "dm.h"

DTYPE ref_dst[P_B * P_N] = {0};
DTYPE dst[P_B * P_N] = {0};
DTYPE src[P_B * P_N] = {0};
DTYPE ref_mu[P_B] = {0};
DTYPE mu[P_B] = {0};
DTYPE gamm[P_N] = {0};
DTYPE ref_sigma[P_B] = {0};
DTYPE sigma[P_B] = {0};
DTYPE beta[P_N] = {0};


int __attribute__((noinline)) main1() {
    printf("Execution started!\n");

    for (size_t i = 0; i < P_B * P_N; i++) {
        src[i] = (DTYPE)(i % 30) - 10;
    }
    for (size_t i = 0; i < P_N; i++) {
        gamm[i] = (DTYPE)(i % 30) - 10;
    }
    for (size_t i = 0; i < P_N; i++) {
        beta[i] = (DTYPE)(i % 30) - 10;
    }
    DTYPE eps = 1e-5;

    unsigned long t1; asm volatile ("csrr %0, mcycle" : "=r"(t1));
    OP_IMPL(
        dst, src, mu, gamm, sigma, beta, eps,
        P_B, P_N, P_N, 1
    );
    unsigned long t2; asm volatile ("csrr %0, mcycle" : "=r"(t2));
    printf("Cycles: %lu\n", t2 - t1);
    
    printf("Running reference implementation...\n");
    layer_norm_fp64(
        ref_dst, src, ref_mu, gamm, ref_sigma, beta, eps,
        P_B, P_N, P_N, 1
    );
    printf("Running reference implementation... Done\n");

    int err = 0;

    printf("Verifying result...\n");
    for (size_t i = 0; i < P_B * P_N; i++) {
        if (fabs(ref_dst[i] - dst[i]) > 1e-4) {
            printf("(dst) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
            err = 1;
        }
    }
    for (size_t i = 0; i < P_B; i++) {
        if (fabs(ref_mu[i] - mu[i]) > 1e-4) {
            printf("(mu) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
            err = 1;
        }
    }
    for (size_t i = 0; i < P_B; i++) {
        if (fabs(ref_sigma[i] - sigma[i]) > 1e-4) {
            printf("(sigma) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
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