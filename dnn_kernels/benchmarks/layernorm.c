#include "layernorm.h"

#include "printf.h"
// #include <stdio.h>
#include <math.h>

#include "snrt.h"
// #include "omp.h"
// #include "dm.h"

#define P_B ARG0
#define P_N ARG1
#define OP_IMPL ARG2
#define DTYPE ARG3

DTYPE ref_dst[P_B * P_N] = {0};
DTYPE dst[P_B * P_N] = {0};
DTYPE src[P_B * P_N] = {0};
DTYPE ref_mu[P_B] = {0};
DTYPE mu[P_B] = {0};
DTYPE gamm[P_N] = {0};
DTYPE ref_sigma[P_B] = {0};
DTYPE sigma[P_B] = {0};
DTYPE beta[P_N] = {0};


int main() {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = snrt_cluster_core_num();

    DTYPE eps = 1e-5;

    snrt_cluster_hw_barrier();

    if (tid == 0) {
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
    }

    unsigned long t1; asm volatile ("csrr %0, mcycle" : "=r"(t1));
    OP_IMPL(
        dst, src, mu, gamm, sigma, beta, eps,
        P_B, P_N, P_N, 1
    );
    unsigned long t2; asm volatile ("csrr %0, mcycle" : "=r"(t2));
    if (tid == 0) {
        printf("Cycles: %lu\n", t2 - t1);
    }

    if (tid == 0) {
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
                //printf("(dst) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
                err = 1;
            }
        }
        // for (size_t i = 0; i < P_B; i++) {
        //     if (fabs(ref_mu[i] - mu[i]) > 1e-4) {
        //         //printf("(mu) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
        //         err = 1;
        //     }
        // }
        // for (size_t i = 0; i < P_B; i++) {
        //     if (fabs(ref_sigma[i] - sigma[i]) > 1e-4) {
        //         //printf("(sigma) Idx %d Ref: %f Res: %f Diff: %f\n", i, ref_dst[i], dst[i], fabs(ref_dst[i] - dst[i]));
        //         err = 1;
        //     }
        // }
        printf("Verifying result... Done\n");
        printf("Err: %d\n", err);

        return err;
    }
    return 0;
}
