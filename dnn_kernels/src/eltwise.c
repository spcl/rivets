
#include "eltwise.h"

#include <math.h>

#include "snrt.h"
#include "omp.h"
#include "dm.h"
#include "printf.h"

static double* volatile g_buf;

void eltwise_abs_fwd_fp32_baseline(float* dst, float* src, size_t n) {

    for (size_t i = 0; i < n; i++) {
        dst[i] = fabsf(src[i]);
    }

}

void eltwise_abs_fwd_fp64_baseline(double* dst, double* src, size_t n) {

    for (size_t i = 0; i < n; i++) {
        dst[i] = fabs(src[i]);
    }

}

void eltwise_abs_bwd_fp32_baseline(float* d_dst, float* d_src, float* src, size_t n) {

    for (size_t i = 0; i < n; i++) {
        d_src[i] = (src[i] > 0) ? d_dst[i] : -d_dst[i];
    }

}

void eltwise_abs_bwd_fp64_baseline(double* d_dst, double* d_src, double* src, size_t n) {

    for (size_t i = 0; i < n; i++) {
        d_src[i] = (src[i] > 0) ? d_dst[i] : -d_dst[i];
    }

}

void eltwise_clip_fwd_fp32_baseline(float* dst, float* src, size_t n, float alpha, float beta) {

    for (size_t i = 0; i < n; i++) {
        dst[i] = fmaxf(fminf(src[i], beta), alpha);
    }

}

void eltwise_clip_fwd_fp64_baseline(double* dst, double* src, size_t n, double alpha, double beta) {

    for (size_t i = 0; i < n; i++) {
        dst[i] = fmax(fmin(src[i], beta), alpha);
    }

}

void eltwise_clip_bwd_fp32_baseline(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta) {

    for (size_t i = 0; i < n; i++) {
        d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;
    }

}

void eltwise_clip_bwd_fp64_baseline(double* d_dst, double* d_src, double* src, size_t n, double alpha, double beta) {

    for (size_t i = 0; i < n; i++) {
        d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;
    }

}

void eltwise_abs_fwd_fp32_snitch_singlecore(float* dst, float* src, size_t n) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(2 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf0 = g_buf;
    double* buf1 = g_buf + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &dst[j - tcdm_buf_elems],
                    /* src */ buf1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 1, 0, 0;"
    "fabs.f ft1, ft0;"
    :: [reps] "r"(tcdm_buf_elems - 1)
    : "ft0", "ft1", "ft2", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf0;
        buf0 = buf1;
        buf1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &dst[j - elems_to_process],
            /* src */ buf1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_abs_fwd_fp64_snitch_singlecore(double* dst, double* src, size_t n) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(2 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf0 = g_buf;
    double* buf1 = g_buf + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &dst[j - tcdm_buf_elems],
                    /* src */ buf1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 1, 0, 0;"
    "fabs.d ft1, ft0;"
    :: [reps] "r"(tcdm_buf_elems - 1)
    : "ft0", "ft1", "ft2", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf0;
        buf0 = buf1;
        buf1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &dst[j - elems_to_process],
            /* src */ buf1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_abs_bwd_fp32_snitch_singlecore(float* d_dst, float* d_src, float* src, size_t n) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(4 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf_grad_0 = g_buf;
    double* buf_grad_1 = buf_grad_0 + tcdm_buf_elems;
    double* buf_src_0 = buf_grad_1 + tcdm_buf_elems;
    double* buf_src_1 = buf_src_0 + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, tcdm_buf_elems, sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM2, 1);
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf_src_0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ buf_grad_0,
            /* src */ &d_dst[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &d_src[j - tcdm_buf_elems],
                    /* src */ buf_grad_1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf_grad_1,
                    /* src */ &d_dst[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ buf_src_1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, buf_src_0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 1, 0, 0;"
    "fsgnjx.f ft1, ft0, ft2;"
    :: [reps] "r"(tcdm_buf_elems - 1)
    : "ft0", "ft1", "ft2", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf_grad_0;
        buf_grad_0 = buf_grad_1;
        buf_grad_1 = tmp_buf;

        tmp_buf = buf_src_0;
        buf_src_0 = buf_src_1;
        buf_src_1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &d_src[j - elems_to_process],
            /* src */ buf_grad_1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_abs_bwd_fp64_snitch_singlecore(double* d_dst, double* d_src, double* src, size_t n) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(4 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf_grad_0 = g_buf;
    double* buf_grad_1 = buf_grad_0 + tcdm_buf_elems;
    double* buf_src_0 = buf_grad_1 + tcdm_buf_elems;
    double* buf_src_1 = buf_src_0 + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, tcdm_buf_elems, sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM2, 1);
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf_src_0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ buf_grad_0,
            /* src */ &d_dst[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &d_src[j - tcdm_buf_elems],
                    /* src */ buf_grad_1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf_grad_1,
                    /* src */ &d_dst[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ buf_src_1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, buf_src_0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 1, 0, 0;"
    "fsgnjx.d ft1, ft0, ft2;"
    :: [reps] "r"(tcdm_buf_elems - 1)
    : "ft0", "ft1", "ft2", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf_grad_0;
        buf_grad_0 = buf_grad_1;
        buf_grad_1 = tmp_buf;

        tmp_buf = buf_src_0;
        buf_src_0 = buf_src_1;
        buf_src_1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &d_src[j - elems_to_process],
            /* src */ buf_grad_1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_clip_fwd_fp32_snitch_singlecore(float* dst, float* src, size_t n, float alpha, float beta) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(2 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf0 = g_buf;
    double* buf1 = g_buf + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &dst[j - tcdm_buf_elems],
                    /* src */ buf1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 2, 0, 0;"
    "fmin.f ft3, ft0, %[beta];"
    "fmax.f ft1, ft3, %[alpha];"
    :: [reps] "r"(tcdm_buf_elems - 1), [alpha] "f"(alpha), [beta] "f"(beta)
    : "ft0", "ft1", "ft2", "ft3", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf0;
        buf0 = buf1;
        buf1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &dst[j - elems_to_process],
            /* src */ buf1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_clip_fwd_fp64_snitch_singlecore(double* dst, double* src, size_t n, double alpha, double beta) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(2 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf0 = g_buf;
    double* buf1 = g_buf + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &dst[j - tcdm_buf_elems],
                    /* src */ buf1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 2, 0, 0;"
    "fmin.d ft3, ft0, %[beta];"
    "fmax.d ft1, ft3, %[alpha];"
    :: [reps] "r"(tcdm_buf_elems - 1), [alpha] "f"(alpha), [beta] "f"(beta)
    : "ft0", "ft1", "ft2", "ft3", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf0;
        buf0 = buf1;
        buf1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &dst[j - elems_to_process],
            /* src */ buf1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_clip_bwd_fp32_snitch_singlecore(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(4 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf_grad_0 = g_buf;
    double* buf_grad_1 = buf_grad_0 + tcdm_buf_elems;
    double* buf_src_0 = buf_grad_1 + tcdm_buf_elems;
    double* buf_src_1 = buf_src_0 + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, tcdm_buf_elems, sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM2, 2);
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf_src_0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ buf_grad_0,
            /* src */ &d_dst[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &d_src[j - tcdm_buf_elems],
                    /* src */ buf_grad_1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf_grad_1,
                    /* src */ &d_dst[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ buf_src_1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, buf_src_0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 5, 0, 0;"
    "fsub.s ft3,ft2,%[a];"
    "fsub.s ft4,%[b],ft2;"
    "fmul.s ft3,ft3,ft4;"
    "fsgnj.s ft3,%[two],ft3;"
    "fmadd.s ft1,ft0,ft3,%[minus_one];"
    :: [reps] "r"(tcdm_buf_elems - 1), [a] "f"(alpha), [b] "f"(beta), [two] "f"(2.), [minus_one] "f"(-1.)
    : "ft0", "ft1", "ft2", "ft3", "ft4", "t0", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf_grad_0;
        buf_grad_0 = buf_grad_1;
        buf_grad_1 = tmp_buf;

        tmp_buf = buf_src_0;
        buf_src_0 = buf_src_1;
        buf_src_1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &d_src[j - elems_to_process],
            /* src */ buf_grad_1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}

void eltwise_clip_bwd_fp64_snitch_singlecore(double* d_dst, double* d_src, double* src, size_t n, double alpha, double beta) {

    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

    if (tid == 0) {
        g_buf = (double*) snrt_l1alloc(4 * tcdm_buf_elems * sizeof(double));
        if (!g_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }
    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    double* buf_grad_0 = g_buf;
    double* buf_grad_1 = buf_grad_0 + tcdm_buf_elems;
    double* buf_src_0 = buf_grad_1 + tcdm_buf_elems;
    double* buf_src_1 = buf_src_0 + tcdm_buf_elems;

    if (tid == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, tcdm_buf_elems, sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM2, 2);
    }

    size_t j = 0;
    size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        snrt_dma_start_1d(
            /* dst */ buf_src_0,
            /* src */ &src[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ buf_grad_0,
            /* src */ &d_dst[j],
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    while (j < n) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        elems_to_process = last_elem - j;

        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (j != 0) {
                snrt_dma_start_1d(
                    /* dst */ &d_src[j - tcdm_buf_elems],
                    /* src */ buf_grad_1,
                    /* size */ tcdm_buf_elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (j + elems_to_process != n) {
                size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                snrt_dma_start_1d(
                    /* dst */ buf_grad_1,
                    /* src */ &d_dst[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ buf_src_1,
                    /* src */ &src[j + tcdm_buf_elems],
                    /* size */ elems * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (tid == 0) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf_grad_0);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, buf_src_0);

            snrt_ssr_enable();
            asm volatile(
                        "frep.o %0, 5, 0, 0;"
    "fsub.d ft3,ft2,%[a];"
    "fsub.d ft4,%[b],ft2;"
    "fmul.d ft3,ft3,ft4;"
    "fsgnj.d ft3,%[two],ft3;"
    "fmadd.d ft1,ft0,ft3,%[minus_one];"
    :: [reps] "r"(tcdm_buf_elems - 1), [a] "f"(alpha), [b] "f"(beta), [two] "f"(2.), [minus_one] "f"(-1.)
    : "ft0", "ft1", "ft2", "ft3", "ft4", "t0", "memory"
            );
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }

        // sync compute and data movement cores
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        // swap current and next buffers
        double* tmp_buf = buf_grad_0;
        buf_grad_0 = buf_grad_1;
        buf_grad_1 = tmp_buf;

        tmp_buf = buf_src_0;
        buf_src_0 = buf_src_1;
        buf_src_1 = tmp_buf;

        j += elems_to_process;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &d_src[j - elems_to_process],
            /* src */ buf_grad_1,
            /* size */ elems_to_process * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);


}
