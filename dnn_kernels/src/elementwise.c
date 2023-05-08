#include "layernorm.h"

#include <math.h>

#include "snrt.h"
#include "omp.h"
#include "dm.h"
#include "printf.h"


// ==================================================================

void eltwise_abs(float* dst, float* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fabsf(src[i]);
    }
}

void eltwise_abs_fp64(double* dst, double* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fabs(src[i]);
    }
}

void eltwise_abs_sdma(float* dst, float* src, size_t n) {
    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(float);
    float* buf = (float*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        dm_memcpy_async(buf, &src[j], elems_to_process * sizeof(float));
        dm_wait();

        for (size_t i = 0; i < elems_to_process; i++) {
            buf[i] = fabsf(buf[i]);
        }

        dm_memcpy_async(&dst[j], buf, elems_to_process * sizeof(float));
        dm_wait();
    }
}

void eltwise_abs_fp64_sdma(double* dst, double* src, size_t n) {
    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(double);
    double* buf = (double*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        dm_memcpy_async(buf, &src[j], elems_to_process * sizeof(double));
        dm_wait();

        for (size_t i = 0; i < elems_to_process; i++) {
            buf[i] = fabs(buf[i]);
        }

        dm_memcpy_async(&dst[j], buf, elems_to_process * sizeof(double));
        dm_wait();
    }
}

void eltwise_abs_sdma_ssr(float* dst, float* src, size_t n) {
    #define DTYPE float
    //#define DTYPE double

    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(DTYPE);

    DTYPE* buf = (DTYPE*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        for (size_t i = 0; i < elems_to_process; i++) {
            buf[i] = src[i+j];
        }

        __builtin_ssr_setup_bound_stride_1d(0, elems_to_process - 1, sizeof(DTYPE));
        __builtin_ssr_read(0, 0, buf);
        __builtin_ssr_setup_bound_stride_1d(1, elems_to_process - 1, sizeof(DTYPE));
        __builtin_ssr_write(1, 0, buf);

        __builtin_ssr_enable();
        asm volatile("" ::: "memory");
        for (size_t i = 0; i < elems_to_process; i++) {
            asm volatile("fabs.s ft1, ft0" ::: "ft0", "ft1", "ft2");
            //asm volatile("fabs.d ft1, ft0" ::: "ft0", "ft1", "ft2");
        }
        asm volatile("" ::: "memory");
        __builtin_ssr_disable();

        for (size_t i = 0; i < elems_to_process; i++) {
            dst[i] = buf[i+j];
        }
    }
}


void eltwise_abs_fp64_sdma_ssr(double* dst, double* src, size_t n) {
    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(double);

    double* buf = (double*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        dm_memcpy_async(buf, &src[j], elems_to_process * sizeof(double));
        dm_wait();

        __builtin_ssr_setup_bound_stride_1d(0, elems_to_process - 1, sizeof(double));
        __builtin_ssr_read(0, 0, buf);
        __builtin_ssr_setup_bound_stride_1d(1, elems_to_process - 1, sizeof(double));
        __builtin_ssr_write(1, 0, buf);

        __builtin_ssr_enable();
        asm volatile("" ::: "memory");
        for (size_t i = 0; i < elems_to_process; i++) {
            asm volatile("fabs.d ft1, ft0" ::: "ft0", "ft1", "ft2");
        }
        asm volatile("" ::: "memory");
        __builtin_ssr_disable();

        dm_memcpy_async(&dst[j], buf, elems_to_process * sizeof(double));
        dm_wait();
    }
}


void eltwise_abs_fp64_sdma_ssr_frep(double* dst, double* src, size_t n) {
    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(double);

    double* buf = (double*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        dm_memcpy_async(buf, &src[j], elems_to_process * sizeof(double));
        dm_wait();

        __builtin_ssr_setup_bound_stride_1d(0, elems_to_process - 1, sizeof(double));
        __builtin_ssr_read(0, 0, buf);
        __builtin_ssr_setup_bound_stride_1d(1, elems_to_process - 1, sizeof(double));
        __builtin_ssr_write(1, 0, buf);

        __builtin_ssr_enable();
        asm volatile(
            "frep.o %0, 1, 0, 0;\n"
            "fabs.d ft1, ft0" 
            :: "r"(elems_to_process - 1)
            : "ft0", "ft1", "ft2", "memory"
        );
        __builtin_ssr_disable();

        dm_memcpy_async(&dst[j], buf, elems_to_process * sizeof(double));
        dm_wait();
    }
}

static double* volatile g_buf;

void eltwise_abs_raw_fp64_sdma_ssr_frep(double* dst, double* src, size_t n) {
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
                "fabs.d ft1, ft0"
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


void eltwise_abs_fp64_sdma_ssr_frep_omp(double* dst, double* src, size_t n) {
    size_t tcdm_buf_size = 64 * 1024;
    size_t tcdm_buf_elems = tcdm_buf_size / sizeof(double);

    double* buf = (double*) snrt_l1alloc(tcdm_buf_size);
    if (!buf) return;
    for (size_t j = 0; j < n; j += tcdm_buf_elems) {
        size_t last_elem = j + tcdm_buf_elems;
        if (last_elem > n) last_elem = n;
        size_t elems_to_process = last_elem - j;

        dm_memcpy_async(buf, &src[j], elems_to_process * sizeof(double));
        dm_wait();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = 8 /* omp_get_num_threads() */;

            int niters = elems_to_process / nthreads;
            int start_iter = tid * niters;
            int it_rem = elems_to_process % nthreads;
            if (tid < it_rem) {
                niters++;
                start_iter += tid;
            } else {
                start_iter += it_rem;
            }
            
            __builtin_ssr_setup_bound_stride_1d(0, niters - 1, sizeof(double));
            __builtin_ssr_read(0, 0, buf + start_iter);
            __builtin_ssr_setup_bound_stride_1d(1, niters - 1, sizeof(double));
            __builtin_ssr_write(1, 0, buf + start_iter);

            __builtin_ssr_enable();
            asm volatile(
                "frep.o %0, 1, 0, 0;\n"
                "fabs.d ft1, ft0" 
                :: "r"(niters - 1)
                : "ft0", "ft1", "ft2", "memory"
            );
            __builtin_ssr_disable();
        }

        dm_memcpy_async(&dst[j], buf, elems_to_process * sizeof(double));
        dm_wait();
    }
}


void eltwise_abs_raw_fp64_sdma_ssr_frep_omp(double* dst, double* src, size_t n) {
    size_t tcdm_buf_elems = 2000;

    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;

    size_t buf_per_thd = tcdm_buf_elems / ntd;

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
    
    if (snrt_is_compute_core()) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, buf_per_thd, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, buf_per_thd, sizeof(double));
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

        if (snrt_is_compute_core()) {

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &buf0[tid * buf_per_thd]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &buf0[tid * buf_per_thd]);

            snrt_ssr_enable();
            asm volatile(
                "frep.o %0, 1, 0, 0;"
                "fabs.d ft1, ft0"
                :: [reps] "r"(buf_per_thd - 1)
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



void eltwise_abs_bwd(float* d_dst, float* d_src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        d_src[i] = (d_dst[i] == 0) ? 0 : fabsf(d_dst[i]);
    }
}

// ==================================================================

void eltwise_clip(float* dst, float* src, size_t n, float alpha, float beta) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fmaxf(fminf(src[i], beta), alpha);
    }
}

void eltwise_clip_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta) {
    for (size_t i = 0; i < n; i++) {
        d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;
    }
}

// ==================================================================

void eltwise_elu(float* dst, float* src, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (src[i] > 0) ? src[i] : (alpha * (expf(src[i]) - 1));
    }
}

void eltwise_elu_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        d_src[i] = (src[i] > 0) ? d_dst[i] : (d_dst[i] * alpha * expf(src[i]));
    }
}

// ==================================================================


void eltwise_exp(float* dst, float* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = expf(src[i]);
    }
}

void eltwise_exp_bwd(float* d_dst, float* d_src, float* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        d_src[i] = d_dst[i] * expf(src[i]);
    }
}

// ==================================================================

void eltwise_gelu_erf(float* dst, float* src, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.5 * src[i] * (1 + erff(src[i] / sqrtf(2)));
    }
}

void eltwise_gelu_erf_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        d_src[i] = d_dst[i] * (0.5 + 0.5 * erff(src[i] / sqrtf(2)) + src[i] / sqrtf(2 * M_PI) * expf(-0.5 * src[i] * src[i]));
    }
}

// ==================================================================
