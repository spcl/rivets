#include "matmul.h"

#include "printf.h"
#include "snrt.h"
#include "omp.h"
#include "dm.h"

#define C_(b, m, n) dst[(b) * stride_dst_b + (m) * stride_dst_m + (n) * stride_dst_n]
#define A_(b, m, k) src[(b) * stride_src_b + (m) * stride_src_m + (k) * stride_src_k]
#define B_(b, k, n) weight[(b) * stride_weight_b + (k) * stride_weight_k + (n) * stride_weight_n]
#define D_(b, m, n) bias[(b) * stride_bias_b + (m) * stride_bias_m + (n) * stride_bias_n]

void matmul(
    float* dst, float* src, float* weight, float* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    for (size_t b = 0; b < B; b++) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                C_(b, m, n) = D_(b, m, n);
                for (size_t k = 0; k < K; k++) {
                    C_(b, m, n) += A_(b, m, k) * B_(b, k, n);
                }
            }
        }
    }
}

void matmul_fp64(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    for (size_t b = 0; b < B; b++) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                C_(b, m, n) = D_(b, m, n);
                for (size_t k = 0; k < K; k++) {
                    C_(b, m, n) += A_(b, m, k) * B_(b, k, n);
                }
            }
        }
    }
}

#define AA_(b, m, k) src_buf[(b) * buf_size_m * buf_size_k + (m) * buf_size_k + (k)]
#define BB_(b, k, n) weight_buf[(b) * buf_size_k * buf_size_n + (k) * buf_size_n + (n)]
#define CC_(b, m, n) dst_buf[(b) * buf_size_m * buf_size_n + (m) * buf_size_n + (n)]

#define AA0_(b, m, k) src_buf0[(b) * buf_size_m * buf_size_k + (m) * buf_size_k + (k)]
#define BB0_(b, k, n) weight_buf0[(b) * buf_size_k * buf_size_n + (k) * buf_size_n + (n)]
#define CC0_(b, m, n) dst_buf0[(b) * buf_size_m * buf_size_n + (m) * buf_size_n + (n)]
#define AA1_(b, m, k) src_buf1[(b) * buf_size_m * buf_size_k + (m) * buf_size_k + (k)]
#define BB1_(b, k, n) weight_buf1[(b) * buf_size_k * buf_size_n + (k) * buf_size_n + (n)]
#define CC1_(b, m, n) dst_buf1[(b) * buf_size_m * buf_size_n + (m) * buf_size_n + (n)]

#define RC_(m, n) dst_reg_ ## m ## _ ## n

// WARNING: currently implementation is not flexible with supported strides
// dm_memcpy2d_async can't be efficiently used in all cases
void matmul_fp64_sdma(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    size_t buf_size_m = 2;
    size_t buf_size_n = 4;
    size_t buf_size_k = 16;

    double* src_buf = (double*) snrt_l1alloc(B * buf_size_m * buf_size_k * sizeof(double));
    double* weight_buf = (double*) snrt_l1alloc(B * buf_size_k * buf_size_n * sizeof(double));

    if (!src_buf || !weight_buf) {
        return;
    }

    for (size_t b = 0; b < B; b++) {

        for (size_t m1 = 0; m1 < M; m1 += buf_size_m) {
            size_t m_end = m1 + buf_size_m;
            if (m_end > M) m_end = M;

            for (size_t n1 = 0; n1 < N; n1 += buf_size_n) {
                size_t n_end = n1 + buf_size_n;
                if (n_end > N) n_end = N;

                double RC_(0, 0) = 0;
                double RC_(0, 1) = 0;
                double RC_(0, 2) = 0;
                double RC_(0, 3) = 0;
                double RC_(1, 0) = 0;
                double RC_(1, 1) = 0;
                double RC_(1, 2) = 0;
                double RC_(1, 3) = 0;

                for (size_t k1 = 0; k1 < K; k1 += buf_size_k) {
                    size_t k_end = k1 + buf_size_k;
                    if (k_end > K) k_end = K;

                    dm_memcpy2d_async(
                        /* src */ (uint64_t) &A_(b, m1, k1),
                        /* dst */ (uint64_t) &AA_(b, 0, 0),
                        /* size */ (k_end - k1) * sizeof(double),
                        /* sstride */ (&A_(0, 1, 0) - &A_(0, 0, 0)) * sizeof(double),
                        /* dstride */ (&AA_(0, 1, 0) - &AA_(0, 0, 0)) * sizeof(double),
                        /* repeats */ (m_end - m1),
                        /* cfg */ 0
                    );

                    dm_memcpy2d_async(
                        /* src */ (uint64_t) &B_(b, k1, n1),
                        /* dst */ (uint64_t) &BB_(b, 0, 0),
                        /* size */ (n_end - n1) * sizeof(double),
                        /* sstride */ (&B_(0, 1, 0) - &B_(0, 0, 0)) * sizeof(double),
                        /* dstride */ (&BB_(0, 1, 0) - &BB_(0, 0, 0)) * sizeof(double),
                        /* repeats */ (k_end - k1),
                        /* cfg */ 0
                    );
                    dm_wait();
                    
                    for (size_t k2 = 0; k2 < k_end - k1; k2++) {
                        RC_(0, 0) += AA_(b, 0, k2) * BB_(b, k2, 0);
                        RC_(0, 1) += AA_(b, 0, k2) * BB_(b, k2, 1);
                        RC_(0, 2) += AA_(b, 0, k2) * BB_(b, k2, 2);
                        RC_(0, 3) += AA_(b, 0, k2) * BB_(b, k2, 3);
                        RC_(1, 0) += AA_(b, 1, k2) * BB_(b, k2, 0);
                        RC_(1, 1) += AA_(b, 1, k2) * BB_(b, k2, 1);
                        RC_(1, 2) += AA_(b, 1, k2) * BB_(b, k2, 2);
                        RC_(1, 3) += AA_(b, 1, k2) * BB_(b, k2, 3);
                    }
                }

                if (m1+0 < m_end && n1+0 < n_end) C_(b, m1+0, n1+0) = RC_(0, 0) + D_(b, m1+0, n1+0);
                if (m1+0 < m_end && n1+1 < n_end) C_(b, m1+0, n1+1) = RC_(0, 1) + D_(b, m1+0, n1+1);
                if (m1+0 < m_end && n1+2 < n_end) C_(b, m1+0, n1+2) = RC_(0, 2) + D_(b, m1+0, n1+2);
                if (m1+0 < m_end && n1+3 < n_end) C_(b, m1+0, n1+3) = RC_(0, 3) + D_(b, m1+0, n1+3);
                if (m1+1 < m_end && n1+0 < n_end) C_(b, m1+1, n1+0) = RC_(1, 0) + D_(b, m1+1, n1+0);
                if (m1+1 < m_end && n1+1 < n_end) C_(b, m1+1, n1+1) = RC_(1, 1) + D_(b, m1+1, n1+1);
                if (m1+1 < m_end && n1+2 < n_end) C_(b, m1+1, n1+2) = RC_(1, 2) + D_(b, m1+1, n1+2);
                if (m1+1 < m_end && n1+3 < n_end) C_(b, m1+1, n1+3) = RC_(1, 3) + D_(b, m1+1, n1+3);
            }
        }
    }
}

void matmul_fp64_sdma_ssr(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    size_t buf_size_m = 2;
    size_t buf_size_n = 4;
    size_t buf_size_k = 16;

    double* src_buf = (double*) snrt_l1alloc(B * buf_size_m * buf_size_k * sizeof(double));
    double* weight_buf = (double*) snrt_l1alloc(B * buf_size_k * buf_size_n * sizeof(double));

    if (!src_buf || !weight_buf) {
        return;
    }

    for (size_t b = 0; b < B; b++) {

        for (size_t m1 = 0; m1 < M; m1 += buf_size_m) {
            size_t m_end = m1 + buf_size_m;
            if (m_end > M) m_end = M;

            for (size_t n1 = 0; n1 < N; n1 += buf_size_n) {
                size_t n_end = n1 + buf_size_n;
                if (n_end > N) n_end = N;

                double RC_(0, 0) = 0;
                double RC_(0, 1) = 0;
                double RC_(0, 2) = 0;
                double RC_(0, 3) = 0;
                double RC_(1, 0) = 0;
                double RC_(1, 1) = 0;
                double RC_(1, 2) = 0;
                double RC_(1, 3) = 0;

                for (size_t k1 = 0; k1 < K; k1 += buf_size_k) {
                    size_t k_end = k1 + buf_size_k;
                    if (k_end > K) k_end = K;

                    dm_memcpy2d_async(
                        /* src */ (uint64_t) &A_(b, m1, k1),
                        /* dst */ (uint64_t) &AA_(b, 0, 0),
                        /* size */ (k_end - k1) * sizeof(double),
                        /* sstride */ (&A_(0, 1, 0) - &A_(0, 0, 0)) * sizeof(double),
                        /* dstride */ (&AA_(0, 1, 0) - &AA_(0, 0, 0)) * sizeof(double),
                        /* repeats */ (m_end - m1),
                        /* cfg */ 0
                    );

                    dm_memcpy2d_async(
                        /* src */ (uint64_t) &B_(b, k1, n1),
                        /* dst */ (uint64_t) &BB_(b, 0, 0),
                        /* size */ (n_end - n1) * sizeof(double),
                        /* sstride */ (&B_(0, 1, 0) - &B_(0, 0, 0)) * sizeof(double),
                        /* dstride */ (&BB_(0, 1, 0) - &BB_(0, 0, 0)) * sizeof(double),
                        /* repeats */ (k_end - k1),
                        /* cfg */ 0
                    );
                    dm_wait();

                    snrt_ssr_repeat(SNRT_SSR_DM0, buf_size_n);
                    snrt_ssr_loop_1d(SNRT_SSR_DM0, k_end - k1, sizeof(double));
                    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &AA_(b, 0, 0));

                    snrt_ssr_repeat(SNRT_SSR_DM1, buf_size_n);
                    snrt_ssr_loop_1d(SNRT_SSR_DM1, k_end - k1, sizeof(double));
                    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, &AA_(b, 1, 0));

                    snrt_ssr_repeat(SNRT_SSR_DM2, buf_size_m);
                    snrt_ssr_loop_1d(SNRT_SSR_DM2, (k_end - k1) * buf_size_n, sizeof(double));
                    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &BB_(b, 0, 0));

                    __builtin_ssr_enable();
                    asm volatile("" ::: "memory");
                    for (size_t k2 = 0; k2 < k_end - k1; k2++) {
                        asm volatile("fmadd.d %0, ft0, ft2, %0" : "+f"(RC_(0, 0)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft1, ft2, %0" : "+f"(RC_(1, 0)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft0, ft2, %0" : "+f"(RC_(0, 1)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft1, ft2, %0" : "+f"(RC_(1, 1)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft0, ft2, %0" : "+f"(RC_(0, 2)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft1, ft2, %0" : "+f"(RC_(1, 2)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft0, ft2, %0" : "+f"(RC_(0, 3)) :: "ft0", "ft1", "ft2");
                        asm volatile("fmadd.d %0, ft1, ft2, %0" : "+f"(RC_(1, 3)) :: "ft0", "ft1", "ft2");
                    }
                    asm volatile("" ::: "memory");
                    __builtin_ssr_disable();
                }

                if (m1+0 < m_end && n1+0 < n_end) C_(b, m1+0, n1+0) = RC_(0, 0) + D_(b, m1+0, n1+0);
                if (m1+0 < m_end && n1+1 < n_end) C_(b, m1+0, n1+1) = RC_(0, 1) + D_(b, m1+0, n1+1);
                if (m1+0 < m_end && n1+2 < n_end) C_(b, m1+0, n1+2) = RC_(0, 2) + D_(b, m1+0, n1+2);
                if (m1+0 < m_end && n1+3 < n_end) C_(b, m1+0, n1+3) = RC_(0, 3) + D_(b, m1+0, n1+3);
                if (m1+1 < m_end && n1+0 < n_end) C_(b, m1+1, n1+0) = RC_(1, 0) + D_(b, m1+1, n1+0);
                if (m1+1 < m_end && n1+1 < n_end) C_(b, m1+1, n1+1) = RC_(1, 1) + D_(b, m1+1, n1+1);
                if (m1+1 < m_end && n1+2 < n_end) C_(b, m1+1, n1+2) = RC_(1, 2) + D_(b, m1+1, n1+2);
                if (m1+1 < m_end && n1+3 < n_end) C_(b, m1+1, n1+3) = RC_(1, 3) + D_(b, m1+1, n1+3);
            }
        }
    }
}

void gemm_fp64_opt(uint32_t M, uint32_t N, uint32_t K, double* A, uint32_t ldA,
                   uint32_t ta, double* B, uint32_t ldB, uint32_t tb, double* C,
                   uint32_t ldC, const uint32_t* ALPHA, uint32_t setup_SSR) {
    // Unrolling factor of most inner loop.
    // Should be at least as high as the FMA delay
    // for maximum utilization
    const uint32_t unroll = 8;

    // SSR strides and bounds only have to be configured
    // once in the beginning
    if (setup_SSR) {
        // First matrix is stored in transposed format
        if (ta) {
            const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
            const uint32_t ssr0_i[4] = {0, 8 * ldA, 0, 8 * 8};

            snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                             ssr0_i[1], ssr0_i[2], ssr0_i[3]);
            snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
        } else {
            const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
            const uint32_t ssr0_i[4] = {0, 8, 0, 8 * ldA};

            snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                             ssr0_i[1], ssr0_i[2], ssr0_i[3]);
            snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
        }

        // Second matrix is stored in transposed format
        if (tb) {
            const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
            const uint32_t ssr1_i[4] = {8 * ldB, 8, 8 * ldB * unroll, 0};

            snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                             ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                             ssr1_i[3]);
        } else {
            const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
            const uint32_t ssr1_i[4] = {8, 8 * ldB, 8 * unroll, 0};

            snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                             ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                             ssr1_i[3]);
        }
    }

    // SSR start address need to be configured each time
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
    snrt_ssr_enable();

    for (uint32_t m = 0; m < M; m++) {
        uint32_t n = 0;
        for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
            double c[unroll];

            // Load intermediate result
            if (*ALPHA) {
                c[0] = C[m * ldC + n + 0];
                c[1] = C[m * ldC + n + 1];
                c[2] = C[m * ldC + n + 2];
                c[3] = C[m * ldC + n + 3];
                c[4] = C[m * ldC + n + 4];
                c[5] = C[m * ldC + n + 5];
                c[6] = C[m * ldC + n + 6];
                c[7] = C[m * ldC + n + 7];
            } else {
                c[0] = 0.0;
                c[1] = 0.0;
                c[2] = 0.0;
                c[3] = 0.0;
                c[4] = 0.0;
                c[5] = 0.0;
                c[6] = 0.0;
                c[7] = 0.0;
            }

            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                "fmadd.d %[c0], ft0, ft1, %[c0] \n"
                "fmadd.d %[c1], ft0, ft1, %[c1] \n"
                "fmadd.d %[c2], ft0, ft1, %[c2] \n"
                "fmadd.d %[c3], ft0, ft1, %[c3] \n"
                "fmadd.d %[c4], ft0, ft1, %[c4] \n"
                "fmadd.d %[c5], ft0, ft1, %[c5] \n"
                "fmadd.d %[c6], ft0, ft1, %[c6] \n"
                "fmadd.d %[c7], ft0, ft1, %[c7] \n"
                : [ c0 ] "+f"(c[0]), [ c1 ] "+f"(c[1]), [ c2 ] "+f"(c[2]),
                  [ c3 ] "+f"(c[3]), [ c4 ] "+f"(c[4]), [ c5 ] "+f"(c[5]),
                  [ c6 ] "+f"(c[6]), [ c7 ] "+f"(c[7])
                : [ n_frep ] "r"(K - 1)
                : "ft0", "ft1", "ft2");

            // Store results back
            C[m * ldC + n + 0] = c[0];
            C[m * ldC + n + 1] = c[1];
            C[m * ldC + n + 2] = c[2];
            C[m * ldC + n + 3] = c[3];
            C[m * ldC + n + 4] = c[4];
            C[m * ldC + n + 5] = c[5];
            C[m * ldC + n + 6] = c[6];
            C[m * ldC + n + 7] = c[7];
            n += unroll;
        }

        // Clean up of leftover columns
        snrt_ssr_disable();

        for (; n < N; n++) {
            double c;
            if (*ALPHA) {
                c = C[m * ldC + n];
            } else {
                c = 0.0;
            }
            for (uint32_t k = 0; k < K; k++) {
                c += A[k + m * ldA] * B[k + n * ldB];
            }
            C[m * ldC + n] = c;
        }

        snrt_ssr_enable();
    }

    snrt_ssr_disable();
}

void matmul_fp64_sdma_ssr_frep(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    size_t buf_size_m = M;
    size_t buf_size_k = K;
    size_t buf_size_n = N;

    size_t scratchpad_max_size = (64 * 1024 / sizeof(double));
    size_t buf_size_b = scratchpad_max_size / (M * K + K * N + M * N);
    size_t buf_size_b_pow2 = 1;
    while(buf_size_b_pow2 < buf_size_b) buf_size_b_pow2 *= 2;
    if (buf_size_b_pow2 > 1 && buf_size_b_pow2 > buf_size_b) buf_size_b_pow2 /= 2;
    buf_size_b = (B < buf_size_b_pow2) ? B : buf_size_b_pow2;

    double* src_buf = (double*) snrt_l1alloc(buf_size_b * (M * K + K * N + M * N) * sizeof(double));
    double* weight_buf = src_buf + buf_size_b * M * K;
    double* dst_buf = weight_buf + buf_size_b * K * N;

    if (!src_buf) {
        printf("Error: failed to allocate scratchpad memory\n");
        while (1) {}
        return;
    }

    unsigned long t1t, t1p, t1c, t1e;
    unsigned long t2t, t2p, t2c, t2e;

    // asm volatile ("csrr %0, mcycle" : "=r"(t1t));

    // asm volatile ("csrr %0, mcycle" : "=r"(t1p));

    for (size_t b1 = 0; b1 < B; b1 += buf_size_b) {
        dm_memcpy_async(
            /* dst */ &AA_(0, 0, 0),
            /* src */ &A_(b1, 0, 0),
            /* size */ buf_size_b * M * K * sizeof(double)
        );
        dm_memcpy_async(
            /* dst */ &BB_(0, 0, 0),
            /* src */ &B_(b1, 0, 0),
            /* size */ buf_size_b * K * N * sizeof(double)
        );
        dm_memcpy_async(
            /* dst */ &CC_(0, 0, 0),
            /* src */ &D_(b1, 0, 0),
            /* size */ buf_size_b * M * N * sizeof(double)
        );
        dm_wait();
        // asm volatile ("csrr %0, mcycle" : "=r"(t2p));

        // asm volatile ("csrr %0, mcycle" : "=r"(t1c));
        for (size_t b2 = 0; b2 < buf_size_b; b2++) {    
            uint32_t ALPHA = 1;
            gemm_fp64_opt(
                /*uint32_t M, uint32_t N, uint32_t K, */ M, N, K,
                /*double* A, uint32_t ldA, uint32_t ta, */ &AA_(b2, 0, 0), K, 0,
                /*double* B, uint32_t ldB, int32_t tb, */ &BB_(b2, 0, 0), N, 0,
                /*double* C, uint32_t ldC,*/ &CC_(b2, 0, 0), N,
                /*const uint32_t* ALPHA,*/ &ALPHA,
                /*uint32_t setup_SSRS*/ 1
            );
        }
        // asm volatile ("csrr %0, mcycle" : "=r"(t2c));

        // asm volatile ("csrr %0, mcycle" : "=r"(t1e));
        dm_memcpy_async(
            /* dst */ &C_(b1, 0, 0),
            /* src */ &CC_(0, 0, 0),
            /* size */ buf_size_b * M * N * sizeof(double)
        );
        dm_wait(); 
    }
    // asm volatile ("csrr %0, mcycle" : "=r"(t2e));

    // asm volatile ("csrr %0, mcycle" : "=r"(t2t));

    // printf("Time copy1 %lu\n", t2p - t1p);
    // printf("Time compute %lu\n", t2c - t1c);
    // printf("Time copy2 %lu\n", t2e - t1e);
    // printf("Time total %lu\n", t2p - t1p + t2c - t1c + t2e - t1e);
    // printf("Time total (+openmp) %lu\n", t2t - t1t);

    ////snrt_l1free(src_buf);
}

static double* g_src_buf;

void matmul_raw_fp64_sdma_ssr_frep(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;
    
    size_t m_len = M / ntd;
    size_t m_off = m_len * tid;

    size_t buf_size_m = M;
    size_t buf_size_k = K;
    size_t buf_size_n = N;

    size_t scratchpad_max_size = (64 * 1024 / sizeof(double));
    size_t buf_size_b = scratchpad_max_size / (M * K + K * N + M * N);
    size_t buf_size_b_pow2 = 1;
    while(buf_size_b_pow2 < buf_size_b) buf_size_b_pow2 *= 2;
    if (buf_size_b_pow2 > 1 && buf_size_b_pow2 > buf_size_b) buf_size_b_pow2 /= 2;
    buf_size_b = (B < buf_size_b_pow2) ? B : buf_size_b_pow2;

    size_t stages = 2;

    if (tid == 0) {
        double* src_buf = (double*) snrt_l1alloc(stages * buf_size_b * (M * K + K * N + M * N) * sizeof(double));
        g_src_buf = src_buf;
        if (!src_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }

    snrt_cluster_hw_barrier();

    double* src_buf0 = g_src_buf;
    double* weight_buf0 = src_buf0 + buf_size_b * M * K;
    double* dst_buf0 = weight_buf0 + buf_size_b * K * N;
    double* src_buf1 = dst_buf0 + buf_size_b * M * N;
    double* weight_buf1 = src_buf1 + buf_size_b * M * K;
    double* dst_buf1 = weight_buf1 + buf_size_b * K * N;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        size_t b1 = 0;
        snrt_dma_start_1d(
                /* dst */ &AA0_(0, 0, 0),
                /* src */ &A_(b1, 0, 0),
                /* size */ buf_size_b * M * K * sizeof(double)
            );
            snrt_dma_start_1d(
                /* dst */ &BB0_(0, 0, 0),
                /* src */ &B_(b1, 0, 0),
                /* size */ buf_size_b * K * N * sizeof(double)
            );
            snrt_dma_start_1d(
                /* dst */ &CC0_(0, 0, 0),
                /* src */ &D_(b1, 0, 0),
                /* size */ buf_size_b * M * N * sizeof(double)
            );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    for (size_t b1 = 0; b1 < B; b1 += buf_size_b) {
        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (b1 != 0) {
                snrt_dma_start_1d(
                    /* dst */ &C_(b1 - buf_size_b, 0, 0),
                    /* src */ &CC1_(0, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (b1 + buf_size_b != B) {
                snrt_dma_start_1d(
                    /* dst */ &AA1_(0, 0, 0),
                    /* src */ &A_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * M * K * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ &BB1_(0, 0, 0),
                    /* src */ &B_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * K * N * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ &CC1_(0, 0, 0),
                    /* src */ &D_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }
        
        if (tid == 0) {
            for (size_t b2 = 0; b2 < buf_size_b; b2++) {    
                uint32_t ALPHA = 1;
                gemm_fp64_opt(
                    /*uint32_t M, uint32_t N, uint32_t K, */ M, N, K,
                    /*double* A, uint32_t ldA, uint32_t ta, */ &AA0_(b2, 0, 0), K, 0,
                    /*double* B, uint32_t ldB, int32_t tb, */ &BB0_(b2, 0, 0), N, 0,
                    /*double* C, uint32_t ldC,*/ &CC0_(b2, 0, 0), N,
                    /*const uint32_t* ALPHA,*/ &ALPHA,
                    /*uint32_t setup_SSRS*/ b1 == 0
                );
            }
        }

        snrt_cluster_hw_barrier();

        // swap current and next buffers
        double* tmp_buf;

        tmp_buf = src_buf0;
        src_buf0 = src_buf1;
        src_buf1 = tmp_buf;

        tmp_buf = weight_buf0;
        weight_buf0 = weight_buf1;
        weight_buf1 = tmp_buf;
        
        tmp_buf = dst_buf0;
        dst_buf0 = dst_buf1;
        dst_buf1 = tmp_buf;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &C_(B - buf_size_b, 0, 0),
            /* src */ &CC1_(0, 0, 0),
            /* size */ buf_size_b * M * N * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
}

void matmul_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    size_t buf_size_m = M;
    size_t buf_size_k = K;
    size_t buf_size_n = N;

    size_t scratchpad_max_size = (64 * 1024 / sizeof(double));
    size_t buf_size_b = scratchpad_max_size / (M * K + K * N + M * N);
    size_t buf_size_b_pow2 = 1;
    while(buf_size_b_pow2 < buf_size_b) buf_size_b_pow2 *= 2;
    if (buf_size_b_pow2 > 1 && buf_size_b_pow2 > buf_size_b) buf_size_b_pow2 /= 2;
    buf_size_b = (B < buf_size_b_pow2) ? B : buf_size_b_pow2;

    double* src_buf = (double*) snrt_l1alloc(buf_size_b * (M * K + K * N + M * N) * sizeof(double));
    double* weight_buf = src_buf + buf_size_b * M * K;
    double* dst_buf = weight_buf + buf_size_b * K * N;

    if (!src_buf) {
        printf("Error: failed to allocate scratchpad memory\n");
        while (1) {}
        return;
    }

    unsigned long t1t, t1p, t1c, t1e;
    unsigned long t2t, t2p, t2c, t2e;

    // asm volatile ("csrr %0, mcycle" : "=r"(t1t));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int ntd = 8 /* omp_get_num_threads() */;

        int m_len = (M / ntd);
        int m_off = tid * m_len;

        for (size_t b1 = 0; b1 < B; b1 += buf_size_b) {
            // asm volatile ("csrr %0, mcycle" : "=r"(t1p));
            if (tid == 0) {
                dm_memcpy_async(
                    /* dst */ &AA_(0, 0, 0),
                    /* src */ &A_(b1, 0, 0),
                    /* size */ buf_size_b * M * K * sizeof(double)
                );
                dm_memcpy_async(
                    /* dst */ &BB_(0, 0, 0),
                    /* src */ &B_(b1, 0, 0),
                    /* size */ buf_size_b * K * N * sizeof(double)
                );
                dm_memcpy_async(
                    /* dst */ &CC_(0, 0, 0),
                    /* src */ &D_(b1, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                dm_wait();
            }
            // asm volatile ("csrr %0, mcycle" : "=r"(t2p));

            #pragma omp barrier

            // asm volatile ("csrr %0, mcycle" : "=r"(t1c));
            for (size_t b2 = 0; b2 < buf_size_b; b2++) {
                //TIME(t3);
            
                uint32_t ALPHA = 1;
                gemm_fp64_opt(
                    /*uint32_t M, uint32_t N, uint32_t K, */ m_len, N, K,
                    /*double* A, uint32_t ldA, uint32_t ta, */ &AA_(b2, m_off, 0), K, 0,
                    /*double* B, uint32_t ldB, int32_t tb, */ &BB_(b2, 0, 0), N, 0,
                    /*double* C, uint32_t ldC,*/ &CC_(b2, m_off, 0), N,
                    /*const uint32_t* ALPHA,*/ &ALPHA,
                    /*uint32_t setup_SSRS*/ 1
                );
            
                //TIME(t4);

            }
            // asm volatile ("csrr %0, mcycle" : "=r"(t2c));

            #pragma omp barrier

            // asm volatile ("csrr %0, mcycle" : "=r"(t1e));
            if (tid == 0) {
                dm_memcpy_async(
                    /* dst */ &C_(b1, 0, 0),
                    /* src */ &CC_(0, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                dm_wait();
            }
        }
        // asm volatile ("csrr %0, mcycle" : "=r"(t2e));
    }
    // asm volatile ("csrr %0, mcycle" : "=r"(t2t));

    // printf("Time copy1 %lu\n", t2p - t1p);
    // printf("Time compute %lu\n", t2c - t1c);
    // printf("Time copy2 %lu\n", t2e - t1e);
    // printf("Time total %lu\n", t2p - t1p + t2c - t1c + t2e - t1e);
    // printf("Time total (+openmp) %lu\n", t2t - t1t);

    ////snrt_l1free(src_buf);
}


void matmul_raw_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
) {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_compute_core_num()*/;

    size_t m_len = M / ntd;
    size_t m_off = m_len * tid;

    size_t buf_size_m = M;
    size_t buf_size_k = K;
    size_t buf_size_n = N;

    size_t scratchpad_max_size = (64 * 1024 / sizeof(double));
    size_t buf_size_b = scratchpad_max_size / (M * K + K * N + M * N);
    size_t buf_size_b_pow2 = 1;
    while(buf_size_b_pow2 < buf_size_b) buf_size_b_pow2 *= 2;
    if (buf_size_b_pow2 > 1 && buf_size_b_pow2 > buf_size_b) buf_size_b_pow2 /= 2;
    buf_size_b = (B < buf_size_b_pow2) ? B : buf_size_b_pow2;

    size_t stages = 2;

    if (tid == 0) {
        g_src_buf = (double*) snrt_l1alloc(stages * buf_size_b * (M * K + K * N + M * N) * sizeof(double));
    
        if (!g_src_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
    }

    snrt_cluster_hw_barrier();

    double* src_buf0 = g_src_buf;
    double* weight_buf0 = src_buf0 + buf_size_b * M * K;
    double* dst_buf0 = weight_buf0 + buf_size_b * K * N;
    double* src_buf1 = dst_buf0 + buf_size_b * M * N;
    double* weight_buf1 = src_buf1 + buf_size_b * M * K;
    double* dst_buf1 = weight_buf1 + buf_size_b * K * N;

    if (snrt_is_dm_core()) {
        // copy data for the first iteration
        size_t b1 = 0;
        snrt_dma_start_1d(
                /* dst */ &AA0_(0, 0, 0),
                /* src */ &A_(b1, 0, 0),
                /* size */ buf_size_b * M * K * sizeof(double)
            );
            snrt_dma_start_1d(
                /* dst */ &BB0_(0, 0, 0),
                /* src */ &B_(b1, 0, 0),
                /* size */ buf_size_b * K * N * sizeof(double)
            );
            snrt_dma_start_1d(
                /* dst */ &CC0_(0, 0, 0),
                /* src */ &D_(b1, 0, 0),
                /* size */ buf_size_b * M * N * sizeof(double)
            );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    for (size_t b1 = 0; b1 < B; b1 += buf_size_b) {
        if (snrt_is_dm_core()) {
            // finish data movement for the previous iteration
            // check it is not the first iteration
            if (b1 != 0) {
                snrt_dma_start_1d(
                    /* dst */ &C_(b1 - buf_size_b, 0, 0),
                    /* src */ &CC1_(0, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }

        if (snrt_is_dm_core()) {
            // start data movement for the next iteration
            // check it is not the last iteration
            if (b1 + buf_size_b != B) {
                snrt_dma_start_1d(
                    /* dst */ &AA1_(0, 0, 0),
                    /* src */ &A_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * M * K * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ &BB1_(0, 0, 0),
                    /* src */ &B_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * K * N * sizeof(double)
                );
                snrt_dma_start_1d(
                    /* dst */ &CC1_(0, 0, 0),
                    /* src */ &D_(b1 + buf_size_b, 0, 0),
                    /* size */ buf_size_b * M * N * sizeof(double)
                );
                snrt_dma_wait_all();
            }
        }
        
        if (snrt_is_compute_core()) {
            for (size_t b2 = 0; b2 < buf_size_b; b2++) {    
                uint32_t ALPHA = 1;
                gemm_fp64_opt(
                    /*uint32_t M, uint32_t N, uint32_t K, */ m_len, N, K,
                    /*double* A, uint32_t ldA, uint32_t ta, */ &AA0_(b2, m_off, 0), K, 0,
                    /*double* B, uint32_t ldB, int32_t tb, */ &BB0_(b2, 0, 0), N, 0,
                    /*double* C, uint32_t ldC,*/ &CC0_(b2, m_off, 0), N,
                    /*const uint32_t* ALPHA,*/ &ALPHA,
                    /*uint32_t setup_SSRS*/ b1 == 0
                );
            }
        }

        snrt_cluster_hw_barrier();

        // swap current and next buffers
        double* tmp_buf;

        tmp_buf = src_buf0;
        src_buf0 = src_buf1;
        src_buf1 = tmp_buf;

        tmp_buf = weight_buf0;
        weight_buf0 = weight_buf1;
        weight_buf1 = tmp_buf;
        
        tmp_buf = dst_buf0;
        dst_buf0 = dst_buf1;
        dst_buf1 = tmp_buf;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ &C_(B - buf_size_b, 0, 0),
            /* src */ &CC1_(0, 0, 0),
            /* size */ buf_size_b * M * N * sizeof(double)
        );
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
}