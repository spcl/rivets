#include "layernorm.h"

#include <math.h>

#include "printf.h"
#include "snrt.h"
#include "omp.h"
#include "dm.h"

#define SQR(x) (x) * (x)

#define SQRT sqrtf

void layer_norm_fp32(
    float* dst, float* src, float* mu, float* gamma, float* sigma, float* beta, float eps,
    size_t n1, size_t n2, size_t s1, size_t s2
) {
    for (size_t i1 = 0; i1 < n1; i1++) {
        mu[i1] = 0;
        for (size_t i2 = 0; i2 < n2; i2++) {
            mu[i1] += src[i1 * s1 + i2 * s2];
        }
        mu[i1] /= n2;
        sigma[i1] = 0;
        for (size_t i2 = 0; i2 < n2; i2++) {
            sigma[i1] += SQR(src[i1 * s1 + i2 * s2] - mu[i1]);
        }
        sigma[i1] = 1.0f / SQRT(sigma[i1] / (n2 - 1) + eps);
        for (size_t i2 = 0; i2 < n2; i2++) {
            dst[i1 * s1 + i2 * s2] = gamma[i2] * (src[i1 * s1 + i2 * s2] - mu[i1]) * sigma[i1] + beta[i2];
        }
    }
}

#undef SQRT
#define SQRT sqrt

void layer_norm_fp64(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
) {
    for (size_t i1 = 0; i1 < n1; i1++) {
        mu[i1] = 0;
        for (size_t i2 = 0; i2 < n2; i2++) {
            mu[i1] += src[i1 * s1 + i2 * s2];
        }
        mu[i1] /= n2;
        sigma[i1] = 0;
        for (size_t i2 = 0; i2 < n2; i2++) {
            sigma[i1] += SQR(src[i1 * s1 + i2 * s2] - mu[i1]);
        }
        sigma[i1] = 1.0 / SQRT(sigma[i1] / (n2 - 1) + eps);
        for (size_t i2 = 0; i2 < n2; i2++) {
            dst[i1 * s1 + i2 * s2] = gamma[i2] * (src[i1 * s1 + i2 * s2] - mu[i1]) * sigma[i1] + beta[i2];
        }
    }
}

void layer_norm_fp64_sdma(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    size_t batch_buf_size = 8;
    double* src_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* dst_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    if (!src_buf || !dst_buf) return;
    for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
        size_t b_end = b1 + batch_buf_size;
        if (b_end > B) b_end = B;

        dm_memcpy_async(
            /* dst */ src_buf,
            /* src */ &src[b1 * stride_B],
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();

        for (size_t b2 = 0; b2 < b_end - b1; b2++) {
            size_t b = b1 + b2;
            mu[b] = 0;
            for (size_t n = 0; n < N; n++) {
                mu[b] += src_buf[b2 * N + n];
            }
            mu[b] /= N;
            sigma[b] = 0;
            for (size_t n = 0; n < N; n++) {
                sigma[b] += SQR(src_buf[b2 * N + n] - mu[b]);
            }
            sigma[b] = SQRT(sigma[b] / (N - 1));
            for (size_t n = 0; n < N; n++) {
                dst_buf[b2 * N + n] = gamma[n] * (src_buf[b2 * N + n] - mu[b]) / SQRT(SQR(sigma[b]) + eps) + beta[n];
            }
        }

        dm_memcpy_async(
            /* dst */ &dst[b1 * stride_B],
            /* src */ dst_buf,
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();
    }
    //snrt_l1free(dst_buf);
    //snrt_l1free(src_buf);
}


void layer_norm_fp64_sdma_ssr(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    const size_t batch_buf_size = 8;
    double* src_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* dst_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* gam_buf = (double*) snrt_l1alloc(N * sizeof(double));
    double* bet_buf = (double*) snrt_l1alloc(N * sizeof(double));

    dm_memcpy_async(
        /* dst */ gam_buf,
        /* src */ gamma,
        /* size */ N * sizeof(double)
    );
    dm_memcpy_async(
        /* dst */ bet_buf,
        /* src */ beta,
        /* size */ N * sizeof(double)
    );
    dm_wait();

    if (!src_buf || !dst_buf) return;
    for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
        size_t b_end = b1 + batch_buf_size;
        if (b_end > B) b_end = B;

        dm_memcpy_async(
            /* dst */ src_buf,
            /* src */ &src[b1 * stride_B],
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();

        for (size_t b2 = 0; b2 < b_end - b1; b2++) {
            size_t b = b1 + b2;
            
            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &src_buf[b2 * N]);

            double lmu = 0;
            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            for (size_t n = 0; n < N; n++) {
                asm volatile("fadd.d %0, ft0, %0" : "+f"(lmu) :: "ft0", "ft1", "ft2");
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();
            lmu = lmu / N;
            mu[b] = lmu;
            

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &src_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_write(1, 0, &dst_buf[b2 * N]);

            double lsigma = 0;
            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            for (size_t n = 0; n < N; n++) {
                double tmp;
                asm volatile(
                    "fsub.d %[tmp], ft0, %[mu];\n"
                    "fmadd.d %[sigma], %[tmp], %[tmp], %[sigma];\n"
                    "fmv.d ft1, %[tmp]"
                    : [tmp] "=f"(tmp), [sigma] "+f"(lsigma)
                    : [mu] "f"(lmu)
                    : "ft0", "ft1", "ft2"
                );
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();
            lsigma = SQRT(lsigma / (N - 1));
            sigma[b] = lsigma;
            

            double factor = 1 / SQRT(SQR(lsigma) + eps);

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_read(1, 0, gam_buf);
            __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
            __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            for (size_t n = 0; n < N; n++) {
                double tmp;
                asm volatile(
                    "fmul.d %[tmp], ft0, %[factor];\n"
                    "fmul.d ft2, %[tmp], ft1"
                    : [tmp] "=f"(tmp)
                    : [factor] "f"(factor)
                    : "ft0", "ft1", "ft2"
                );
                //dst_buf[b2 * N + n] = dst_buf[b2 * N + n] * factor * gamma[n];
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_read(1, 0, bet_buf);
            __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
            __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            for (size_t n = 0; n < N; n++) {
                asm volatile(
                    "fadd.d ft2, ft0, ft1" ::: "ft0", "ft1", "ft2", "memory"
                );

                //dst_buf[b2 * N + n] = dst_buf[b2 * N + n] + beta[n];
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();
        }

        dm_memcpy_async(
            /* dst */ &dst[b1 * stride_B],
            /* src */ dst_buf,
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();
    }
    //snrt_l1free(bet_buf);
    //snrt_l1free(gam_buf);
    //snrt_l1free(dst_buf);
    //snrt_l1free(src_buf);
}

void layer_norm_fp64_sdma_ssr_frep(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    const size_t batch_buf_size = 8;
    double* src_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* dst_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* gam_buf = (double*) snrt_l1alloc(N * sizeof(double));
    double* bet_buf = (double*) snrt_l1alloc(N * sizeof(double));

    dm_memcpy_async(
        /* dst */ gam_buf,
        /* src */ gamma,
        /* size */ N * sizeof(double)
    );
    dm_memcpy_async(
        /* dst */ bet_buf,
        /* src */ beta,
        /* size */ N * sizeof(double)
    );
    dm_wait();

    if (!src_buf || !dst_buf) return;
    for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
        size_t b_end = b1 + batch_buf_size;
        if (b_end > B) b_end = B;

        dm_memcpy_async(
            /* dst */ src_buf,
            /* src */ &src[b1 * stride_B],
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();

        for (size_t b2 = 0; b2 < b_end - b1; b2++) {
            size_t b = b1 + b2;
            
            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &src_buf[b2 * N]);

            double lmu = 0;
            __builtin_ssr_enable();
            {
                double tmp;
                asm volatile(
                    "frep.o %[reps], 1, 0, 0;\n"
                    "fadd.d %[mu], ft0, %[mu];"
                    : [mu] "+f"(lmu), [tmp] "=f"(tmp)
                    : [reps] "r"(N - 1)
                    : "ft0", "ft1", "ft2", "memory"
                );
            }
            __builtin_ssr_disable();
            lmu = lmu / N;
            mu[b] = lmu;
            

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &src_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_write(1, 0, &dst_buf[b2 * N]);

            double lsigma = 0;
            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            {
                double tmp;
                asm volatile(
                    "frep.o %[rep], 3, 0, 0;\n"
                    "fsub.d %[tmp], ft0, %[mu];\n"
                    "fmadd.d %[sigma], %[tmp], %[tmp], %[sigma];\n"
                    "fmv.d ft1, %[tmp];"
                    : [tmp] "=&f"(tmp), [sigma] "+f"(lsigma)
                    : [mu] "f"(lmu), [rep] "r"(N - 1)
                    : "ft0", "ft1", "ft2", "memory"
                );
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();


            lsigma = SQRT(lsigma / (N - 1));
            sigma[b] = lsigma;
            

            double factor = 1 / SQRT(SQR(lsigma) + eps);

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_read(1, 0, gam_buf);
            __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
            __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            {
                double tmp;
                asm volatile(
                    "frep.o %[rep], 2, 0, 0;\n"
                    "fmul.d %[tmp], ft0, %[factor];\n"
                    "fmul.d ft2, %[tmp], ft1"
                    : [tmp] "=&f"(tmp)
                    : [factor] "f"(factor), [rep] "r"(N - 1)
                    : "ft0", "ft1", "ft2"
                );
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();

            __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
            __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
            __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
            __builtin_ssr_read(1, 0, bet_buf);
            __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
            __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

            __builtin_ssr_enable();
            asm volatile("" ::: "memory");
            for (size_t n = 0; n < N; n++) {
                asm volatile(
                    "fadd.d ft2, ft0, ft1" ::: "ft0", "ft1", "ft2", "memory"
                );
            }
            asm volatile("" ::: "memory");
            __builtin_ssr_disable();
        }

        dm_memcpy_async(
            /* dst */ &dst[b1 * stride_B],
            /* src */ dst_buf,
            /* size */ (b_end - b1) * N * sizeof(double)
        );
        dm_wait();
    }
    //snrt_l1free(bet_buf);
    //snrt_l1free(gam_buf);
    //snrt_l1free(dst_buf);
    //snrt_l1free(src_buf);
}


static double* g_src_buf;

void layer_norm_raw_fp64_sdma_ssr_frep(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;
    const int unroll = 4;

    size_t scratchpad_max_size = 1024 * 8;
    size_t batch_buf_size = (B * N < scratchpad_max_size) ? B : (scratchpad_max_size / N);
    if (batch_buf_size < ntd * unroll) batch_buf_size = ntd * unroll;

    if (tid == 0) {
        double* src_buf = (double*) snrt_l1alloc((batch_buf_size * N + N + N) * sizeof(double));
        if (!src_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
        g_src_buf = src_buf;
    }

    snrt_cluster_hw_barrier();

    double* tmp_buf = g_src_buf;
    double* gam_buf = tmp_buf + batch_buf_size * N;
    double* bet_buf = gam_buf + N;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ gam_buf,
            /* src */ gamma,
            /* size */ N * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ bet_buf,
            /* src */ beta,
            /* size */ N * sizeof(double)
        );
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (tid == 0) {
        snrt_ssr_loop_4d(SNRT_SSR_DM0,
            unroll, N, 4, batch_buf_size / unroll, 
            N * sizeof(double), sizeof(double), 0, N * sizeof(double) * unroll
        );
        snrt_ssr_loop_4d(SNRT_SSR_DM1,
            unroll, 2, N, batch_buf_size / unroll,
            0, (uint32_t)bet_buf - (uint32_t)gam_buf, sizeof(double), 0 * unroll
        );
        snrt_ssr_loop_4d(SNRT_SSR_DM2,
            unroll, N, 2, batch_buf_size / unroll,
            N * sizeof(double), sizeof(double), 0, N * sizeof(double) * unroll
        );
    }

    for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
        size_t b_end = b1 + batch_buf_size;
        if (b_end > B) b_end = B;
        size_t b2_len = b_end - b1;

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(
                /* dst */ tmp_buf,
                /* src */ &src[b1 * stride_B],
                /* size */ (b_end - b1) * N * sizeof(double)
            );
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (tid == 0) {
            
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, tmp_buf);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, gam_buf);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_4D, tmp_buf);

            snrt_ssr_enable();
            for (size_t b2 = 0; b2 < b2_len; b2 += unroll) {
                size_t b = b1 + b2;

                register double r[unroll];

                for (size_t i = 0; i < unroll; i++) {
                    r[i] = 0;
                }

                asm volatile(
                    "frep.o %[reps], %[unroll], 0, 0;"
                    "fadd.d %[mu0], ft0, %[mu0];"
                    "fadd.d %[mu1], ft0, %[mu1];"
                    "fadd.d %[mu2], ft0, %[mu2];"
                    "fadd.d %[mu3], ft0, %[mu3];"
                    : [mu0] "+f"(r[0]), [mu1] "+f"(r[1]), [mu2] "+f"(r[2]), [mu3] "+f"(r[3])
                    : [reps] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile("fmul.d %0, %0, %1" : "+f"(r[i]) : "f"(1. / N) : "ft0", "ft1", "ft2", "memory");
                }

                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fsub.d ft2, ft0, %[mu0];"
                    "fsub.d ft2, ft0, %[mu1];"
                    "fsub.d ft2, ft0, %[mu2];"
                    "fsub.d ft2, ft0, %[mu3];"
                    :
                    : [mu0] "f"(r[0]), [mu1] "f"(r[1]), [mu2] "f"(r[2]), [mu3] "f"(r[3])
                    , [rep] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );

                for (size_t i = 0; i < unroll; i++) {
                    r[i] = 0;
                }

                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fmadd.d %[s0], ft0, ft0, %[s0];"
                    "fmadd.d %[s1], ft0, ft0, %[s1];"
                    "fmadd.d %[s2], ft0, ft0, %[s2];"
                    "fmadd.d %[s3], ft0, ft0, %[s3];"
                    : [s0] "+f"(r[0]), [s1] "+f"(r[1]), [s2] "+f"(r[2]), [s3] "+f"(r[3])
                    : [rep] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );

                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fmadd.d %[s], %[s], %[x], %[eps];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fsqrt.d %[s], %[s];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fdiv.d %[s], %[one], %[s];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }

                register double t[unroll];
                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fmul.d %[t0], ft1, %[f0];"
                    "fmul.d %[t1], ft1, %[f1];"
                    "fmul.d %[t2], ft1, %[f2];"
                    "fmul.d %[t3], ft1, %[f3];"
                    "fmadd.d ft2, %[t0], ft0, ft1;"
                    "fmadd.d ft2, %[t1], ft0, ft1;"
                    "fmadd.d ft2, %[t2], ft0, ft1;"
                    "fmadd.d ft2, %[t3], ft0, ft1;"
                    : [t0] "=&f"(t[0]), [t1] "=&f"(t[1]), [t2] "=&f"(t[2]), [t3] "=&f"(t[3])
                    : [f0] "f"(r[0]), [f1] "f"(r[1]), [f2] "f"(r[2]), [f3] "f"(r[3])
                    , [rep] "r"(N - 1), [unroll] "i"(2 * unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );
            }
            __builtin_ssr_barrier(SNRT_SSR_DM2);
            snrt_ssr_disable();
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(
                /* dst */ &dst[b1 * stride_B],
                /* src */ tmp_buf,
                /* size */ (b_end - b1) * N * sizeof(double)
            );
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

}


void layer_norm_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    const size_t batch_buf_size = 8;
    double* src_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* dst_buf = (double*) snrt_l1alloc(batch_buf_size * N * sizeof(double));
    double* gam_buf = (double*) snrt_l1alloc(N * sizeof(double));
    double* bet_buf = (double*) snrt_l1alloc(N * sizeof(double));

    dm_memcpy_async(
        /* dst */ gam_buf,
        /* src */ gamma,
        /* size */ N * sizeof(double)
    );
    dm_memcpy_async(
        /* dst */ bet_buf,
        /* src */ beta,
        /* size */ N * sizeof(double)
    );
    dm_wait();

    if (!src_buf || !dst_buf) return;

    #pragma omp parallel
    {
        for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
            size_t b_end = b1 + batch_buf_size;
            if (b_end > B) b_end = B;

            if (omp_get_thread_num() == 0) {
                dm_memcpy_async(
                    /* dst */ src_buf,
                    /* src */ &src[b1 * stride_B],
                    /* size */ (b_end - b1) * N * sizeof(double)
                );
                dm_wait();
            }
            #pragma omp barrier

            #pragma omp for
            for (size_t b2 = 0; b2 < b_end - b1; b2++) {
                size_t b = b1 + b2;
                
                __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
                __builtin_ssr_read(0, 0, &src_buf[b2 * N]);

                double lmu = 0;
                __builtin_ssr_enable();
                {
                    double tmp;
                    asm volatile(
                        "frep.o %[reps], 1, 0, 0;\n"
                        "fadd.d %[mu], ft0, %[mu];"
                        : [mu] "+f"(lmu), [tmp] "=f"(tmp)
                        : [reps] "r"(N - 1)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                __builtin_ssr_disable();
                lmu = lmu / N;
                mu[b] = lmu;
                

                __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
                __builtin_ssr_read(0, 0, &src_buf[b2 * N]);
                __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
                __builtin_ssr_write(1, 0, &dst_buf[b2 * N]);

                double lsigma = 0;
                __builtin_ssr_enable();
                asm volatile("" ::: "memory");
                {
                    double tmp;
                    asm volatile(
                        "frep.o %[rep], 3, 0, 0;\n"
                        "fsub.d %[tmp], ft0, %[mu];\n"
                        "fmadd.d %[sigma], %[tmp], %[tmp], %[sigma];\n"
                        "fmv.d ft1, %[tmp];"
                        : [tmp] "=&f"(tmp), [sigma] "+f"(lsigma)
                        : [mu] "f"(lmu), [rep] "r"(N - 1)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                asm volatile("" ::: "memory");
                __builtin_ssr_disable();


                lsigma = SQRT(lsigma / (N - 1));
                sigma[b] = lsigma;
                

                double factor = 1 / SQRT(SQR(lsigma) + eps);

                __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
                __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
                __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
                __builtin_ssr_read(1, 0, gam_buf);
                __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
                __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

                __builtin_ssr_enable();
                asm volatile("" ::: "memory");
                {
                    double tmp;
                    asm volatile(
                        "frep.o %[rep], 2, 0, 0;\n"
                        "fmul.d %[tmp], ft0, %[factor];\n"
                        "fmul.d ft2, %[tmp], ft1"
                        : [tmp] "=&f"(tmp)
                        : [factor] "f"(factor), [rep] "r"(N - 1)
                        : "ft0", "ft1", "ft2"
                    );
                }
                asm volatile("" ::: "memory");
                __builtin_ssr_disable();

                __builtin_ssr_setup_bound_stride_1d(0, N - 1, sizeof(double));
                __builtin_ssr_read(0, 0, &dst_buf[b2 * N]);
                __builtin_ssr_setup_bound_stride_1d(1, N - 1, sizeof(double));
                __builtin_ssr_read(1, 0, bet_buf);
                __builtin_ssr_setup_bound_stride_1d(2, N - 1, sizeof(double));
                __builtin_ssr_write(2, 0, &dst_buf[b2 * N]);

                __builtin_ssr_enable();
                asm volatile("" ::: "memory");
                for (size_t n = 0; n < N; n++) {
                    asm volatile(
                        "fadd.d ft2, ft0, ft1" ::: "ft0", "ft1", "ft2", "memory"
                    );
                }
                asm volatile("" ::: "memory");
                __builtin_ssr_disable();
            }

            if (omp_get_thread_num() == 0) {
                dm_memcpy_async(
                    /* dst */ &dst[b1 * stride_B],
                    /* src */ dst_buf,
                    /* size */ (b_end - b1) * N * sizeof(double)
                );
                dm_wait();
            }
        }
    }
    //snrt_l1free(bet_buf);
    //snrt_l1free(gam_buf);
    //snrt_l1free(dst_buf);
    //snrt_l1free(src_buf);
}

void layer_norm_raw_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t B, size_t N, size_t stride_B, size_t stride_N
) {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = 8 /*snrt_cluster_core_num()*/;
    const int unroll = 4;

    size_t scratchpad_max_size = 1024 * 8;
    size_t batch_buf_size = (B * N < scratchpad_max_size) ? B : (scratchpad_max_size / N);
    if (batch_buf_size < ntd * unroll) batch_buf_size = ntd * unroll;

    if (tid == 0) {
        double* src_buf = (double*) snrt_l1alloc((batch_buf_size * N + N + N) * sizeof(double));
        if (!src_buf) {
            printf("Error: failed to allocate scratchpad memory\n");
            while (1) {}
            return;
        }
        g_src_buf = src_buf;
    }

    snrt_cluster_hw_barrier();

    double* tmp_buf = g_src_buf;
    double* gam_buf = tmp_buf + batch_buf_size * N;
    double* bet_buf = gam_buf + N;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(
            /* dst */ gam_buf,
            /* src */ gamma,
            /* size */ N * sizeof(double)
        );
        snrt_dma_start_1d(
            /* dst */ bet_buf,
            /* src */ beta,
            /* size */ N * sizeof(double)
        );
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    size_t b2_len_thr = batch_buf_size / ntd;

    if (snrt_is_compute_core()) {
        snrt_ssr_loop_4d(SNRT_SSR_DM0,
            unroll, N, 4, b2_len_thr / unroll, 
            N * sizeof(double), sizeof(double), 0, N * sizeof(double) * unroll
        );
        snrt_ssr_loop_4d(SNRT_SSR_DM1,
            unroll, 2, N, b2_len_thr / unroll,
            0, (uint32_t)bet_buf - (uint32_t)gam_buf, sizeof(double), 0 * unroll
        );
        snrt_ssr_loop_4d(SNRT_SSR_DM2,
            unroll, N, 2, b2_len_thr / unroll,
            N * sizeof(double), sizeof(double), 0, N * sizeof(double) * unroll
        );
    }

    for (size_t b1 = 0; b1 < B; b1 += batch_buf_size) {
        size_t b_end = b1 + batch_buf_size;
        if (b_end > B) b_end = B;
        size_t b2_len = b_end - b1;

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(
                /* dst */ tmp_buf,
                /* src */ &src[b1 * stride_B],
                /* size */ (b_end - b1) * N * sizeof(double)
            );
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();
        if (snrt_is_compute_core()) {
            
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, &tmp_buf[N * b2_len_thr * tid]);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, gam_buf);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_4D, &tmp_buf[N * b2_len_thr * tid]);

            snrt_ssr_enable();
            for (size_t b2 = b2_len_thr * tid; b2 < b2_len_thr * (tid + 1); b2 += unroll) {
                size_t b = b1 + b2;

                register double r[unroll];

                for (size_t i = 0; i < unroll; i++) {
                    r[i] = 0;
                }

                asm volatile(
                    "frep.o %[reps], %[unroll], 0, 0;"
                    "fadd.d %[mu0], ft0, %[mu0];"
                    "fadd.d %[mu1], ft0, %[mu1];"
                    "fadd.d %[mu2], ft0, %[mu2];"
                    "fadd.d %[mu3], ft0, %[mu3];"
                    : [mu0] "+f"(r[0]), [mu1] "+f"(r[1]), [mu2] "+f"(r[2]), [mu3] "+f"(r[3])
                    : [reps] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile("fmul.d %0, %0, %1" : "+f"(r[i]) : "f"(1. / N) : "ft0", "ft1", "ft2", "memory");
                }

                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fsub.d ft2, ft0, %[mu0];"
                    "fsub.d ft2, ft0, %[mu1];"
                    "fsub.d ft2, ft0, %[mu2];"
                    "fsub.d ft2, ft0, %[mu3];"
                    :
                    : [mu0] "f"(r[0]), [mu1] "f"(r[1]), [mu2] "f"(r[2]), [mu3] "f"(r[3])
                    , [rep] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );

                for (size_t i = 0; i < unroll; i++) {
                    r[i] = 0;
                }

                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fmadd.d %[s0], ft0, ft0, %[s0];"
                    "fmadd.d %[s1], ft0, ft0, %[s1];"
                    "fmadd.d %[s2], ft0, ft0, %[s2];"
                    "fmadd.d %[s3], ft0, ft0, %[s3];"
                    : [s0] "+f"(r[0]), [s1] "+f"(r[1]), [s2] "+f"(r[2]), [s3] "+f"(r[3])
                    : [rep] "r"(N - 1), [unroll] "i"(unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );

                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fmadd.d %[s], %[s], %[x], %[eps];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fsqrt.d %[s], %[s];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }
                for (size_t i = 0; i < unroll; i++) {
                    asm volatile(
                        "fdiv.d %[s], %[one], %[s];"
                        : [s] "+f"(r[i])
                        : [x] "f"(1. / (N - 1)), [eps] "f"(eps), [one] "f"(1.0)
                        : "ft0", "ft1", "ft2", "memory"
                    );
                }

                register double t[unroll];
                asm volatile(
                    "frep.o %[rep], %[unroll], 0, 0;"
                    "fmul.d %[t0], ft1, %[f0];"
                    "fmul.d %[t1], ft1, %[f1];"
                    "fmul.d %[t2], ft1, %[f2];"
                    "fmul.d %[t3], ft1, %[f3];"
                    "fmadd.d ft2, %[t0], ft0, ft1;"
                    "fmadd.d ft2, %[t1], ft0, ft1;"
                    "fmadd.d ft2, %[t2], ft0, ft1;"
                    "fmadd.d ft2, %[t3], ft0, ft1;"
                    : [t0] "=&f"(t[0]), [t1] "=&f"(t[1]), [t2] "=&f"(t[2]), [t3] "=&f"(t[3])
                    : [f0] "f"(r[0]), [f1] "f"(r[1]), [f2] "f"(r[2]), [f3] "f"(r[3])
                    , [rep] "r"(N - 1), [unroll] "i"(2 * unroll)
                    : "ft0", "ft1", "ft2", "memory"
                );
            }
            __builtin_ssr_barrier(SNRT_SSR_DM2);
            snrt_ssr_disable();
        }

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(
                /* dst */ &dst[b1 * stride_B],
                /* src */ tmp_buf,
                /* size */ (b_end - b1) * N * sizeof(double)
            );
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

}