#include "batchnorm.h"

#include <math.h>

#define SQR(x) ((x) * (x))
#define S_(n, c, hw) src[(n) * stride_src_N + (c) * stride_src_C + (hw)]
#define D_(n, c, hw) dst[(n) * stride_dst_N + (c) * stride_dst_C + (hw)]

void batch_norm_fp64(
    double* dst, double* src, double* mu, double* sigma2, double* gamma, double* beta, double eps,
    size_t N, size_t C, size_t HW,
    size_t stride_dst_N, size_t stride_dst_C,
    size_t stride_src_N, size_t stride_src_C
) {
    // compute mean
    for (size_t c = 0; c < C; c++) {
        double mean = 0;
        for (size_t n = 0; n < N; n++) {
            for (size_t hw = 0; hw < HW; hw++) {
                mean += S_(n, c, hw);
            }
        }
        mu[c] = mean / (N * HW);
    }
    // compute variance
    for (size_t c = 0; c < C; c++) {
        double mean = mu[c];
        double var = 0;
        for (size_t n = 0; n < N; n++) {
            for (size_t hw = 0; hw < HW; hw++) {
                double diff = S_(n, c, hw) - mean;
                D_(n, c, hw) = diff;
                var += SQR(diff);
            }
        }
        sigma2[c] = var / (N * HW);
    }
    // compute result
    for (size_t c = 0; c < C; c++) {
        double rsigma = 1.0 / sqrt(sigma2[c] + eps);
        double gamma_c = gamma[c];
        double beta_c = beta[c];
        for (size_t n = 0; n < N; n++) {
            for (size_t hw = 0; hw < HW; hw++) {
                double diff = D_(n, c, hw);
                D_(n, c, hw) = diff * rsigma * gamma_c + beta_c;
            }
        }
    }
}