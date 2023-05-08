#pragma once

#include <stddef.h>

void batch_norm_fp64(
    double* dst, double* src, double* mu, double* sigma2, double* gamma, double* beta, double eps,
    size_t N, size_t C, size_t HW,
    size_t stride_dst_N, size_t stride_dst_C,
    size_t stride_src_N, size_t stride_src_C
);