#pragma once

#include <stddef.h>

void local_response_normalization_across_channels(
    double* dst, double* src, double* tmp,
    size_t N, size_t C, size_t H, size_t W, size_t L,
    size_t stride_n, size_t stride_c,
    double k, double alpha, double beta
);