#pragma once

#include <stddef.h>

void prelu_fp64(
    double* dst, double* src, double* weight,
    size_t N, size_t C, size_t H, size_t W,
    size_t stride_dst_N, size_t stride_dst_C, size_t stride_dst_H, size_t stride_dst_W,
    size_t stride_src_N, size_t stride_src_C, size_t stride_src_H, size_t stride_src_W,
    size_t stride_weight_N, size_t stride_weight_C, size_t stride_weight_H, size_t stride_weight_W
);
