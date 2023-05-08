#pragma once

#include <stddef.h>

void innerproduct_fp64(
    double* dst, double* src, double* weight, double* bias,
    size_t N, size_t OC, size_t IC,
    size_t stride_dst_N, size_t stride_dst_OC,
    size_t stride_src_N, size_t stride_src_IC,
    size_t stride_weight_OC, size_t stride_weight_IC
);