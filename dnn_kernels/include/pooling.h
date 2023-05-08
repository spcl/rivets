#pragma once

#include <stddef.h>

void maxpool_fp64(
    double* dst, double* src,
    size_t N, size_t C,
    size_t src_h, size_t src_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,    
    size_t dilation_h, size_t dilation_w,
    size_t padding_h, size_t padding_w,
    size_t dst_stride_n, size_t dst_stride_c,
    size_t src_stride_n, size_t src_stride_c
);