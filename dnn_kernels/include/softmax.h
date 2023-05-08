#pragma once

#include <stddef.h>

// net[o, i] = max_c(src[o, c, i])
// dst[o, c, i] = exp(src[o, c, i] - net[o, i]) / sum_c(exp(src[o, c, i] - net[o, i]))
void softmax(
    float* dst, float* src,
    size_t O, size_t C, size_t I,
    size_t stride_dst_out, size_t stride_dst_compute, size_t stride_dst_in,
    size_t stride_src_out, size_t stride_src_compute, size_t stride_src_in
);