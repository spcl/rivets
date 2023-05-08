#pragma once

#include <stddef.h>

void resampling_nearest_fp64(
    double* dst, double* src,
    size_t N, size_t C, 
    size_t OH, size_t OW,
    size_t IH, size_t IW,
    size_t stride_dst_n, size_t stride_dst_c, size_t stride_dst_h, size_t stride_dst_w,
    size_t stride_src_n, size_t stride_src_c, size_t stride_src_h, size_t stride_src_w
);
