#pragma once

#include <stddef.h>

void conv(
    float* dst, float* src, float* weight, float* bias,
    size_t batch, size_t group, size_t in_channels, size_t out_channels,
    size_t src_x, size_t src_y,
    size_t weight_x, size_t weight_y,
    size_t padding_x, size_t padding_y,
    size_t stride_x, size_t stride_y,
    size_t dilation_x, size_t dilation_y,
    size_t stride_dst_batch, size_t stride_dst_group, size_t stride_dst_out_ch, size_t stride_dst_x, size_t stride_dst_y,
    size_t stride_src_batch, size_t stride_src_group, size_t stride_src_in_ch, size_t stride_src_x, size_t stride_src_y,
    size_t stride_weight_group, size_t stride_weight_out_ch, size_t stride_weight_in_ch, size_t stride_weight_x, size_t stride_weight_y
);