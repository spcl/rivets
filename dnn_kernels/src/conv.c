#include "conv.h"

#define D_(b, g, o, x, y) dst[(b) * stride_dst_batch + (g) * stride_dst_group + (o) * stride_dst_out_ch + (x) * stride_dst_x + (y) * stride_dst_y]
#define S_(b, g, i, x, y) src[(b) * stride_src_batch + (g) * stride_src_group + (i) * stride_src_in_ch + (x) * stride_src_x + (y) * stride_src_y]
#define W_(g, o, i, x, y) weight[(g) * stride_weight_group + (o) * stride_weight_out_ch + (i) * stride_weight_in_ch + (x) * stride_weight_x + (y) * stride_weight_y]
#define B_(g, o) bias[(g) * out_channels + (o)]

#define SI(dst_idx, stride, weight_idx, dilation, padding) (dst_idx) * (stride) + (weight_idx) * (dilation + 1) - (padding)

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
) {
    // https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html
    size_t dilated_weight_x = (weight_x - 1) * (dilation_x + 1) + 1;
    size_t dilated_weight_y = (weight_y - 1) * (dilation_y + 1) + 1;
    size_t dst_x = (src_x - dilated_weight_x + 2 * padding_x) / stride_x + 1;
    size_t dst_y = (src_y - dilated_weight_y + 2 * padding_y) / stride_y + 1;

    for (size_t b = 0; b < batch; b++)
    for (size_t g = 0; g < group; g++)
    for (size_t o = 0; o < out_channels; o++)
    for (size_t dx = 0; dx < dst_x; dx++) {
        size_t wx_start = 0;
        while (SI(dx, stride_x, wx_start, dilation_x, padding_x) < 0) wx_start++;
        size_t wx_end = weight_x;
        while (SI(dx, stride_x, wx_end - 1, dilation_x, padding_x) >= src_x) wx_end--;
        for (size_t dy = 0; dy < dst_y; dy++) {
            size_t wy_start = 0;
            while (SI(dy, stride_y, wy_start, dilation_y, padding_y) < 0) wy_start++;
            size_t wy_end = weight_y;
            while (SI(dy, stride_y, wy_end - 1, dilation_y, padding_y) >= src_y) wy_end--;
            float tmp = B_(g, o);
            for (size_t i = 0; i < in_channels; i++)
            for (size_t wx = wx_start; wx < wx_end; wx++)
            for (size_t wy = wy_start; wy < wy_end; wy++) {
                size_t sx = SI(dx, stride_x, wx, dilation_x, padding_x);
                size_t sy = SI(dy, stride_y, wy, dilation_y, padding_y);
                tmp += S_(b, g, i, sx, sy) * W_(g, o, i, wx, wy);
            }
            D_(b, g, o, dx, dy) = tmp;
        }
    }
}