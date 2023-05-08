#include "prelu.h"

#include <math.h>

#define D_(n, c, h, w) dst[(n) * stride_dst_N + (c) * stride_dst_C + (h) * stride_dst_H + (w) * stride_dst_W]
#define S_(n, c, h, w) src[(n) * stride_src_N + (c) * stride_src_C + (h) * stride_src_H + (w) * stride_src_W]
#define W_(n, c, h, w) weight[(n) * stride_weight_N + (c) * stride_weight_C + (h) * stride_weight_H + (w) * stride_weight_W]

// d = s if s > 0
// d = s * w if s <= 0
void prelu_fp64(
    double* dst, double* src, double* weight,
    size_t N, size_t C, size_t H, size_t W,
    size_t stride_dst_N, size_t stride_dst_C, size_t stride_dst_H, size_t stride_dst_W,
    size_t stride_src_N, size_t stride_src_C, size_t stride_src_H, size_t stride_src_W,
    size_t stride_weight_N, size_t stride_weight_C, size_t stride_weight_H, size_t stride_weight_W
) {
    for (size_t n = 0; n < N; n++)
    for (size_t c = 0; c < C; c++)
    for (size_t h = 0; h < H; h++)
    for (size_t w = 0; w < W; w++) {
        double ts = S_(n, c, h, w);
        double tw = W_(n, c, h, w);
        D_(n, c, h, w) = fmax(ts, 0) + fmin(ts, 0) * tw;
    }
}


