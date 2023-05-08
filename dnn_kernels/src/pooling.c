#include "pooling.h"

#include <math.h>
#include <float.h>

#define D_(n, c, h, w) dst[(n) * dst_stride_n + (c) * dst_stride_c + (h) * dst_w + (w)]
#define S_(n, c, h, w) src[(n) * src_stride_n + (c) * src_stride_c + (h) * src_w + (w)]

#define SI(dst_idx, stride, weight_idx, dilation, padding) \
    ((dst_idx) * (stride) + (weight_idx) * (dilation + 1) - (padding))

// dst[n,c,oh,ow] = max_(kh,kw) src[n,c,oh*stride_h+kh*(dilation_h+1)-padding_h,ow*stride_w+kw*(dilation_w+1)-padding_w]


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
) {
    size_t dilated_weight_h = (kernel_h - 1) * (dilation_h + 1) + 1;
    size_t dilated_weight_w = (kernel_w - 1) * (dilation_w + 1) + 1;
    size_t dst_h = (src_h - dilated_weight_h + 2 * padding_h) / stride_h + 1;
    size_t dst_w = (src_w - dilated_weight_w + 2 * padding_w) / stride_w + 1;

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t oh = 0; oh < dst_h; oh++) {
                size_t kh_start = 0;
                while (SI(oh, stride_h, kh_start, dilation_h, padding_h) < 0) kh_start++;
                size_t kh_end = kernel_h;
                while (SI(oh, stride_h, kh_end - 1, dilation_h, padding_h) >= src_h) kh_end--;
                for (size_t ow = 0; ow < dst_w; ow++) {
                    size_t kw_start = 0;
                    while (SI(ow, stride_w, kw_start, dilation_w, padding_w) < 0) kw_start++;
                    size_t kw_end = kernel_w;
                    while (SI(ow, stride_w, kw_end - 1, dilation_w, padding_w) >= src_w) kw_end--;

                    double max = -DBL_MAX;
                    for (size_t kh = kh_start; kh < kh_end; kh++)
                    for (size_t kw = kw_start; kw < kw_end; kw++) {
                        size_t ih = SI(oh, stride_h, kh, dilation_h, padding_h);
                        size_t iw = SI(ow, stride_w, kw, dilation_w, padding_w);
                        max = fmax(max, S_(n, c, ih, iw));
                    }
                    D_(n, c, oh, ow) = max;
                }
            }
        }
    }
}