#include "resampling.h"

#include <math.h>

#define D_(n, c, h, w) dst[(n) * stride_dst_n + (c) * stride_dst_c + (h) * stride_dst_h + (w) * stride_dst_w]
#define S_(n, c, h, w) dst[(n) * stride_src_n + (c) * stride_src_c + (h) * stride_src_h + (w) * stride_src_w]
#define I_(o, f) ((size_t)round(((o) + 0.5) / (f) - 0.5))

// dst[n,c,oh,ow] = src[n,c,ih,iw]
void resampling_nearest_fp64(
    double* dst, double* src,
    size_t N, size_t C, 
    size_t OH, size_t OW,
    size_t IH, size_t IW,
    size_t stride_dst_n, size_t stride_dst_c, size_t stride_dst_h, size_t stride_dst_w,
    size_t stride_src_n, size_t stride_src_c, size_t stride_src_h, size_t stride_src_w
) {
    double fh = OH / (double)IH;
    double fw = OW / (double)IW;

    int oh_start = 0;
    while (I_(oh_start, fh) < 0) oh_start++;
    int oh_end = OH;
    while (I_(oh_end - 1, fh) >= OH) oh_end--;

    int ow_start = 0;
    while (I_(ow_start, fw) < 0) ow_start++;
    int ow_end = OW;
    while (I_(ow_end - 1, fw) >= OW) ow_end--;

    for (size_t n = 0; n < N; n++)
    for (size_t c = 0; c < C; c++) {
        for (int oh = 0; oh < oh_start; oh++) {
            for (int ow = 0; ow < ow_start; ow++) {
                D_(n, c, oh, ow) = S_(n, c, 0, 0);
            }
            for (int ow = ow_start; ow < ow_end; ow++) {
                D_(n, c, oh, ow) = S_(n, c, 0, I_(ow, fw));
            }
            for (int ow = ow_end; ow < OW; ow++) {
                D_(n, c, oh, ow) = S_(n, c, 0, IW-1);
            }
        }
        for (int oh = oh_start; oh < oh_end; oh++) {
            for (int ow = 0; ow < ow_start; ow++) {
                D_(n, c, oh, ow) = S_(n, c, I_(oh, fh), 0);
            }
            for (int ow = ow_start; ow < ow_end; ow++) {
                D_(n, c, oh, ow) = S_(n, c, I_(oh, fh), I_(ow, fw));
            }
            for (int ow = ow_end; ow < OW; ow++) {
                D_(n, c, oh, ow) = S_(n, c, I_(oh, fh), IW-1);
            }
        }
        for (int oh = oh_end; oh < OH; oh++) {
            for (int ow = 0; ow < ow_start; ow++) {
                D_(n, c, oh, ow) = S_(n, c, IH-1, 0);
            }
            for (int ow = ow_start; ow < ow_end; ow++) {
                D_(n, c, oh, ow) = S_(n, c, IH-1, I_(ow, fw));
            }
            for (int ow = ow_end; ow < OW; ow++) {
                D_(n, c, oh, ow) = S_(n, c, IH-1, IW-1);
            }
        }
    }
}