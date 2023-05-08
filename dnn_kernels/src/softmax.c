#include "softmax.h"

#include <float.h>
#include <math.h>

#define D_(o, c, i) dst[(o) * stride_dst_out + (c) * stride_dst_compute + (i) * stride_dst_in]
#define S_(o, c, i) src[(o) * stride_src_out + (c) * stride_src_compute + (i) * stride_src_in]

void softmax(
    float* dst, float* src,
    size_t O, size_t C, size_t I,
    size_t stride_dst_out, size_t stride_dst_compute, size_t stride_dst_in,
    size_t stride_src_out, size_t stride_src_compute, size_t stride_src_in
) {
    for (size_t o = 0; o < O; o++) {
        for (size_t i = 0; i < I; i++) {
            float max_c = - FLT_MAX;
            for (size_t c = 0; c < C; c++) {
                max_c = fmaxf(S_(o, c, i), max_c);
            }
            float sum_c = 0;
            for (size_t c = 0; c < C; c++) {
                float diff = expf(S_(o, c, i) - max_c);
                D_(o, c, i) = diff;
                sum_c += diff;  
            }
            for (size_t c = 0; c < C; c++) {
                D_(o, c, i) /= sum_c;
            }
        }
    }
}