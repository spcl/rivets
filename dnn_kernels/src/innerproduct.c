#include "innerproduct.h"

#include "matmul.h"

#define D_(n,oc) dst[(n) * stride_dst_N + (oc) * stride_dst_OC]
#define S_(n,ic) src[(n) * stride_src_N + (ic) * stride_src_IC]
#define W_(oc,ic) weight[(oc) * stride_weight_OC + (ic) * stride_weight_IC]

// dst[n,oc] = bias[oc] + sum_ic (src[n,ic] * weight[oc,ic])
void innerproduct_fp64(
    double* dst, double* src, double* weight, double* bias,
    size_t N, size_t OC, size_t IC,
    size_t stride_dst_N, size_t stride_dst_OC,
    size_t stride_src_N, size_t stride_src_IC,
    size_t stride_weight_OC, size_t stride_weight_IC
) {
    for (size_t oc = 0; oc < OC; oc++) {
        for (size_t n = 0; n < N; n++) {
            D_(n, oc) = bias[oc];
        }
    }
    matmul_fp64(
        dst, src, weight, dst,
        1, N, IC, OC,
        1, stride_dst_N, stride_dst_OC,
        1, stride_src_N, stride_src_IC,
        1, stride_weight_OC, stride_weight_IC,
        1, stride_dst_N, stride_dst_OC
    );
}