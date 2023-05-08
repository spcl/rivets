#include "rnn.h"

#include <stdlib.h>
#include <math.h>

// B - batch size
// L - num layers
// S - sequence length
// HI - input size
// HO - output size
// dst_layer[B,S,HO] dst_iter[B,L,HO]
// src_layer[B,S,HI] src_iter[B,L,HO]
// weight_layer_0[L,HO,HI] weight_layer_1L[L,HO,HO]
// weight_iter[L,HO,HO] bias[L,HO]

#define DL_(b, s, ho) dst_layer[(b) * stride_dst_layer_B + (s) * stride_dst_layer_S + (ho)]
#define DI_(b, l, ho) dst_iter[(b) * stride_dst_iter_B + (l) * stride_dst_iter_L + (ho)]
#define TL_(b, s, ho) tmp_layer[(b) * S * HO + (s) * HO + (ho)]
#define TI_(b, l, ho) tmp_iter[(b) * L * HO + (l) * HO + (ho)]
#define SL_(b, s, hi) src_layer[(b) * stride_src_layer_B + (s) * stride_src_layer_S + (hi)]
#define SI_(b, l, ho) src_iter[(b) * stride_src_iter_B + (l) * stride_src_iter_L + (ho)]
#define WL0_(l, ho, hi) weight_layer_0[(l) * HO * HI + (ho) * HI + (hi)]
#define WL1L_(l, ho1, ho2) weight_layer_1L[(l) * HO * HO + (ho1) * HO + (ho2)]
#define WI_(l, ho1, ho2) weight_iter[(l) * HO * HO + (ho1) * HO + (ho2)]
#define B_(l, ho) bias[(l) * HO + (ho)]

void vanila_rnn_relu_fp64(
    double* dst_layer, double* dst_iter,
    double* src_layer, double* src_iter,
    double* weight_layer_0, double* weight_layer_1L,
    double* weight_iter, double* bias,
    size_t B, size_t L, size_t S, size_t HI, size_t HO,
    size_t stride_dst_layer_B, size_t stride_dst_layer_S,
    size_t stride_dst_iter_B, size_t stride_dst_iter_L,
    size_t stride_src_layer_B, size_t stride_src_layer_S,
    size_t stride_src_iter_B, size_t stride_src_iter_L
) {
    double* tmp_layer = (double*) malloc(B * S * HO * sizeof(double));
    double* tmp_iter = (double*) malloc(B * L * HO * sizeof(double));
    for (size_t b = 0; b < B; b++) {
        for (size_t l = 0; l < 1; l++) {
            for (size_t s = 0; s < S; s++) {
                for (size_t ho = 0; ho < HO; ho++) {
                    double res = B_(l, ho);
                    for (size_t hi = 0; hi < HI; hi++) {
                        res += SL_(b, s, hi) * WL0_(l, ho, hi);
                    }
                    for (size_t ho2 = 0; ho2 < HO; ho2++) {
                        res += SI_(b, s, ho2) * WI_(l, ho, ho2);
                    }
                    TL_(b, s, ho) = fmax(res, 0);
                }
                for (size_t ho = 0; ho < HO; ho++) {
                    DL_(b, s, ho) = TL_(b, s, ho);
                }
            }
            for (size_t ho = 0; ho < HO; ho++) {
                DI_(b, l, ho) = DL_(b, S-1, ho);
            }
        }
        for (size_t l = 1; l < L; l++) {
            for (size_t s = 0; s < S; s++) {
                for (size_t ho = 0; ho < HO; ho++) {
                    double res = B_(l, ho);
                    for (size_t ho2 = 0; ho2 < HO; ho2++) {
                        res += DL_(b, s, ho2) * WL1L_(l, ho, ho2);
                        res += DI_(b, s, ho2) * WI_(l, ho, ho2);
                    }
                    TL_(b, s, ho) = fmax(res, 0);
                }
                for (size_t ho = 0; ho < HO; ho++) {
                    DL_(b, s, ho) = TL_(b, s, ho);
                }
            }
            for (size_t ho = 0; ho < HO; ho++) {
                DI_(b, l, ho) = DL_(b, S-1, ho);
            }
        }
    }
}