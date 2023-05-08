#include "lrn.h"

#include <math.h>

#define D_(n, c, h, w) dst[(n) * stride_n + (c) * stride_c + (h) * W + (w)]
#define S_(n, c, h, w) src[(n) * stride_n + (c) * stride_c + (h) * W + (w)]
#define T_(n, c, h, w) tmp[(n) * stride_n + (c) * stride_c + (h) * W + (w)]

#define SQR(x) ((x) * (x))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


// tmp has the same shape as dst
void local_response_normalization_across_channels(
    double* dst, double* src, double* tmp,
    size_t N, size_t C, size_t H, size_t W, size_t L,
    size_t stride_n, size_t stride_c,
    double k, double alpha, double beta
) {
    double factor = alpha / L;
    for (size_t n = 0; n < N; n++)
    for (size_t c = 0; c < C; c++)
    for (size_t h = 0; h < H; h++)
    for (size_t w = 0; w < W; w++) {
        T_(n, c, h, w) = factor * SQR(S_(n, c, h, w));
    }

    size_t l_min = -(L - 1) / 2;
    size_t l_max = (L + 1) / 2;
    for (size_t n = 0; n < N; n++)
    for (size_t c = 0; c < C; c++)
    for (size_t h = 0; h < H; h++)
    for (size_t w = 0; w < W; w++) {
        double sum = k; 
        for (size_t i = MAX(0, c - l_min); i < MIN(C, c + l_max); i++) {
            sum += T_(n, c + i, h, w);
        }
        D_(n, c, h, w) = pow(sum, -beta) * S_(n, c, h, w);
    }
}
