#include "shuffle.h"

#define D_(p, m, n, q) dst[((p) * N * M + (m) * N + (n)) * stride + (q)]
#define S_(p, n, m, q) src[((p) * N * M + (n) * M + (m)) * stride + (q)]

// dst[p,m,n,q] = src[p,n,m,q]
void shuffle(
    double* dst, double* src,
    size_t P, size_t N, size_t M, size_t Q, 
    size_t stride
) {
    for (size_t p = 0; p < P; p++)
    for (size_t n = 0; n < N; n++)
    for (size_t m = 0; m < M; m++)
    for (size_t q = 0; q < Q; q++) {
        D_(p, m, n, q) = S_(p, n, m, q);
    }
}

