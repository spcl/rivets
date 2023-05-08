#pragma once

#include <stddef.h>

void shuffle(
    double* dst, double* src,
    size_t P, size_t N, size_t M, size_t Q, 
    size_t stride
);
