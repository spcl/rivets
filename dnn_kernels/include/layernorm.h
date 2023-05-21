#pragma once

#include <stddef.h>

void layer_norm_fp32(
    float* dst, float* src, float* mu, float* gamma, float* sigma, float* beta, float eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_fp64(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_fp64_sdma(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_fp64_sdma_ssr(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_fp64_sdma_ssr_frep(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_raw_fp64_sdma_ssr_frep(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);

void layer_norm_raw_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* mu, double* gamma, double* sigma, double* beta, double eps,
    size_t n1, size_t n2, size_t s1, size_t s2
);