#pragma once

#include <stddef.h>

// dst[b,m,n] = sum_k(src[b,m,k] * weight[b,k,n]) + bias[b,m,n]
void matmul(
    float* dst, float* src, float* weight, float* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_fp64(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_fp64_sdma(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_fp64_sdma_ssr(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_fp64_sdma_ssr_frep(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_raw_fp64_sdma_ssr_frep(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);

void matmul_raw_fp64_sdma_ssr_frep_omp(
    double* dst, double* src, double* weight, double* bias,
    size_t B, size_t M, size_t K, size_t N,
    size_t stride_dst_b, size_t stride_dst_m, size_t stride_dst_n,
    size_t stride_src_b, size_t stride_src_m, size_t stride_src_k,
    size_t stride_weight_b, size_t stride_weight_k, size_t stride_weight_n,
    size_t stride_bias_b, size_t stride_bias_m, size_t stride_bias_n
);