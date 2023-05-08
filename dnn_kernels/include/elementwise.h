#pragma once

#include <stddef.h>

void eltwise_abs(float* dst, float* src, size_t n);
void eltwise_abs_bwd(float* d_dst, float* d_src, size_t n);

void eltwise_abs_sdma(float* dst, float* src, size_t n);
void eltwise_abs_sdma_ssr(float* dst, float* src, size_t n);
void eltwise_abs_sdma_ssr_frep(float* dst, float* src, size_t n);
void eltwise_abs_sdma_ssr_frep_omp(float* dst, float* src, size_t n);

void eltwise_abs_fp64_sdma(double* dst, double* src, size_t n);
void eltwise_abs_fp64_sdma_ssr(double* dst, double* src, size_t n);
void eltwise_abs_fp64_sdma_ssr_frep(double* dst, double* src, size_t n);
void eltwise_abs_raw_fp64_sdma_ssr_frep(double* dst, double* src, size_t n);
void eltwise_abs_fp64_sdma_ssr_frep_omp(double* dst, double* src, size_t n);
void eltwise_abs_raw_fp64_sdma_ssr_frep_omp(double* dst, double* src, size_t n);

void eltwise_clip(float* dst, float* src, size_t n, float alpha, float beta);
void eltwise_clip_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta);

void eltwise_elu(float* dst, float* src, size_t n, float alpha);
void eltwise_elu_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha);

void eltwise_exp(float* dst, float* src, size_t n);
void eltwise_exp_bwd(float* d_dst, float* d_src, float* src, size_t n);

void eltwise_gelu_erf(float* dst, float* src, size_t n, float alpha);
void eltwise_gelu_erf_bwd(float* d_dst, float* d_src, float* src, size_t n, float alpha);
