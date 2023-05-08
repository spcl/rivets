
#pragma once
#include <stddef.h>

void eltwise_abs_fwd_fp32_baseline(float* dst, float* src, size_t n);

void eltwise_abs_fwd_fp64_baseline(double* dst, double* src, size_t n);

void eltwise_abs_bwd_fp32_baseline(float* d_dst, float* d_src, float* src, size_t n);

void eltwise_abs_bwd_fp64_baseline(double* d_dst, double* d_src, double* src, size_t n);

void eltwise_clip_fwd_fp32_baseline(float* dst, float* src, size_t n, float alpha, float beta);

void eltwise_clip_fwd_fp64_baseline(double* dst, double* src, size_t n, double alpha, double beta);

void eltwise_clip_bwd_fp32_baseline(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta);

void eltwise_clip_bwd_fp64_baseline(double* d_dst, double* d_src, double* src, size_t n, double alpha, double beta);

void eltwise_abs_fwd_fp32_snitch_singlecore(float* dst, float* src, size_t n);

void eltwise_abs_fwd_fp64_snitch_singlecore(double* dst, double* src, size_t n);

void eltwise_abs_bwd_fp32_snitch_singlecore(float* d_dst, float* d_src, float* src, size_t n);

void eltwise_abs_bwd_fp64_snitch_singlecore(double* d_dst, double* d_src, double* src, size_t n);

void eltwise_clip_fwd_fp32_snitch_singlecore(float* dst, float* src, size_t n, float alpha, float beta);

void eltwise_clip_fwd_fp64_snitch_singlecore(double* dst, double* src, size_t n, double alpha, double beta);

void eltwise_clip_bwd_fp32_snitch_singlecore(float* d_dst, float* d_src, float* src, size_t n, float alpha, float beta);

void eltwise_clip_bwd_fp64_snitch_singlecore(double* d_dst, double* d_src, double* src, size_t n, double alpha, double beta);
