#define P_B 8
#define P_M 128
#define P_K 128
#define P_N 128
#define OP_IMPL matmul_raw_fp64_sdma_ssr_frep_omp
#define DTYPE double
#include "templates/matmul_raw.tpl.c"