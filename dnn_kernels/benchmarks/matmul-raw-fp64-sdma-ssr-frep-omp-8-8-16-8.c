#define P_B 8
#define P_M 8
#define P_K 16
#define P_N 8
#define OP_IMPL matmul_raw_fp64_sdma_ssr_frep_omp
#define DTYPE double
#include "templates/matmul_raw.tpl.c"