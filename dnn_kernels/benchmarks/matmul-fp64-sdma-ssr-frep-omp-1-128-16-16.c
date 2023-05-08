#define P_B 1
#define P_M 128
#define P_K 16
#define P_N 16
#define OP_IMPL matmul_fp64_sdma_ssr_frep_omp
#define DTYPE double
#include "templates/matmul.tpl.c"