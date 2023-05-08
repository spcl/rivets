#define P_B 16
#define P_M 16
#define P_K 16
#define P_N 16
#define OP_IMPL matmul_raw_fp64_sdma_ssr_frep
#define DTYPE double
#include "templates/matmul_raw.tpl.c"