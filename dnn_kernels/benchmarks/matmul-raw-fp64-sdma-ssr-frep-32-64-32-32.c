#define P_B 32
#define P_M 64
#define P_K 32
#define P_N 32
#define OP_IMPL matmul_raw_fp64_sdma_ssr_frep
#define DTYPE double
#include "templates/matmul_raw.tpl.c"