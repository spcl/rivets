#define P_B 1
#define P_M 100
#define P_K 100
#define P_N 100
#define OP_IMPL matmul_fp64_sdma_ssr
#define DTYPE double
#include "templates/matmul.tpl.c"