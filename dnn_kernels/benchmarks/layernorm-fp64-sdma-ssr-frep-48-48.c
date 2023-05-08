#define P_B 48
#define P_N 48
#define OP_IMPL layer_norm_fp64_sdma_ssr_frep
#define DTYPE double
#include "templates/layernorm.tpl.c"