#define P_B 256
#define P_N 256
#define OP_IMPL layer_norm_raw_fp64_sdma_ssr_frep_omp
#define OP_IMPL_DM layer_norm_raw_dm_fp64_sdma_ssr_frep
#define DTYPE double
#include "templates/layernorm_raw.tpl.c"