#define DATA_SIZE 40000
#define OP_IMPL eltwise_abs_raw_fp64_sdma_ssr_frep
#define DTYPE double
#include "templates/abs_raw.tpl.c"