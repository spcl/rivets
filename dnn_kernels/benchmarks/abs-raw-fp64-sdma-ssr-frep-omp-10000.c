#define DATA_SIZE 10000
#define OP_IMPL eltwise_abs_raw_fp64_sdma_ssr_frep_omp
#define DTYPE double
#include "templates/abs_raw.tpl.c"