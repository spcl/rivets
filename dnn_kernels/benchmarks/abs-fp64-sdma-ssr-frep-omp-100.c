#define DATA_SIZE 100
#define OP_IMPL eltwise_abs_fp64_sdma_ssr_frep_omp
#define DTYPE double
#include "templates/abs.tpl.c"