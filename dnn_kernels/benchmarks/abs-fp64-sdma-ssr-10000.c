#define DATA_SIZE 10000
#define OP_IMPL eltwise_abs_fp64_sdma_ssr
#define DTYPE double
#include "templates/abs.tpl.c"