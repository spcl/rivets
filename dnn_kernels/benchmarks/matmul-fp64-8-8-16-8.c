#define P_B 8
#define P_M 8
#define P_K 16
#define P_N 8
#define OP_IMPL matmul_fp64
#define DTYPE double
#include "templates/matmul.tpl.c"