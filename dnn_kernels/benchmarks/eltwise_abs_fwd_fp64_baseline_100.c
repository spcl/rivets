
#include "eltwise.h"

#include <stdlib.h>
#include <math.h>
#include "printf.h"
#include "snrt.h"

double* volatile gs;
double* volatile gd;
double* volatile gds;
double* volatile gdd;
double* volatile grd;
double* volatile grds;

int main() {
    unsigned tid = snrt_cluster_core_idx();
    unsigned ntd = snrt_cluster_core_num();

    if (tid == 0) {
        gs = (double*)malloc(100 * sizeof(double));
        gd = (double*)malloc(100 * sizeof(double));
        gds = (double*)malloc(100 * sizeof(double));
        gdd = (double*)malloc(100 * sizeof(double));
        grd = (double*)malloc(100 * sizeof(double));
        grds = (double*)malloc(100 * sizeof(double));
    }

    snrt_cluster_hw_barrier();
    double* s = gs;
    double* d = gd;
    double* ds = gds;
    double* dd = gdd;
    double* rd = grd;
    double* rds = grds;

    double alpha = 2.0;
    double beta = 3.0;

    if (tid == 0) {
        printf("Execution started!\n");
        for (int i = 0; i < 100; i++) {
            s[i] = rand() * 1.0 / RAND_MAX - 0.5;
            d[i] = rand() * 1.0 / RAND_MAX - 0.5;
            ds[i] = rand() * 1.0 / RAND_MAX - 0.5;
            dd[i] = rand() * 1.0 / RAND_MAX - 0.5;
            rd[i] = d[i];
            rds[i] = ds[i];
        }
    }

    unsigned long t1 = read_csr(mcycle);
    if (tid == 0) eltwise_abs_fwd_fp64_baseline(d, s, 100);
    unsigned long t2 = read_csr(mcycle);

    if (tid == 0) {
        printf("Cycles: %lu\n", t2 - t1);

        printf("Running reference implementation...\n");
        eltwise_abs_fwd_fp64_baseline(rd, s, 100);
        printf("Running reference implementation... Done\n");

        printf("Verifying result...\n");
        int err = 0;
        for (int i = 0; i < 100; i++) {
            if (fabs(d[i] - rd[i]) > 1e-3) {
                err = 1;
            }
            if (fabs(ds[i] - rds[i]) > 1e-3) {
                err = 1;
            }
        }
        printf("Verifying result... Done\n");
        printf("Err: %d\n", err);
        return err;
    }

    return 0;
}
