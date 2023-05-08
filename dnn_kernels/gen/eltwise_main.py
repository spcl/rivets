import argparse
import sys
from textwrap import dedent
import os

from eltwise_kernel import get_kernel_name

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=100)
    parser.add_argument('--opt', type=str, default='baseline', choices=['baseline', 'snitch_singlecore'])
    parser.add_argument('--func', type=str, default='abs', choices=['abs', 'clip'])
    parser.add_argument('--bwd', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dtype', type=str, default='fp64', choices=['fp32', 'fp64'])
    return parser.parse_args()


def main(args):
    is_bwd = bool(args.bwd)

    dtype = {
        'fp32': 'float',
        'fp64': 'double',
    }[args.dtype]
    op_impl_baseline = get_kernel_name(args.func, is_bwd, args.dtype, 'baseline')
    op_impl = get_kernel_name(args.func, is_bwd, args.dtype, args.opt)
    data_size = args.n


    call_signature = {
        'abs': (
            f"d, s, {data_size}",
            f"dd, ds, s, {data_size}",
        ),
        'clip': (
            f"d, s, {data_size}, alpha, beta",
            f"dd, ds, s, {data_size}, alpha, beta",
        ),
    }[args.func][is_bwd]

    ref_call_signature = {
        'abs': (
            f"rd, s, {data_size}",
            f"dd, rds, s, {data_size}",
        ),
        'clip': (
            f"rd, s, {data_size}, alpha, beta",
            f"dd, rds, s, {data_size}, alpha, beta",
        ),
    }[args.func][is_bwd]

    code = dedent(f"""
        #include "eltwise.h"

        #include <stdlib.h>
        #include <math.h>
        #include "printf.h"
        #include "snrt.h"

        {dtype}* volatile gs;
        {dtype}* volatile gd;
        {dtype}* volatile gds;
        {dtype}* volatile gdd;
        {dtype}* volatile grd;
        {dtype}* volatile grds;

        int main() {{
            unsigned tid = snrt_cluster_core_idx();
            unsigned ntd = snrt_cluster_core_num();

            if (tid == 0) {{
                gs = ({dtype}*)malloc({data_size} * sizeof({dtype}));
                gd = ({dtype}*)malloc({data_size} * sizeof({dtype}));
                gds = ({dtype}*)malloc({data_size} * sizeof({dtype}));
                gdd = ({dtype}*)malloc({data_size} * sizeof({dtype}));
                grd = ({dtype}*)malloc({data_size} * sizeof({dtype}));
                grds = ({dtype}*)malloc({data_size} * sizeof({dtype}));
            }}

            snrt_cluster_hw_barrier();
            {dtype}* s = gs;
            {dtype}* d = gd;
            {dtype}* ds = gds;
            {dtype}* dd = gdd;
            {dtype}* rd = grd;
            {dtype}* rds = grds;

            double alpha = 2.0;
            double beta = 3.0;

            if (tid == 0) {{
                printf("Execution started!\\n");
                for (int i = 0; i < {data_size}; i++) {{
                    s[i] = rand() * 1.0 / RAND_MAX - 0.5;
                    d[i] = rand() * 1.0 / RAND_MAX - 0.5;
                    ds[i] = rand() * 1.0 / RAND_MAX - 0.5;
                    dd[i] = rand() * 1.0 / RAND_MAX - 0.5;
                    rd[i] = d[i];
                    rds[i] = ds[i];
                }}
            }}

            unsigned long t1 = read_csr(mcycle);
            if (tid == 0) {op_impl}({call_signature});
            unsigned long t2 = read_csr(mcycle);
            
            if (tid == 0) {{
                printf("Cycles: %lu\\n", t2 - t1);

                printf("Running reference implementation...\\n");
                {op_impl_baseline}({ref_call_signature});
                printf("Running reference implementation... Done\\n");

                printf("Verifying result...\\n");
                int err = 0;
                for (int i = 0; i < {data_size}; i++) {{
                    if (fabs(d[i] - rd[i]) > 1e-3) {{
                        err = 1;
                    }}
                    if (fabs(ds[i] - rds[i]) > 1e-3) {{
                        err = 1;
                    }}
                }}
                printf("Verifying result... Done\\n");
                printf("Err: %d\\n", err);
                return err;
            }}

            return 0;
        }}
    """)

    pass_name = 'bwd' if is_bwd else 'fwd'
    kernel_name = f"eltwise_{args.func}_{pass_name}_{args.dtype}_{args.opt}_{args.n}"

    fs = open(f'{root_dir}/benchmarks/{kernel_name}.c', 'w')
    
    fs.write(code)

if __name__ == '__main__':

    main(get_args())
    
    