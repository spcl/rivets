import argparse
import sys
from textwrap import dedent, indent
import itertools
import os

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=f'{root_dir}/src/eltwise.c')
    parser.add_argument('--header', type=str, default=f'{root_dir}/include/eltwise.h')
    return parser.parse_args()


def gen_snitch_singlecore_kernel_body_fwd(func, dtype):
    innermost_body = '\n'.join({
        'abs': {
            'fp32': (
                '"frep.o %0, 1, 0, 0;"',
                '"fabs.f ft1, ft0;"',
                ':: [reps] "r"(tcdm_buf_elems - 1)',
                ': "ft0", "ft1", "ft2", "memory"',
            ),
            'fp64': (
                '"frep.o %0, 1, 0, 0;"',
                '"fabs.d ft1, ft0;"',
                ':: [reps] "r"(tcdm_buf_elems - 1)',
                ': "ft0", "ft1", "ft2", "memory"',
            ),
        },
        'clip': {
            'fp32': (
                '"frep.o %0, 2, 0, 0;"',
                '"fmin.f ft3, ft0, %[beta];"',
                '"fmax.f ft1, ft3, %[alpha];"',
                ':: [reps] "r"(tcdm_buf_elems - 1), [alpha] "f"(alpha), [beta] "f"(beta)',
                ': "ft0", "ft1", "ft2", "ft3", "memory"',
            ),
            'fp64': (
                '"frep.o %0, 2, 0, 0;"',
                '"fmin.d ft3, ft0, %[beta];"',
                '"fmax.d ft1, ft3, %[alpha];"',
                ':: [reps] "r"(tcdm_buf_elems - 1), [alpha] "f"(alpha), [beta] "f"(beta)',
                ': "ft0", "ft1", "ft2", "ft3", "memory"',
            ),
        },
    }[func][dtype])

    innermost_body = indent(innermost_body, 8 * ' ')

    kernel_body = dedent(f"""
        size_t tcdm_buf_elems = 2000;

        unsigned tid = snrt_cluster_core_idx();
        unsigned ntd = 8 /*snrt_cluster_core_num()*/;

        //uint32_t hw_bar_addr = snrt_hw_barrier_addr();

        if (tid == 0) {{
            g_buf = (double*) snrt_l1alloc(2 * tcdm_buf_elems * sizeof(double));
            if (!g_buf) {{
                printf("Error: failed to allocate scratchpad memory\\n");
                while (1) {{}}
                return;
            }}
        }}
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        double* buf0 = g_buf;
        double* buf1 = g_buf + tcdm_buf_elems;
        
        if (tid == 0) {{
            snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
        }}

        size_t j = 0;
        size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

        if (snrt_is_dm_core()) {{
            // copy data for the first iteration
            snrt_dma_start_1d(
                /* dst */ buf0,
                /* src */ &src[j],
                /* size */ elems_to_process * sizeof(double)
            );
            snrt_dma_wait_all();
        }}

        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        while (j < n) {{
            size_t last_elem = j + tcdm_buf_elems;
            if (last_elem > n) last_elem = n;
            elems_to_process = last_elem - j;

            if (snrt_is_dm_core()) {{
                // finish data movement for the previous iteration
                // check it is not the first iteration
                if (j != 0) {{
                    snrt_dma_start_1d(
                        /* dst */ &dst[j - tcdm_buf_elems],
                        /* src */ buf1,
                        /* size */ tcdm_buf_elems * sizeof(double)
                    );
                    snrt_dma_wait_all();
                }}
            }}

            if (snrt_is_dm_core()) {{
                // start data movement for the next iteration
                // check it is not the last iteration
                if (j + elems_to_process != n) {{
                    size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                    snrt_dma_start_1d(
                        /* dst */ buf1,
                        /* src */ &src[j + tcdm_buf_elems],
                        /* size */ elems * sizeof(double)
                    );
                    snrt_dma_wait_all();
                }}
            }}

            if (tid == 0) {{

                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf0);
                snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf0);

                snrt_ssr_enable();
                asm volatile(
                    {innermost_body}
                );
                __builtin_ssr_barrier(SNRT_SSR_DM1);
                snrt_ssr_disable();
            }}
            
            // sync compute and data movement cores
            snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

            // swap current and next buffers
            double* tmp_buf = buf0;
            buf0 = buf1;
            buf1 = tmp_buf;

            j += elems_to_process;
        }}

        if (snrt_is_dm_core()) {{
            snrt_dma_start_1d(
                /* dst */ &dst[j - elems_to_process],
                /* src */ buf1,
                /* size */ elems_to_process * sizeof(double)
            );
            snrt_dma_wait_all();
        }}

        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

    """)

    return kernel_body


def gen_snitch_singlecore_kernel_body_bwd(func, dtype):
    innermost_body = '\n'.join({
        'abs': {
            # d_src[i] = (src[i] > 0) ? d_dst[i] : -d_dst[i];
            'fp32': (
                '"frep.o %0, 1, 0, 0;"',
                '"fsgnjx.f ft1, ft0, ft2;"',
                ':: [reps] "r"(tcdm_buf_elems - 1)',
                ': "ft0", "ft1", "ft2", "memory"',
            ),
            'fp64': (
                '"frep.o %0, 1, 0, 0;"',
                '"fsgnjx.d ft1, ft0, ft2;"',
                ':: [reps] "r"(tcdm_buf_elems - 1)',
                ': "ft0", "ft1", "ft2", "memory"',
            ),
        },
        'clip': {
            # d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;
            'fp32': (
                '"frep.o %0, 5, 0, 0;"',
                '"fsub.s ft3,ft2,%[a];"',
                '"fsub.s ft4,%[b],ft2;"',
                '"fmul.s ft3,ft3,ft4;"',
                '"fsgnj.s ft3,%[two],ft3;"',
                '"fmadd.s ft1,ft0,ft3,%[minus_one];"',
                ':: [reps] "r"(tcdm_buf_elems - 1), [a] "f"(alpha), [b] "f"(beta), [two] "f"(2.), [minus_one] "f"(-1.)',
                ': "ft0", "ft1", "ft2", "ft3", "ft4", "t0", "memory"',
            ),
            'fp64': (
                '"frep.o %0, 5, 0, 0;"',
                '"fsub.d ft3,ft2,%[a];"',
                '"fsub.d ft4,%[b],ft2;"',
                '"fmul.d ft3,ft3,ft4;"',
                '"fsgnj.d ft3,%[two],ft3;"',
                '"fmadd.d ft1,ft0,ft3,%[minus_one];"',
                ':: [reps] "r"(tcdm_buf_elems - 1), [a] "f"(alpha), [b] "f"(beta), [two] "f"(2.), [minus_one] "f"(-1.)',
                ': "ft0", "ft1", "ft2", "ft3", "ft4", "t0", "memory"',
            ),
        },
    }[func][dtype])

    innermost_body = indent(innermost_body, 8 * ' ')

    src_reps = {
        'abs': 1,
        'clip': 2,
    }[func]

    kernel_body = dedent(f"""
        size_t tcdm_buf_elems = 2000;

        unsigned tid = snrt_cluster_core_idx();
        unsigned ntd = 8 /*snrt_cluster_core_num()*/;

        uint32_t hw_bar_addr = snrt_hw_barrier_addr();

        if (tid == 0) {{
            g_buf = (double*) snrt_l1alloc(4 * tcdm_buf_elems * sizeof(double));
            if (!g_buf) {{
                printf("Error: failed to allocate scratchpad memory\\n");
                while (1) {{}}
                return;
            }}
        }}
        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        double* buf_grad_0 = g_buf;
        double* buf_grad_1 = buf_grad_0 + tcdm_buf_elems;
        double* buf_src_0 = buf_grad_1 + tcdm_buf_elems;
        double* buf_src_1 = buf_src_0 + tcdm_buf_elems;
        
        if (tid == 0) {{
            snrt_ssr_loop_1d(SNRT_SSR_DM0, tcdm_buf_elems, sizeof(double));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, tcdm_buf_elems, sizeof(double));
            snrt_ssr_loop_1d(SNRT_SSR_DM2, tcdm_buf_elems, sizeof(double));
            snrt_ssr_repeat(SNRT_SSR_DM2, {src_reps});
        }}

        size_t j = 0;
        size_t elems_to_process = (n < tcdm_buf_elems) ? n : tcdm_buf_elems;

        if (snrt_is_dm_core()) {{
            // copy data for the first iteration
            snrt_dma_start_1d(
                /* dst */ buf_src_0,
                /* src */ &src[j],
                /* size */ elems_to_process * sizeof(double)
            );
            snrt_dma_start_1d(
                /* dst */ buf_grad_0,
                /* src */ &d_dst[j],
                /* size */ elems_to_process * sizeof(double)
            );
            snrt_dma_wait_all();
        }}

        snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

        while (j < n) {{
            size_t last_elem = j + tcdm_buf_elems;
            if (last_elem > n) last_elem = n;
            elems_to_process = last_elem - j;

            if (snrt_is_dm_core()) {{
                // finish data movement for the previous iteration
                // check it is not the first iteration
                if (j != 0) {{
                    snrt_dma_start_1d(
                        /* dst */ &d_src[j - tcdm_buf_elems],
                        /* src */ buf_grad_1,
                        /* size */ tcdm_buf_elems * sizeof(double)
                    );
                    snrt_dma_wait_all();
                }}
            }}

            if (snrt_is_dm_core()) {{
                // start data movement for the next iteration
                // check it is not the last iteration
                if (j + elems_to_process != n) {{
                    size_t elems = (n < j + 2 * tcdm_buf_elems) ? (n - (j + tcdm_buf_elems)) : tcdm_buf_elems;
                    snrt_dma_start_1d(
                        /* dst */ buf_grad_1,
                        /* src */ &d_dst[j + tcdm_buf_elems],
                        /* size */ elems * sizeof(double)
                    );
                    snrt_dma_start_1d(
                        /* dst */ buf_src_1,
                        /* src */ &src[j + tcdm_buf_elems],
                        /* size */ elems * sizeof(double)
                    );
                    snrt_dma_wait_all();
                }}
            }}

            if (tid == 0) {{

                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, buf_grad_0);
                snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, buf_grad_0);
                snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, buf_src_0);

                snrt_ssr_enable();
                asm volatile(
                    {innermost_body}
                );
                __builtin_ssr_barrier(SNRT_SSR_DM1);
                snrt_ssr_disable();
            }}
            
            // sync compute and data movement cores
            snrt_cluster_hw_barrier(); //snrt_use_hw_barrier(hw_bar_addr);

            // swap current and next buffers
            double* tmp_buf = buf_grad_0;
            buf_grad_0 = buf_grad_1;
            buf_grad_1 = tmp_buf;

            tmp_buf = buf_src_0;
            buf_src_0 = buf_src_1;
            buf_src_1 = tmp_buf;

            j += elems_to_process;
        }}

        if (snrt_is_dm_core()) {{
            snrt_dma_start_1d(
                /* dst */ &d_src[j - elems_to_process],
                /* src */ buf_grad_1,
                /* size */ elems_to_process * sizeof(double)
            );
            snrt_dma_wait_all();
        }}

        snrt_use_hw_barrier(hw_bar_addr);

    """)

    return kernel_body


def get_kernel_name(func, is_bwd, dtype, opt):
    pass_name = 'bwd' if is_bwd else 'fwd'
    kernel_name = f"eltwise_{func}_{pass_name}_{dtype}_{opt}"
    return kernel_name


def get_ctype(dtype):
    ctype = {
        'fp32': 'float',
        'fp64': 'double',
    }[dtype]
    return ctype


def get_kernel_args(func, is_bwd, dtype):
    ctype = get_ctype(dtype)
    args = {
        'abs': (
            f"{ctype}* dst, {ctype}* src, size_t n",
            f"{ctype}* d_dst, {ctype}* d_src, {ctype}* src, size_t n",
        ),
        'clip': (
            f"{ctype}* dst, {ctype}* src, size_t n, {ctype} alpha, {ctype} beta",
            f"{ctype}* d_dst, {ctype}* d_src, {ctype}* src, size_t n, {ctype} alpha, {ctype} beta",
        ),
    }[func][is_bwd]
    return args


def gen_kernel(opt, func, is_bwd, dtype):
    ctype = get_ctype(dtype)

    args = get_kernel_args(func, is_bwd, dtype)

    kernel_name = get_kernel_name(func, is_bwd, dtype, opt)

    if opt == 'baseline':
        innermost_body = {
            'abs': {
                'fp32': (
                    f"dst[i] = fabsf(src[i]);",
                    f"d_src[i] = (src[i] > 0) ? d_dst[i] : -d_dst[i];",
                ),
                'fp64': (
                    f"dst[i] = fabs(src[i]);",
                    f"d_src[i] = (src[i] > 0) ? d_dst[i] : -d_dst[i];",
                ),
            },
            'clip': {
                'fp32': (
                    f"dst[i] = fmaxf(fminf(src[i], beta), alpha);",
                    f"d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;",
                ),
                'fp64': (
                    f"dst[i] = fmax(fmin(src[i], beta), alpha);",
                    f"d_src[i] = (alpha < src[i] && src[i] < beta) ? d_dst[i] : 0;",
                ),
            },
        }[func][dtype][is_bwd]

        kernel_body = dedent(f"""
            for (size_t i = 0; i < n; i++) {{
                {innermost_body}
            }}
        """)
    elif opt == 'snitch_singlecore':
        if is_bwd:
            kernel_body = gen_snitch_singlecore_kernel_body_bwd(func, dtype)
        else:
            kernel_body = gen_snitch_singlecore_kernel_body_fwd(func, dtype)
    else:
        raise NotImplementedError()

    kernel_body = indent(kernel_body, 12 * ' ')

    header = dedent(f"""
        void {kernel_name}({args});
    """)

    source = dedent(f"""
        void {kernel_name}({args}) {{
        {kernel_body}
        }}
    """)

    return source, header


if __name__ == '__main__':
    args = get_args()

    opts = ['baseline', 'snitch_singlecore']#, 'openmp', , 'snitch_multicore']
    funcs = ['abs', 'clip']
    is_bwd = [False, True]
    dtypes = ['fp32', 'fp64']

    fs = open(args.source, 'w')
    fh = open(args.header, 'w')

    header_start = dedent(f"""
        #pragma once
        #include <stddef.h>
    """)
    source_start = dedent(f"""
        #include "eltwise.h"

        #include <math.h>

        #include "snrt.h"
        #include "omp.h"
        #include "dm.h"
        #include "printf.h"

        static double* volatile g_buf;
    """)

    fs.write(source_start)
    fh.write(header_start)

    for o, f, p, d in itertools.product(opts, funcs, is_bwd, dtypes):
        source, header = gen_kernel(o, f, p, d)
        fh.write(header)
        fs.write(source)