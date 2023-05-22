# RIVETS: An Efficient Training and Inference Library for RISC-V with Snitch Extensions

### Quick start

Run simulation in banshee. Suitable for correctness checks but not cycle-accurate.

```
make banshee-dnn-matmul-8-64-32-32-matmul_raw_fp64_sdma_ssr_frep_omp-double-bench
make banshee-dnn-layernorm-64-64-layer_norm_raw_fp64_sdma_ssr_frep-double-bench
make banshee-dnn-abs-10000-eltwise_abs_raw_fp64_sdma_ssr_frep_omp-double-bench
```

Run simulation with verilator: cycle-accurate but slow.

```
make verilator-dnn-matmul-8-64-32-32-matmul_raw_fp64_sdma_ssr_frep_omp-double-bench
make verilator-dnn-layernorm-64-64-layer_norm_raw_fp64_sdma_ssr_frep-double-bench
make verilator-dnn-abs-10000-eltwise_abs_raw_fp64_sdma_ssr_frep_omp-double-bench
```
