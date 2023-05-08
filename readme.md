# RIVETS: An Efficient Training and Inference Library for RISC-V with Snitch Extensions

### Quick start

Run simulation in banshee. Suitable for correctness checks but not cycle-accurate.

```
make banshee-dnn-abs-raw-fp64-sdma-ssr-frep-omp-10000-bench
```

Run simulation with verilator: cycle-accurate but slow.

```
make verialtor-dnn-abs-raw-fp64-sdma-ssr-frep-omp-10000-bench
```

See the list of all available targets in [dnn_kernels/benchmarks](dnn_kernels/benchmarks).
Kernels has the following names `{kernel}.c`. Targets to run simulation are `verilator-dnn-{kernel}-bench` and `banshee-dnn-{kernel}-bench`.