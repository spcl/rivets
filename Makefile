DOCKER ?= podman
DOCKER_OPTS ?= --security-opt label=disable --rm
DOCKER_RUN ?= $(DOCKER) run $(DOCKER_OPTS)
DOCKER_IMG ?= ghcr.io/pulp-platform/snitch@sha256:f43d2db7c97bdcd653deb567664032940e8d9051a80a52c22f239e19fe4c310b


.SECONDARY: 
# no target is removed because it is considered intermediate

.SECONDEXPANSION:
#https://www.gnu.org/software/make/manual/make.html#Secondary-Expansion


.PHONY: clean
clean:
	rm -rf snRuntime-build logs dnn_kernels/build-cluster dnn_kernels/build-banshee


snRuntime-build/libsnRuntime-cluster.a snRuntime-build/libsnRuntime-banshee.a:
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			rm -rf /repo/snRuntime-build/ && \
			mkdir -p /repo/snRuntime-build && \
			cd /repo/snRuntime-build && \
			cmake /repo/snitch/sw/snRuntime \
				-DSNITCH_BANSHEE=/repo/banshee \
				-DSNITCH_SIMULATOR=/repo/snitch_cluster.vlt \
				-DBUILD_TESTS=ON \
				-DSNITCH_RUNTIME=snRuntime-cluster \
				-DCMAKE_TOOLCHAIN_FILE=toolchain-llvm && \
			cmake --build . -j `nproc`"


dnn_kernels/build-%/libDNNKernels.a: snRuntime-build/libsnRuntime-%.a
	export SNITCH_PLATFORM=$(@:dnn_kernels/build-%/libDNNKernels.a=%) && \
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			export CC=/tools/riscv-llvm/bin/clang && \
			export CFLAGS=\" \
				-g -O3 \
				-mcpu=snitch -mcmodel=medany -ffast-math \
				-flto \
				-fno-builtin-printf -fno-common -ffunction-sections \
				-static -mllvm -enable-misched=false -mno-relax \
				-fopenmp -menable-experimental-extensions \
				-isystem /tools/riscv-llvm/riscv32-unknown-elf/include/ \
				-isystem /tools/riscv-llvm/lib/clang/12.0.1/include/ \
				-isystem /repo/snitch/sw/snRuntime/vendor/ \
				-isystem /repo/snitch/sw/snRuntime/include/ \
				-isystem /repo/snitch/sw/snRuntime/vendor/riscv-opcodes/ \
				\" && \
			export AR=/tools/riscv-llvm/bin/llvm-ar && \
			export RANLIB=/tools/riscv-llvm/bin/llvm-ranlib && \
			cd /repo/dnn_kernels && rm -rf build-$$SNITCH_PLATFORM && mkdir build-$$SNITCH_PLATFORM && \
			export LDFLAGS=\" \
				-flto \
				-mcpu=snitch -nostartfiles -fuse-ld=lld -Wl,--image-base=0x80000000 \
				-nostdlib \
				-static \
				-Wl,-z,norelro \
				-Wl,--gc-sections \
				-Wl,--no-relax \
				-nodefaultlibs \
				-T /repo/snRuntime-build/common.ld \
				/tools/riscv-llvm/riscv32-unknown-elf/lib/libc.a \
				/tools/riscv-llvm/riscv32-unknown-elf/lib/libm.a \
				/tools/riscv-llvm/lib/clang/12.0.1/lib/libclang_rt.builtins-riscv32.a \
				/repo/snRuntime-build/libsnRuntime-$$SNITCH_PLATFORM.a \
				\" && \
			export OBJDUMP=\"/tools/riscv-llvm/bin/llvm-objdump --mcpu=snitch\" && \
			BUILD_DIR=build-$$SNITCH_PLATFORM make build-$$SNITCH_PLATFORM/libDNNKernels.a \
		"


get-platform = $(word 3, $(subst -, , $(subst /, , $@)))

# dnn_kernels/build-{banshee,cluster}/kernel-arg1-arg2-arg3-...
# dnn_kernels/build-<platform>/<kernel>
dnn_kernels/%: snRuntime-build/libsnRuntime-$$(get-platform).a
	PLATFORM=$(get-platform) && \
	KERNEL=$(word 3, $(subst /, , $@)) && \
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			export CC=/tools/riscv-llvm/bin/clang && \
			export CFLAGS=\" \
				-g -O3 \
				-mcpu=snitch -mcmodel=medany -ffast-math \
				-flto \
				-fno-builtin-printf -fno-common -ffunction-sections \
				-static -mllvm -enable-misched=false -mno-relax \
				-fopenmp -menable-experimental-extensions \
				-isystem /tools/riscv-llvm/riscv32-unknown-elf/include/ \
				-isystem /tools/riscv-llvm/lib/clang/12.0.1/include/ \
				-isystem /repo/snitch/sw/snRuntime/vendor/ \
				-isystem /repo/snitch/sw/snRuntime/include/ \
				-isystem /repo/snitch/sw/snRuntime/vendor/riscv-opcodes/ \
				\" && \
			export AR=/tools/riscv-llvm/bin/llvm-ar && \
			export RANLIB=/tools/riscv-llvm/bin/llvm-ranlib && \
			cd /repo/dnn_kernels && mkdir -p build-$$PLATFORM && \
			export LDFLAGS=\" \
				-flto \
				-mcpu=snitch -nostartfiles -fuse-ld=lld -Wl,--image-base=0x80000000 \
				-nostdlib \
				-static \
				-Wl,-z,norelro \
				-Wl,--gc-sections \
				-Wl,--no-relax \
				-nodefaultlibs \
				-T /repo/snRuntime-build/common.ld \
				/tools/riscv-llvm/riscv32-unknown-elf/lib/libc.a \
				/tools/riscv-llvm/riscv32-unknown-elf/lib/libm.a \
				/tools/riscv-llvm/lib/clang/12.0.1/lib/libclang_rt.builtins-riscv32.a \
				/repo/snRuntime-build/libsnRuntime-$$PLATFORM.a \
				\" && \
			export OBJDUMP=\"/tools/riscv-llvm/bin/llvm-objdump --mcpu=snitch\" && \
			BUILD_DIR=build-$$PLATFORM make build-$$PLATFORM/$$KERNEL && \
			echo target: $@ done"


snitch_cluster.vlt:
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) /bin/bash -c "\
		mkdir /workspace && cd /workspace && \
		git clone https://github.com/pulp-platform/snitch.git && \
		cd snitch/hw/system/snitch_cluster && \
		git checkout 20e9d26 && \
		make CFG_OVERRIDE=/repo/occamy_cluster.default.hjson bin/snitch_cluster.vlt && \
		cp bin/snitch_cluster.vlt /repo/snitch_cluster.vlt"


# make verilator-dnn-matmul-8-64-32-32-matmul_raw_fp64_sdma_ssr_frep_omp-double-bench
.PHONY: verilator-dnn-%
verilator-dnn-%: dnn_kernels/build-cluster/% snitch_cluster.vlt
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			./snitch_cluster.vlt \
			/repo/dnn_kernels/build-cluster/$*"


.PHONY: postprocess-logs
postprocess-logs:
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			for f in logs/trace_hart_*.dasm; do \
				(cat \$$f | spike-dasm > \$${f/dasm/txt}) && \
				(/repo/snitch/util/gen_trace.py \$${f/dasm/txt} > \$${f/dasm/trace}) && \
				echo \"Postprocessed \$$f -spike-dasm-> \$${f/dasm/txt} -gen_trace.py-> \$${f/dasm/trace}\"; \
			done"


# make banshee-dnn-matmul-8-64-32-32-matmul_raw_fp64_sdma_ssr_frep_omp-double-bench
# make banshee-dnn-layernorm-256-256-layer_norm_raw_fp64_sdma_ssr_frep-double-bench
# make banshee-dnn-abs-10000-eltwise_abs_raw_fp64_sdma_ssr_frep_omp-double-bench
.PHONY: banshee-dnn-%
banshee-dnn-%: dnn_kernels/build-banshee/%
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		$(DOCKER_IMG) \
		/bin/bash -c "\
			RUST_MIN_STACK=134217728 \
			SNITCH_LOG= \
			banshee \
				--configuration /repo/snitch/sw/banshee/config/snitch_cluster.yaml \
				--latency \
				/repo/dnn_kernels/build-banshee/$*"

