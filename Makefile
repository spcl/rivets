DOCKER ?= podman
DOCKER_OPTS ?= --security-opt label=disable
DOCKER_RUN ?= $(DOCKER) run $(DOCKER_OPTS)


.SECONDEXPANSION:
#https://www.gnu.org/software/make/manual/make.html#Secondary-Expansion


.PHONY: clean
clean:
	rm -rf snRuntime-build logs dnn_kernels/build-cluster dnn_kernels/build-banshee


snRuntime-build/libsnRuntime-cluster.a snRuntime-build/libsnRuntime-banshee.a:
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		ghcr.io/pulp-platform/snitch \
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
		ghcr.io/pulp-platform/snitch \
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


# dnn_kernels/build-{banshee,cluster}/kernel-arg1-arg2-arg3-...
# dnn_kernels/build-<platform>/<kernel>
dnn_kernels/%: snRuntime-build/libsnRuntime-$$(word 3, $$(subst -, , $$(subst /, , $$@))).a
	PLATFORM=$(word 3, $(subst -, , $(subst /, , $@))) && \
	KERNEL=$(word 3, $(subst /, , $@)) && \
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		ghcr.io/pulp-platform/snitch \
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
			cd /repo/dnn_kernels && rm -rf build-banshee build-cluster && mkdir build-banshee && mkdir build-cluster && \
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
		ghcr.io/pulp-platform/snitch /bin/bash -c "\
		mkdir /workspace && cd /workspace && \
		git clone https://github.com/pulp-platform/snitch.git && \
		cd snitch/hw/system/snitch_cluster && \
		git checkout ed24b24 && \
		make bin/snitch_cluster.vlt && \
		cp bin/snitch_cluster.vlt /repo/snitch_cluster.vlt"


# make verilator-dnn-abs-raw-fp64-sdma-ssr-frep-omp-10000-bench
verilator-dnn-%: dnn_kernels/build-cluster/% snitch_cluster.vlt
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		dmlsn \
		/bin/bash -c "\
			./snitch_cluster.vlt \
			/repo/dnn_kernels/build-cluster/$*"


# make banshee-dnn-abs-raw-fp64-sdma-ssr-frep-omp-10000-bench
banshee-dnn-%: dnn_kernels/build-banshee/%
	$(DOCKER_RUN) -it \
		-v `pwd`:/repo \
		-w /repo \
		ghcr.io/pulp-platform/snitch \
		/bin/bash -c "\
			RUST_MIN_STACK=134217728 \
			SNITCH_LOG= \
			banshee \
				--configuration /repo/snitch/sw/banshee/config/snitch_cluster.yaml \
				--latency \
				/repo/dnn_kernels/build-banshee/$*"

