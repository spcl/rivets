BUILD_DIR ?= build
OBJDUMP ?= objdump

SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src%,$(BUILD_DIR)%,$(SRCS:.c=.o))
BENCH_SRCS = $(wildcard benchmarks/*.c)
BENCH_OBJS = $(patsubst benchmarks%,$(BUILD_DIR)%,$(BENCH_SRCS:.c=-bench.o))
BENCH_BINS = $(patsubst benchmarks%,$(BUILD_DIR)%,$(BENCH_SRCS:.c=-bench))
BENCH_DUMPS = $(BENCH_BINS:%=%.s)

.PHONY: all
all: $(BENCH_DUMPS) $(BENCH_BINS) $(BUILD_DIR)/libDNNKernels.a


$(BUILD_DIR)/%-bench.s: $(BUILD_DIR)/%-bench
	$(OBJDUMP) -dhS $^ > $@


$(BUILD_DIR)/%-bench.o: benchmarks/%.c
	$(CC) $(CFLAGS) -c $< -I include -o $@


$(BUILD_DIR)/%-bench: $(BUILD_DIR)/%-bench.o $(BUILD_DIR)/libDNNKernels.a
	$(CC) $(LDFLAGS) $^ -I include -o $@


$(BUILD_DIR)/libDNNKernels.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	

$(BUILD_DIR)/%.o: src/%.c include/%.h
	$(CC) $(CFLAGS) -c $< -I include -o $@


.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*
