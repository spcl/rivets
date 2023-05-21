BUILD_DIR ?= build
OBJDUMP ?= objdump


SHELL := /bin/bash
.SECONDEXPANSION:


SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src%,$(BUILD_DIR)%,$(SRCS:.c=.o))

.PHONY: all
all: $(BUILD_DIR)/libDNNKernels.a



get-tokens = $(subst -, , $*)


$(BUILD_DIR)/%-bench.o: benchmarks/$$(word 1, $$(get-tokens)).c
	BENCHSRC=$(word 1, $(get-tokens)) && \
	ARGS="$(wordlist 2, $(words $(get-tokens)), $(get-tokens))" && \
	ARR=($${ARGS}) && \
	PARAMS="$$(for idx in $${!ARR[@]}; do echo -n "-DARG$$idx=$${ARR[idx]} " ; done)" && \
	$(CC) $(CFLAGS) -c $< -I include $$PARAMS -o $@


$(BUILD_DIR)/%-bench: $(BUILD_DIR)/%-bench.o $(BUILD_DIR)/libDNNKernels.a
	$(CC) $(LDFLAGS) $^ -I include -o $@
	$(OBJDUMP) -dhS $@ > $@.s


$(BUILD_DIR)/libDNNKernels.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	

$(BUILD_DIR)/%.o: src/%.c include/%.h
	$(CC) $(CFLAGS) -c $< -I include -o $@


.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*
