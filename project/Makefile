CUDA_PATH := /usr/local/cuda

CC := clang
NVCC := nvcc

COMMON_FLAGS := \
	-O0 \
	-Wall -Wextra -Wpedantic \
        -Werror -Wnoerror=unused-parameter
LD_FLAGS := -lm

CFLAGS := $(COMMON_FLAGS) -std=c23
NVCC_FLAGS := $(COMMON_FLAGS) -L$(CUDA_PATH)/lib64 -lcudart

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

ENC_TARGET := $(BUILD_DIR)/c63enc
DEC_TARGET := $(BUILD_DIR)/c63dec


ENC_SRCS := $(SRC_DIR)/c63enc.c
DEC_SRCS := $(SRC_DIR)/c63dec.c

C_SRCS := $(wildcard $(SRC_DIR)/*.c)
CUDA_SRCS := $(wildcard $(SRC_DIR)/*.cu)
COMMON_C_SRCS := $(filter-out $(ENC_SRCS) $(DEC_SRCS), $(C_SRCS))
COMMON_CUDA_SRCS := $(filter-out $(ENC_SRCS) $(DEC_SRCS), $(CUDA_SRCS))

ENC_C_OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(ENC_SRCS))
DEC_C_OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(DEC_SRCS))
COMMON_C_OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(COMMON_C_SRCS))
CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRCS))


all: $(ENC_TARGET) $(DEC_TARGET)

$(ENC_TARGET): $(ENC_C_OBJS) $(COMMON_C_OBJS) $(CUDA_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(LD_FLAGS) -o $@ $^

$(DEC_TARGET): $(DEC_OBJS) $(COMMON_C_OBJS) $(CUDA_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(LD_FLAGS) -o $@ $^


$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(ENC_TARGET) $(DEC_TARGET)

.PHONY: all clean
