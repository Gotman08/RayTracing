# ==============================================================================
# CUDA Ray Tracer - Makefile for HPC environments (Romeo2025)
# ==============================================================================

# Compiler
NVCC := nvcc

# Project settings
PROJECT := raytracer
VERSION := 1.0.0

# Directories
SRC_DIR := src
INC_DIR := include
EXT_DIR := external
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

# CUDA Architecture
# GH200/H100: sm_90, A100: sm_80, V100: sm_70
CUDA_ARCH ?= sm_90

# Flags
NVCC_FLAGS := -std=c++17 \
              -arch=$(CUDA_ARCH) \
              --expt-relaxed-constexpr \
              --extended-lambda \
              -lineinfo \
              --use_fast_math \
              -Xcompiler="-Wall -O3"

NVCC_FLAGS_DEBUG := -std=c++17 \
                    -arch=$(CUDA_ARCH) \
                    --expt-relaxed-constexpr \
                    --extended-lambda \
                    -G -g -O0

INCLUDES := -I$(INC_DIR) -I$(EXT_DIR)

# Libraries
LIBS := -lcudart -lcurand

# Source files
SOURCES := $(SRC_DIR)/main.cu
OBJECTS := $(OBJ_DIR)/main.o

# Main targets
.PHONY: all clean debug release dirs help

all: release

release: NVCC_FLAGS := $(NVCC_FLAGS) -DNDEBUG
release: dirs $(BIN_DIR)/$(PROJECT)
	@echo "Build complete: $(BIN_DIR)/$(PROJECT)"

debug: NVCC_FLAGS := $(NVCC_FLAGS_DEBUG)
debug: dirs $(BIN_DIR)/$(PROJECT)_debug
	@echo "Debug build complete: $(BIN_DIR)/$(PROJECT)_debug"

# Create directories
dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Compile main
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cu
	@echo "Compiling $<..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(BIN_DIR)/$(PROJECT): $(OBJECTS)
	@echo "Linking $@..."
	@$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(LIBS) -o $@

$(BIN_DIR)/$(PROJECT)_debug: $(OBJECTS)
	@echo "Linking $@..."
	@$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(LIBS) -o $@

# Clean
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Help
help:
	@echo "CUDA Ray Tracer Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [CUDA_ARCH=sm_XX]"
	@echo ""
	@echo "Targets:"
	@echo "  all/release  - Build optimized release version"
	@echo "  debug        - Build debug version with symbols"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Architecture Options:"
	@echo "  CUDA_ARCH=sm_70  - NVIDIA V100"
	@echo "  CUDA_ARCH=sm_80  - NVIDIA A100"
	@echo "  CUDA_ARCH=sm_90  - NVIDIA H100/GH200 (default)"
	@echo ""
	@echo "Examples:"
	@echo "  make                        # Build for H100 (default)"
	@echo "  make CUDA_ARCH=sm_80        # Build for A100"
	@echo "  make debug CUDA_ARCH=sm_90  # Debug build for H100"
