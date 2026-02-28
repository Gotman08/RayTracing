#!/bin/bash
#===============================================================================
# SLURM Job Script for CUDA Ray Tracer on Romeo2025 HPC
# Target: GH200 Superchips (H100 GPU)
#===============================================================================

#SBATCH --job-name=raytracer
#SBATCH --output=raytracer_%j.out
#SBATCH --error=raytracer_%j.err

# Partition and account (MODIFY THIS)
#SBATCH --partition=gpu
#SBATCH --account=your_account

# Resources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# GPU allocation - H100 on Romeo2025
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

# Time limit
#SBATCH --time=02:00:00

#===============================================================================
# Environment Setup
#===============================================================================

echo "=================================================="
echo "CUDA Ray Tracer - Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Load required modules (adjust for Romeo's module system)
module purge
module load cuda/12.6
module load cmake/3.28
module load gcc/13.2

# Print environment info
echo ""
echo "=== Environment ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo ""

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR}

#===============================================================================
# Build (if needed)
#===============================================================================

BUILD_DIR="build"

if [ ! -f "${BUILD_DIR}/raytracer" ]; then
    echo "=== Building project ==="
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=90

    make -j${SLURM_CPUS_PER_TASK}

    cd ..
    echo "Build complete."
else
    echo "Using existing build."
fi

#===============================================================================
# Run Ray Tracer
#===============================================================================

echo ""
echo "=== Running Ray Tracer ==="

OUTPUT_DIR="output/${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

# Default render
./build/raytracer \
    --width 1920 \
    --height 1080 \
    --samples 100 \
    --depth 50 \
    --scene scenes/spheres.json \
    --output ${OUTPUT_DIR}/render.png

#===============================================================================
# Summary
#===============================================================================

echo ""
echo "=== Job Summary ==="
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}/render.png"
ls -la ${OUTPUT_DIR}
