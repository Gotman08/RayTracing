#!/bin/bash
#===============================================================================
# Interactive GPU session for development/debugging on Romeo2025
#===============================================================================

#SBATCH --job-name=rt_interactive
#SBATCH --partition=gpu
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Usage:
#   sbatch scripts/slurm/interactive.slurm
# Or for direct interactive:
#   salloc -p gpu --account=your_account --gres=gpu:1 --time=01:00:00

module purge
module load cuda/12.6 cmake/3.28 gcc/13.2

echo "=== Interactive GPU Session Ready ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo ""
echo "Quick commands:"
echo "  Build:  mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 && make -j4"
echo "  Run:    ./build/raytracer --info"
echo "  Test:   ./build/raytracer -s 10 -o test.png"
echo ""

# Keep session alive
exec bash
