#!/bin/bash
#SBATCH --job-name=rt_bench
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=instant
#SBATCH --account=r250127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=armgpu
#SBATCH --time=00:30:00

echo "=== BENCHMARK RAYTRACER CUDA - ROMEO GH200 ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

module load cuda/12.6
export PATH=$HOME/.local_arm/bin:$PATH

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CMake: $(cmake --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"
echo ""

cd $HOME/RayTracing

# Compile on the GPU node (aarch64)
echo "=== Compilation ==="
rm -rf build_romeo
mkdir -p build_romeo && cd build_romeo
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DENABLE_INTERACTIVE=OFF 2>&1
make -j${SLURM_CPUS_PER_TASK} 2>&1
cd ..

if [ ! -f build_romeo/raytracer ]; then
    echo "ERREUR: compilation echouee"
    exit 1
fi

echo ""
echo "=== Benchmarks GPU (NVIDIA GH200 H100) ==="

for config in "640 480 100" "1280 720 100" "1920 1080 100" "1920 1080 500" "3840 2160 100" "3840 2160 500"; do
    read -r w h spp <<< "$config"
    echo ""
    echo "--- GPU ${w}x${h} @ ${spp} spp ---"
    ./build_romeo/raytracer -w $w -h $h -s $spp -d 50 -o output/bench_${w}x${h}_${spp}spp.png --profile 2>&1
done

echo ""
echo "=== Benchmarks CPU (GH200 ARM Neoverse V2, ${SLURM_CPUS_PER_TASK} threads) ==="

for config in "640 480 100" "1280 720 100" "1920 1080 100"; do
    read -r w h spp <<< "$config"
    echo ""
    echo "--- CPU ${w}x${h} @ ${spp} spp ---"
    ./build_romeo/raytracer -w $w -h $h -s $spp -d 50 -o output/bench_cpu_${w}x${h}_${spp}spp.png --cpu --profile 2>&1
done

echo ""
echo "=== BENCHMARK TERMINE ==="
