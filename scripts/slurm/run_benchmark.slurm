#!/bin/bash
#===============================================================================
# SLURM Benchmark Script - Multi-resolution performance testing
# Target: Romeo2025 GH200/H100
#===============================================================================

#SBATCH --job-name=rt_bench
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=gpu
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
module load cuda/12.6 cmake/3.28 gcc/13.2

cd ${SLURM_SUBMIT_DIR}

# Build if needed
if [ ! -f "build/raytracer" ]; then
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
    make -j${SLURM_CPUS_PER_TASK}
    cd ..
fi

# Benchmark configurations
RESOLUTIONS=("640 480" "1280 720" "1920 1080" "3840 2160")
SAMPLES=(1 10 50 100 500)
OUTPUT_DIR="benchmarks/${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

echo "Resolution,Samples,Time(s),MRays/s" > ${OUTPUT_DIR}/results.csv

echo "=== CUDA Ray Tracer Benchmark ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

for res in "${RESOLUTIONS[@]}"; do
    read -r width height <<< "$res"
    for spp in "${SAMPLES[@]}"; do
        echo "Running: ${width}x${height} @ ${spp} spp"

        # Time the execution
        start=$(date +%s.%N)
        ./build/raytracer \
            --width ${width} \
            --height ${height} \
            --samples ${spp} \
            --depth 50 \
            --scene scenes/benchmark.json \
            --output /dev/null \
            --quiet
        end=$(date +%s.%N)

        elapsed=$(echo "$end - $start" | bc)
        total_rays=$(echo "$width * $height * $spp" | bc)
        mrays=$(echo "scale=2; $total_rays / $elapsed / 1000000" | bc)

        echo "${width}x${height},${spp},${elapsed},${mrays}" >> ${OUTPUT_DIR}/results.csv
        echo "  -> ${elapsed}s (${mrays} MRays/s)"
    done
done

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: ${OUTPUT_DIR}/results.csv"
echo ""
cat ${OUTPUT_DIR}/results.csv
