#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${RT_MODE:-both}"
build="${RT_BUILD_DIR:-build_bench}"
if [[ "$mode" == cpu ]]; then
    cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU_ONLY=ON -DENABLE_FAST_MATH=ON
else
    arch="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')}"
    cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" -DENABLE_FAST_MATH=ON
fi
cmake --build "$build" --target raytracer --parallel "${BUILD_JOBS:-4}"
python3 bench/measure.py --binary "$build/raytracer" --mode "$mode" --output "${RT_RESULTS_DIR:-bench/results/wsl-2026-09-12}" --threads "${OMP_NUM_THREADS:-4}"
python3 bench/plot.py "${RT_RESULTS_DIR:-bench/results/wsl-2026-09-12}"
