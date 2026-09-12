# Romeo cluster notes

These are adaptation notes for the historical Slurm scripts, not a validated cluster deployment guide. No Romeo job was submitted for the current report. Cluster performance is **non mesuré**, and local WSL2 results must not be extrapolated to GH200 or H100 hardware.

Confirm the site's current partition, account, GPU request syntax and available compiler/CUDA modules with the cluster documentation. Values in `scripts/slurm/` are examples tied to the original coursework environment.

## Build and verify on an allocated node

After loading a compatible toolchain, use CMake with the architecture of the allocated device:

```bash
nvidia-smi
cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DBUILD_TESTS=ON
cmake --build build_gpu --parallel 4
./build_gpu/raytracer --info
ctest --test-dir build_gpu --output-on-failure
```

The architecture value above is the original Hopper target; verify it for the actual allocation. It is not the value used for the local RTX 4070 measurements.

## Scripts needing review

`run_benchmark.slurm` passes `--scene scenes/benchmark.json`, but there is no scene loader and unknown options are ignored. It times complete process execution while older tables mix other timing scopes. `run_raytracer.slurm` and related examples also require site-specific review. Do not use their presence as proof that the current branch works on that cluster.

For a fresh experiment, adapt [bench/run.sh](../bench/run.sh) to the job environment and give it a new results directory. Keep the raw outputs and environment metadata. Its ordinary-render invocations avoid the camera mismatch in the built-in multi-resolution benchmark.
