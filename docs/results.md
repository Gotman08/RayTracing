# Results and measurement scope

The published comparison uses the [isolated WSL2 collection](../bench/results/wsl-2026-09-12-isolated/). Each result below is derived from its CSV by `bench/plot.py`. Medians use five observations after one declared warmup. Q1 and Q3 use Python `statistics.quantiles(..., method="inclusive")`; IQR is Q3 minus Q1. These intervals describe the observed spread and are not confidence intervals.

## Complete process latency

A fresh process constructs the scene, renders, post-processes, exports PNG and exits. The timer is Python `perf_counter_ns`, outside the executable. It includes CUDA initialization and filesystem work.

| Image | Mode | Median (ms) | Q1 (ms) | Q3 (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|
| 320 × 180 | CPU | 469.3 | 466.8 | 471.3 | 4.5 |
| 320 × 180 | GPU | 817.8 | 808.2 | 822.4 | 14.2 |
| 800 × 450 | CPU | 2,649.0 | 2,647.5 | 2,649.2 | 1.7 |
| 800 × 450 | GPU | 939.9 | 929.9 | 965.6 | 35.6 |

At 800 × 450, the ratio of CPU to GPU medians is 2.82×. The CPU uses four OpenMP workers on an i9-13900H; the GPU is a RTX 4070 Laptop. Both modes use 50 samples/pixel and depth 20. This is a ratio of medians for the current programs, not a claim of equal image quality or a paired speedup distribution.

## Reported render stages

CPU timing includes its sampling and ordinary tone mapping. GPU timing is a CUDA-event measurement of the HDR sampling kernel and excludes adaptive tone mapping, reduction, transfers and process startup. Comparing these timers as if they were the same pipeline would overstate the practical benefit.

| Image | Mode | Median (ms) | Q1 (ms) | Q3 (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|
| 320 × 180 | CPU | 421.9 | 420.0 | 423.0 | 3.0 |
| 320 × 180 | GPU | 23.1 | 23.0 | 24.1 | 1.2 |
| 800 × 450 | CPU | 2,572.0 | 2,571.6 | 2,573.9 | 2.2 |
| 800 × 450 | GPU | 78.2 | 78.0 | 78.3 | 0.3 |

## Experimental limits

The scene is fixed, seeded by `srand(77)`. CPU random streams are seeded by row with `j * 1337 + 42`, whereas GPU initialization uses `time(NULL)`. No common seed argument exists. The two renderers also use different tone mapping. This report does not measure convergence, noise, perceptual quality or error against a reference.

Warmup means an excluded complete invocation for each mode and size. Every later trial still launches a new process, so it is not a benchmark of a persistent warmed CUDA context. The PNG is removed before each run and its signature/dimensions are checked afterwards; the digest is recorded. The image is not kept as a build artifact.

The measured executable was built by nvcc with architecture 89 and fast math enabled. CPU-via-CUDA and CPU-only builds use different host build routes, so the CPU-only executable is validated separately but not substituted into this comparison. Detailed hardware, WSL topology, memory, compiler commands and fingerprints are in the [manifest](../bench/results/wsl-2026-09-12-isolated/environment.json).

The first complete collection overlapped a repository cleanup. It is retained as [preliminary evidence](../bench/results/wsl-2026-09-12/README.md). The second collection used an exclusive agent benchmark interval with no concurrent operation by the measuring agent. Neither collection controls unrelated host activity, power limits or thermal state.

Only these two image sizes were measured. Native Linux, native Windows/MSVC, GH200/H100 cluster performance, energy use and image quality are **non mesuré**, because no corresponding experiment was run. No published comparison to another renderer is available.

## Local validation and CI

The CPU-only build initially failed on unguarded CUDA functions and a missing standard include. The isolated build repair is documented in [design.md](design.md). After repair, the CPU renderer builds, 210 CPU tests pass, a CPU-only image is produced, and 8 GPU tests pass on the local GPU. [Raw validation logs](../bench/results/validation/)

The Ubuntu GitHub Actions workflow builds the CPU executable, runs tests, and validates the PNG dimensions from a small render. Those command categories were exercised locally under Ubuntu WSL2. The hosted workflow has not run because the branch is prepared locally and has not been published.

## Figure reproduction

```bash
python3 -m pip install -r bench/requirements.txt
python3 bench/plot.py bench/results/wsl-2026-09-12-isolated
```

The generated light/dark SVGs use the same data and colors with adjusted contrast. Both PNG previews were visually inspected. The summary JSON and figures are derived artifacts; raw CSV, environment metadata and individual logs remain the primary evidence.
