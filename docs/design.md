# Design of the current renderer

The current program renders a procedural scene using stochastic ray bounces. The CPU and GPU share the scene definition and most geometry/material representations. This document describes the code at the measured source fingerprint in the [environment manifest](../bench/results/wsl-2026-09-12-isolated/environment.json).

## Representation and traversal

`HittableObject` selects sphere or plane data through an enum and union. A hit record carries the intersection and a material pointer. The GPU path copies the material array, changes object material pointers to device addresses, then uploads the BVH. The CPU path retains host pointers.

This representation avoids device-side virtual dispatch and keeps the supported types explicit. A class hierarchy would make adding types more extensible, but it would require a different ownership and dispatch design across host/device memory. No measured comparison to virtual dispatch is available.

`BVHBuilder` sorts object bounds along the longest axis and splits at the median. It builds on the CPU and stores nodes in flat arrays for iterative traversal. A surface-area heuristic could reduce traversal work for some scenes at greater build complexity; it has not been implemented or benchmarked. A brute-force reference is available in the geometry utilities, but the committed performance experiment compares the existing OpenMP and CUDA renderer paths, not BVH heuristics.

`main_cpu.cpp` retains its own `CPUBVHBuilder`. Removing that duplication would be a renderer refactor and was not part of the documentation and measurement work.

## Rendering stages

The default scene is created in `src/scene/default_scene.cuh`: a ground sphere, central objects, planets, a ring and stars. Its pseudo-random placement is seeded with `srand(77)`. It contains 80 spheres. Planes have an implementation and tests but are not used in this scene. The camera is initialized from the requested image dimensions before ordinary rendering.

The CPU renderer parallelizes image rows with OpenMP and seeds a `CPURandom` instance per row. It samples rays, traverses the BVH, scatters through Lambertian, metal or dielectric materials, accumulates sky/background contribution, and applies ordinary Reinhard tone mapping plus gamma correction.

The GPU assigns one thread to each pixel in a 16 × 16 block. It initializes cuRAND states, stores camera and render configuration in constant memory, and renders HDR pixel colors. A reduction computes average luminance using shared memory, warp operations and an atomic sum. The program also runs a Thrust reduction for comparison before applying adaptive Reinhard tone mapping. Pinned host memory receives the framebuffer before export.

A megakernel keeps this small implementation straightforward. A wavefront renderer could organize rays by stage and material, but would need queues, compaction and more kernels. That alternative has not been evaluated here.

Two CUDA streams are created, but the present synchronization and synchronous scene uploads do not demonstrate useful overlap between transfers and rendering. Stream overlap and memory-bandwidth estimates printed by the profiler must not be read as measured hardware utilization.

## Supported scope

The active material dispatch supports Lambertian, metal and dielectric scattering. Texture lookup is constant color. There is no JSON loader, interactive window, triangle/box geometry, emissive-light sampling, volumetric material or Russian Roulette. Ray timestamps exist in the camera/ray representation, but the default geometry is static; this is not evidence of a moving-object renderer.

CPU and GPU tone mapping and random generators differ. Their output images are not used as bitwise references for each other. The legacy multi-resolution `--benchmark` mutates render dimensions without rebuilding the camera, so the new harness invokes ordinary rendering for each size.

## Build repair scope

Commit `4bcaf47` makes the existing CPU-only path compile: device-only camera and material functions are guarded by `__CUDACC__`, and `main_cpu.cpp` includes `<algorithm>` for `std::sort`. The random formulas, scene, traversal, material computations and CUDA kernel bodies were preserved. Existing tests supplied CUDA stubs in host builds, which explains why unit tests passed before the real CPU executable compiled.

The [validation logs](../bench/results/validation/) retain both the original compiler failure and the successful build/tests after repair. Packaging and CLI behavior remain separate known issues.

## Documentation sources

`doxygen Doxyfile` regenerates API documentation under `docs/html/`; generated files are intentionally ignored. The configuration reads `include/`, `src/` and `tests/`. Historical reports remain in [rapport/](../rapport/README.md), with their provenance and unverified claims explicitly identified.
