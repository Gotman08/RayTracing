# Local validation evidence

Validation ran under Ubuntu 24.04.4 on WSL2. These are local command outputs; hosted GitHub Actions execution is pending publication. Checkout prefixes are normalized to `<repo>` and the final LF is preserved. No assertion or timing output was edited.

| Log | Command and outcome |
|---|---|
| [cpu-build-before-fix.log](cpu-build-before-fix.log) | Original `cmake --build build_cpu_audit --parallel 4`: failed on device functions and missing `<algorithm>` |
| [cpu-build.log](cpu-build.log) | Same build after `4bcaf47`: CPU renderer and test targets built |
| [cpu-tests.log](cpu-tests.log) | `ctest --test-dir build_cpu_audit --output-on-failure --parallel 4`: 210 passed |
| [cpu-smoke.log](cpu-smoke.log) | CPU-only executable rendered a 64 × 36 PNG with 2 samples/pixel and depth 5 |
| [gpu-build.log](gpu-build.log) | `cmake --build build_gpu_audit --target raytracer raytracer_gpu_tests --parallel 4`: passed after repair |
| [gpu-tests.log](gpu-tests.log) | `./build_gpu_audit/raytracer_gpu_tests`: 8 passed after repair |
| [gpu-info.log](gpu-info.log) | Project executable queried the actual CUDA device successfully |

The CPU executable compile check matters because some unit-test translation units provide host substitutes for CUDA functions. Test success alone did not establish that the CPU application built. The smoke output is evidence of execution, not a performance benchmark.
