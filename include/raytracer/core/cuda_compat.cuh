/**
 * CUDA Compatibility Header
 * Provides stub definitions when CUDA is not available
 * This allows host-only code to compile without CUDA toolkit
 */

#ifndef RAYTRACER_CORE_CUDA_COMPAT_CUH
#define RAYTRACER_CORE_CUDA_COMPAT_CUH

// Check if we're compiling with CUDA
#ifdef __CUDACC__
    // CUDA is available, include real headers
    #include <cuda_runtime.h>
#else
    // CUDA not available, define stubs

    // Define __host__ and __device__ as empty macros
    #ifndef __host__
        #define __host__
    #endif

    #ifndef __device__
        #define __device__
    #endif

    #ifndef __global__
        #define __global__
    #endif

    #ifndef __constant__
        #define __constant__
    #endif

    // Stub for curandState (not used in host tests)
    struct curandState;

#endif // __CUDACC__

#endif // RAYTRACER_CORE_CUDA_COMPAT_CUH
