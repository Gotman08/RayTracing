#ifndef RAYTRACER_CORE_CUDA_COMPAT_CUH
#define RAYTRACER_CORE_CUDA_COMPAT_CUH

/** @file cuda_compat.cuh
 *  @brief Compat CUDA/CPU - macros vides si pas de nvcc */

#ifdef __CUDACC__
    #include <cuda_runtime.h>
#else
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

    #ifndef __align__
        #define __align__(n) alignas(n)
    #endif

    struct curandState; ///< Forward decl pour compil sans CUDA

#endif

#endif
