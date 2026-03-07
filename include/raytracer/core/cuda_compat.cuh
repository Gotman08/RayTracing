#ifndef RAYTRACER_CORE_CUDA_COMPAT_CUH
#define RAYTRACER_CORE_CUDA_COMPAT_CUH

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

    struct curandState;

#endif

#endif
