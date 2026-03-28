#ifndef RAYTRACER_CORE_CUDA_UTILS_CUH
#define RAYTRACER_CORE_CUDA_UTILS_CUH

/** @file cuda_utils.cuh
 *  @brief Utilitaires CUDA - alloc/free GPU, check erreurs, init curand */

#include "raytracer/core/cuda_compat.cuh"
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace rt {

constexpr float PI = 3.14159265358979323846f;
constexpr float INFINITY_F = 1e30f;
constexpr float EPSILON = 1e-6f;

/** @brief Degres -> radians */
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

#ifdef __CUDACC__

/** @brief Macro check erreur CUDA (abort si echec) */
#define CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)

/** @brief Fn interne pour CUDA_CHECK - affiche erreur + exit */
inline void check_cuda(cudaError_t result, const char* func,
                       const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d - %s\n    %s: %s\n",
                file, line, func,
                cudaGetErrorName(result),
                cudaGetErrorString(result));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
#define CUDA_SYNC_CHECK() CUDA_CHECK(cudaDeviceSynchronize())

/** @brief Alloc mem device (cudaMalloc wrapper) */
template<typename T>
T* cuda_alloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

/** @brief Alloc mem managed (CPU+GPU) */
template<typename T>
T* cuda_alloc_managed(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

/** @brief Free mem GPU (safe si nullptr) */
template<typename T>
void cuda_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/** @brief Copie host -> device */
template<typename T>
void cuda_copy_to_device(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

/** @brief Copie device -> host */
template<typename T>
void cuda_copy_to_host(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

/** @brief Affiche les infos GPU (nom, mem, SM, etc.) */
inline void print_device_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Shared memory per block: %.2f KB\n",
               prop.sharedMemPerBlock / 1024.0);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("\n");
    }
}

/** @brief Kernel init curand (1 state par pixel) */
__global__ inline void init_curand_states(curandState* states, int width, int height, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &states[pixel_index]);
}

#endif

}

#endif
