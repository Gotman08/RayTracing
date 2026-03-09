#ifndef RAYTRACER_INTERACTIVE_ACCUMULATION_BUFFER_CUH
#define RAYTRACER_INTERACTIVE_ACCUMULATION_BUFFER_CUH

#ifdef ENABLE_INTERACTIVE

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"

namespace rt {

// Kernel to clear accumulation buffer
__global__ void clear_accumulation_kernel(Color* buffer, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;
    buffer[idx] = Color(0, 0, 0);
}

// Kernel to add new samples to accumulation buffer
__global__ void accumulate_samples_kernel(
    Color* accumulation_buffer,
    const Color* new_samples,
    int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;
    accumulation_buffer[idx] = accumulation_buffer[idx] + new_samples[idx];
}

class AccumulationBuffer {
public:
    Color* d_buffer;              // Device HDR accumulation buffer
    int width, height;
    int accumulated_samples;       // Total samples accumulated

    AccumulationBuffer()
        : d_buffer(nullptr), width(0), height(0), accumulated_samples(0) {}

    bool initialize(int w, int h) {
        width = w;
        height = h;
        accumulated_samples = 0;

        cudaError_t err = cudaMalloc(&d_buffer, width * height * sizeof(Color));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate accumulation buffer: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        reset();
        return true;
    }

    void reset() {
        accumulated_samples = 0;

        constexpr int BLOCK_SIZE = 16;
        dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        clear_accumulation_kernel<<<blocks, threads>>>(d_buffer, width, height);
        cudaDeviceSynchronize();
    }

    void accumulate(const Color* d_new_samples, int new_sample_count) {
        constexpr int BLOCK_SIZE = 16;
        dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        accumulate_samples_kernel<<<blocks, threads>>>(
            d_buffer, d_new_samples, width, height);

        accumulated_samples += new_sample_count;
    }

    int get_accumulated_samples() const {
        return accumulated_samples;
    }

    void cleanup() {
        if (d_buffer) {
            cudaFree(d_buffer);
            d_buffer = nullptr;
        }
    }

    ~AccumulationBuffer() {
        // Note: cleanup() should be called explicitly before CUDA context destruction
    }
};

} // namespace rt

#endif // ENABLE_INTERACTIVE
#endif // RAYTRACER_INTERACTIVE_ACCUMULATION_BUFFER_CUH
