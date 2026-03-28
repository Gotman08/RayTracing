/**
 * @file test_cuda_kernels.cu
 * @brief Tests GPU : curand init, reduction luminance, tonemapping, memoire constante
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/rendering/tone_mapping.cuh"
#include "raytracer/rendering/render_config.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/rendering/renderer.cuh"

using namespace rt;

__global__ void read_const_camera_kernel(Camera* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = d_const_camera;
    }
}

__global__ void read_const_config_kernel(RenderConfig* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = d_const_config;
    }
}

/** @brief Init curand states 4x4 sans crash GPU */
TEST(CudaKernelsTest, InitRandStates) {
    const int W = 4, H = 4;
    curandState* d_states;
    cudaMalloc(&d_states, W * H * sizeof(curandState));

    dim3 blocks(1, 1);
    dim3 threads(W, H);
    init_rand_states_fast<<<blocks, threads>>>(d_states, W, H, 12345);
    cudaDeviceSynchronize();

    std::vector<curandState> h_states(W * H);
    cudaMemcpy(h_states.data(), d_states, W * H * sizeof(curandState), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    ASSERT_TRUE(err == cudaSuccess);

    cudaFree(d_states);
}

/** @brief Reduction rouge pur -> luminance ~ 0.2126 */
TEST(CudaKernelsTest, LuminanceReductionRed) {
    const int N = 1024;
    std::vector<Color> h_pixels(N, Color(1.0f, 0.0f, 0.0f));

    Color* d_pixels;
    float* d_total;
    cudaMalloc(&d_pixels, N * sizeof(Color));
    cudaMalloc(&d_total, sizeof(float));

    cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Color), cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(float));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    compute_avg_luminance<<<grid_size, block_size>>>(d_pixels, d_total, N);
    cudaDeviceSynchronize();

    float h_total = 0.0f;
    cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
    float avg = h_total / (float)N;

    EXPECT_NEAR(avg, 0.2126f, 1e-3f);

    cudaFree(d_pixels);
    cudaFree(d_total);
}

/** @brief Reduction blanc (1,1,1) -> luminance = 1.0 */
TEST(CudaKernelsTest, LuminanceReductionWhite) {
    const int N = 512;
    std::vector<Color> h_pixels(N, Color(1.0f, 1.0f, 1.0f));

    Color* d_pixels;
    float* d_total;
    cudaMalloc(&d_pixels, N * sizeof(Color));
    cudaMalloc(&d_total, sizeof(float));

    cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Color), cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(float));

    compute_avg_luminance<<<(N + 255) / 256, 256>>>(d_pixels, d_total, N);
    cudaDeviceSynchronize();

    float h_total = 0.0f;
    cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
    float avg = h_total / (float)N;

    EXPECT_NEAR(avg, 1.0f, 1e-3f);

    cudaFree(d_pixels);
    cudaFree(d_total);
}

/** @brief Tonemapping adaptatif -> tous pixels dans [0,1] */
TEST(CudaKernelsTest, AdaptiveTonemappingRange) {
    const int N = 256;
    std::vector<Color> h_pixels(N);
    for (int i = 0; i < N; i++) {
        float v = (float)(i + 1) * 0.5f;
        h_pixels[i] = Color(v, v * 0.5f, v * 0.1f);
    }

    Color* d_pixels;
    cudaMalloc(&d_pixels, N * sizeof(Color));
    cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Color), cudaMemcpyHostToDevice);

    float avg_lum = 5.0f;
    apply_tonemapping_kernel<<<(N + 255) / 256, 256>>>(d_pixels, avg_lum, N);
    cudaDeviceSynchronize();

    std::vector<Color> h_result(N);
    cudaMemcpy(h_result.data(), d_pixels, N * sizeof(Color), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        ASSERT_TRUE(h_result[i].x >= 0.0f && h_result[i].x <= 1.0f);
        ASSERT_TRUE(h_result[i].y >= 0.0f && h_result[i].y <= 1.0f);
        ASSERT_TRUE(h_result[i].z >= 0.0f && h_result[i].z <= 1.0f);
    }

    cudaFree(d_pixels);
}

/** @brief Const memory : Camera host -> device -> relue OK */
TEST(CudaKernelsTest, ConstantMemoryCamera) {
    Camera h_camera;
    h_camera.center = Point3(1.0f, 2.0f, 3.0f);
    h_camera.image_width = 800;
    h_camera.image_height = 600;
    h_camera.defocus_angle = 0.5f;

    cudaMemcpyToSymbol(d_const_camera, &h_camera, sizeof(Camera));

    Camera* d_out;
    cudaMalloc(&d_out, sizeof(Camera));
    read_const_camera_kernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    Camera h_result;
    cudaMemcpy(&h_result, d_out, sizeof(Camera), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_result.center.x, 1.0f, 1e-5f);
    EXPECT_NEAR(h_result.center.y, 2.0f, 1e-5f);
    EXPECT_NEAR(h_result.center.z, 3.0f, 1e-5f);
    ASSERT_TRUE(h_result.image_width == 800);
    ASSERT_TRUE(h_result.image_height == 600);
    EXPECT_NEAR(h_result.defocus_angle, 0.5f, 1e-5f);

    cudaFree(d_out);
}

/** @brief Const memory : RenderConfig roundtrip OK */
TEST(CudaKernelsTest, ConstantMemoryConfig) {
    RenderConfig h_config;
    h_config.width = 1920;
    h_config.height = 1080;
    h_config.samples_per_pixel = 64;
    h_config.max_depth = 10;

    cudaMemcpyToSymbol(d_const_config, &h_config, sizeof(RenderConfig));

    RenderConfig* d_out;
    cudaMalloc(&d_out, sizeof(RenderConfig));
    read_const_config_kernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    RenderConfig h_result;
    cudaMemcpy(&h_result, d_out, sizeof(RenderConfig), cudaMemcpyDeviceToHost);

    ASSERT_TRUE(h_result.width == 1920);
    ASSERT_TRUE(h_result.height == 1080);
    ASSERT_TRUE(h_result.samples_per_pixel == 64);
    ASSERT_TRUE(h_result.max_depth == 10);

    cudaFree(d_out);
}

/** @brief Reduction N=300 (non aligne) -> pas de bruit des threads hors bornes */
TEST(CudaKernelsTest, LuminanceReductionNonAligned) {
    const int N = 300;
    std::vector<Color> h_pixels(N, Color(0.0f, 1.0f, 0.0f));

    Color* d_pixels;
    float* d_total;
    cudaMalloc(&d_pixels, N * sizeof(Color));
    cudaMalloc(&d_total, sizeof(float));

    cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Color), cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(float));

    compute_avg_luminance<<<(N + 255) / 256, 256>>>(d_pixels, d_total, N);
    cudaDeviceSynchronize();

    float h_total = 0.0f;
    cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
    float avg = h_total / (float)N;

    EXPECT_NEAR(avg, 0.7152f, 1e-3f);

    cudaFree(d_pixels);
    cudaFree(d_total);
}

/** @brief Tonemapping scene noire -> reste noir */
TEST(CudaKernelsTest, TonemappingBlackScene) {
    const int N = 64;
    std::vector<Color> h_pixels(N, Color(0.0f, 0.0f, 0.0f));

    Color* d_pixels;
    cudaMalloc(&d_pixels, N * sizeof(Color));
    cudaMemcpy(d_pixels, h_pixels.data(), N * sizeof(Color), cudaMemcpyHostToDevice);

    apply_tonemapping_kernel<<<1, N>>>(d_pixels, 0.5f, N);
    cudaDeviceSynchronize();

    std::vector<Color> h_result(N);
    cudaMemcpy(h_result.data(), d_pixels, N * sizeof(Color), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_result[i].x, 0.0f, 1e-5f);
        EXPECT_NEAR(h_result[i].y, 0.0f, 1e-5f);
        EXPECT_NEAR(h_result[i].z, 0.0f, 1e-5f);
    }

    cudaFree(d_pixels);
}
