#ifndef RAYTRACER_RENDERING_RENDERER_CUH
#define RAYTRACER_RENDERING_RENDERER_CUH

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/rendering/render_config.cuh"
#include "raytracer/rendering/material_dispatch.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

namespace rt {

__device__ inline Color ray_color(
    const Ray& initial_ray,
    const BVH& bvh,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    Color accumulated(0, 0, 0);
    Color attenuation(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (bvh.hit(current_ray, Interval(0.001f, INFINITY_F), rec)) {
            Ray scattered;
            Color scatter_attenuation;

            if (scatter(*rec.mat, current_ray, rec, scatter_attenuation, scattered, rand_state)) {
                attenuation = attenuation * scatter_attenuation;
                current_ray = scattered;
            } else {
                break;
            }
        } else {
            if (config.use_sky) {
                accumulated += attenuation * config.sky.get_color(current_ray.direction());
            } else {
                accumulated += attenuation * config.background;
            }
            break;
        }
    }

    return accumulated;
}

__global__ void render_kernel(
    Color* frame_buffer,
    Camera camera,
    BVH bvh,
    RenderConfig config,
    curandState* rand_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= config.width || j >= config.height) return;

    int pixel_index = j * config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < config.samples_per_pixel; s++) {
        Ray r = camera.get_ray(i, j, local_rand_state);
        pixel_color += ray_color(r, bvh, config.max_depth, config, local_rand_state);
    }

    pixel_color = pixel_color / static_cast<float>(config.samples_per_pixel);

    pixel_color = apply_tone_mapping(pixel_color);
    pixel_color = gamma_correct(pixel_color);

    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

// Fast hash function for quick RNG seeding (Wang hash)
__device__ __forceinline__ unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// OPTIMIZED: Fast random state initialization using hash instead of curand_init
__global__ void init_rand_states_fast(
    curandState* rand_states,
    int width, int height,
    unsigned int seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    // Use fast hash-based initialization
    unsigned int hash = wang_hash(seed + pixel_index);
    curand_init(hash, 0, 0, &rand_states[pixel_index]);
}

// Original slow version (kept for compatibility)
__global__ void init_rand_states(
    curandState* rand_states,
    int width, int height,
    unsigned long long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &rand_states[pixel_index]);
}

#ifdef ENABLE_INTERACTIVE
// Render kernel for progressive accumulation (no tone mapping)
__global__ void render_kernel_accumulate(
    Color* accumulation_buffer,
    Camera camera,
    BVH bvh,
    RenderConfig config,
    curandState* rand_states,
    int samples_this_frame
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= config.width || j >= config.height) return;

    int pixel_index = j * config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < samples_this_frame; s++) {
        Ray r = camera.get_ray(i, j, local_rand_state);
        pixel_color += ray_color(r, bvh, config.max_depth, config, local_rand_state);
    }

    // Add to accumulation buffer (raw HDR, no tone mapping yet)
    accumulation_buffer[pixel_index] = accumulation_buffer[pixel_index] + pixel_color;
}
#endif

}

#endif
