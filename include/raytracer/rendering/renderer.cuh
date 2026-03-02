/**
 * GPU Renderer - Path Tracing Kernels
 */

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

// =============================================================================
// Path Tracing
// =============================================================================

/**
 * Template-based path tracing - works with any acceleration structure that has hit()
 */
template<typename AccelerationStructure>
__device__ inline Color ray_color_impl(
    const Ray& initial_ray,
    const AccelerationStructure& accel,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    Color accumulated(0, 0, 0);
    Color attenuation(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (accel.hit(current_ray, Interval(0.001f, INFINITY_F), rec)) {
            // Get emission from material
            Color emitted = rec.mat->emitted(rec.u, rec.v, rec.p);
            accumulated += attenuation * emitted;

            // Try to scatter
            Ray scattered;
            Color scatter_attenuation;

            if (scatter(*rec.mat, current_ray, rec, scatter_attenuation, scattered, rand_state)) {
                attenuation = attenuation * scatter_attenuation;
                current_ray = scattered;

                // Russian Roulette after min bounces
                if (depth > 3) {
                    float p = fmaxf(attenuation.x, fmaxf(attenuation.y, attenuation.z));
                    if (curand_uniform(rand_state) > p) {
                        break;
                    }
                    attenuation = attenuation / p;
                }
            } else {
                break;  // Ray absorbed
            }
        } else {
            // Ray escaped - add background/sky
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

// Convenience wrappers for backward compatibility
__device__ inline Color ray_color(
    const Ray& initial_ray,
    const HittableList& world,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    return ray_color_impl(initial_ray, world, max_depth, config, rand_state);
}

__device__ inline Color ray_color_bvh(
    const Ray& initial_ray,
    const BVH& bvh,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    return ray_color_impl(initial_ray, bvh, max_depth, config, rand_state);
}

// =============================================================================
// Render Kernels
// =============================================================================

/**
 * Template-based render kernel - works with any acceleration structure
 */
template<typename AccelerationStructure>
__global__ void render_kernel_impl(
    Color* frame_buffer,
    Camera camera,
    AccelerationStructure accel,
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
        pixel_color += ray_color_impl(r, accel, config.max_depth, config, local_rand_state);
    }

    // Average samples
    pixel_color = pixel_color / static_cast<float>(config.samples_per_pixel);

    // Post-processing pipeline
    pixel_color = apply_exposure(pixel_color, config.exposure);
    pixel_color = apply_tone_mapping(pixel_color, config.tone_map);
    pixel_color = gamma_correct(pixel_color);

    // Clamp to [0, 1]
    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

// Convenience kernel wrappers
__global__ void render_kernel(
    Color* frame_buffer,
    Camera camera,
    HittableList world,
    RenderConfig config,
    curandState* rand_states
) {
    render_kernel_impl(frame_buffer, camera, world, config, rand_states);
}

__global__ void render_kernel_bvh(
    Color* frame_buffer,
    Camera camera,
    BVH bvh,
    RenderConfig config,
    curandState* rand_states
) {
    render_kernel_impl(frame_buffer, camera, bvh, config, rand_states);
}

// =============================================================================
// Random State Initialization
// =============================================================================

__global__ void init_render_rand_states(
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

} // namespace rt

#endif // RAYTRACER_RENDERING_RENDERER_CUH
