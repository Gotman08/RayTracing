#ifndef RAYTRACER_RENDERING_INTEGRATOR_CUH
#define RAYTRACER_RENDERING_INTEGRATOR_CUH

#include "raytracer/rendering/renderer.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/lighting/importance_sampling.cuh"

namespace rt {

// Path tracing integrator with BVH acceleration
__device__ inline Color path_trace_bvh(
    const Ray& initial_ray,
    const BVH& bvh,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    Color accumulated(0, 0, 0);
    Color throughput(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (bvh.hit(current_ray, Interval(0.001f, INFINITY_F), rec)) {
            // Get emission
            Color emitted = rec.mat->emitted(rec.u, rec.v, rec.p);
            accumulated += throughput * emitted;

            // Scatter
            Ray scattered;
            Color attenuation;

            if (scatter(*rec.mat, current_ray, rec, attenuation, scattered, rand_state)) {
                throughput = throughput * attenuation;
                current_ray = scattered;

                // Russian Roulette
                if (depth > 3) {
                    float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                    if (curand_uniform(rand_state) > p) {
                        break;
                    }
                    throughput = throughput / p;
                }
            } else {
                break;
            }
        } else {
            // Background
            if (config.use_sky) {
                accumulated += throughput * config.sky.get_color(current_ray.direction());
            } else {
                accumulated += throughput * config.background;
            }
            break;
        }
    }

    return accumulated;
}

// Render kernel with BVH
__global__ void render_kernel_bvh(
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
        pixel_color += path_trace_bvh(r, bvh, config.max_depth, config, local_rand_state);
    }

    pixel_color = pixel_color / static_cast<float>(config.samples_per_pixel);
    pixel_color = apply_exposure(pixel_color, config.exposure);
    pixel_color = apply_tone_mapping(pixel_color, config.tone_map);
    pixel_color = gamma_correct(pixel_color);

    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

// Progressive rendering (accumulate samples over multiple passes)
__global__ void render_kernel_progressive(
    Color* accumulator,
    Color* frame_buffer,
    Camera camera,
    HittableList world,
    RenderConfig config,
    curandState* rand_states,
    int current_sample
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= config.width || j >= config.height) return;

    int pixel_index = j * config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    // Trace one sample
    Ray r = camera.get_ray(i, j, local_rand_state);
    Color sample_color = ray_color(r, world, config.max_depth, config, local_rand_state);

    // Accumulate
    accumulator[pixel_index] += sample_color;

    // Compute current average
    Color pixel_color = accumulator[pixel_index] / static_cast<float>(current_sample + 1);

    // Post-process
    pixel_color = apply_exposure(pixel_color, config.exposure);
    pixel_color = apply_tone_mapping(pixel_color, config.tone_map);
    pixel_color = gamma_correct(pixel_color);

    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

} // namespace rt

#endif // RAYTRACER_RENDERING_INTEGRATOR_CUH
