#ifndef RAYTRACER_RENDERING_RENDERER_CUH
#define RAYTRACER_RENDERING_RENDERER_CUH

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/materials/emissive.cuh"
#include "raytracer/materials/isotropic.cuh"
#include "raytracer/environment/sky.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

namespace rt {

struct RenderConfig {
    int width;
    int height;
    int samples_per_pixel;
    int max_depth;
    ToneMapMode tone_map;
    float exposure;
    bool use_sky;
    Sky sky;
    Color background;

    RenderConfig()
        : width(800), height(600), samples_per_pixel(100), max_depth(50),
          tone_map(ToneMapMode::ACES), exposure(0.0f), use_sky(true),
          background(0, 0, 0) {
        sky = Sky(Color(1, 1, 1), Color(0.5f, 0.7f, 1.0f));
    }
};

// Material scatter dispatch
__device__ inline bool scatter(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    switch (mat.type) {
        case MaterialType::LAMBERTIAN:
            return scatter_lambertian(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::METAL:
            return scatter_metal(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::DIELECTRIC:
            return scatter_dielectric(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::EMISSIVE:
            return scatter_emissive(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::ISOTROPIC:
            return scatter_isotropic(mat, r_in, rec, attenuation, scattered, rand_state);
        default:
            return false;
    }
}

// Iterative path tracing (avoids CUDA stack overflow)
__device__ inline Color ray_color(
    const Ray& initial_ray,
    const HittableList& world,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    Color accumulated(0, 0, 0);
    Color attenuation(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (world.hit(current_ray, Interval(0.001f, INFINITY_F), rec)) {
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
                // Ray absorbed
                break;
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

// Main render kernel
__global__ void render_kernel(
    Color* frame_buffer,
    Camera camera,
    HittableList world,
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
        pixel_color += ray_color(r, world, config.max_depth, config, local_rand_state);
    }

    // Average samples
    pixel_color = pixel_color / static_cast<float>(config.samples_per_pixel);

    // Apply exposure
    pixel_color = apply_exposure(pixel_color, config.exposure);

    // Tone mapping
    pixel_color = apply_tone_mapping(pixel_color, config.tone_map);

    // Gamma correction
    pixel_color = gamma_correct(pixel_color);

    // Clamp
    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

// Initialize random states
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
