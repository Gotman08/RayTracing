/**
 * Render Configuration
 */

#ifndef RAYTRACER_RENDERING_RENDER_CONFIG_CUH
#define RAYTRACER_RENDERING_RENDER_CONFIG_CUH

#include "raytracer/core/vec3.cuh"
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

} // namespace rt

#endif // RAYTRACER_RENDERING_RENDER_CONFIG_CUH
