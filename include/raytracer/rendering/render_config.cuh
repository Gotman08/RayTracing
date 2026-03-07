#ifndef RAYTRACER_RENDERING_RENDER_CONFIG_CUH
#define RAYTRACER_RENDERING_RENDER_CONFIG_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/environment/sky.cuh"

namespace rt {

struct RenderConfig {
    int width;
    int height;
    int samples_per_pixel;
    int max_depth;
    bool use_sky;
    Sky sky;
    Color background;

    RenderConfig()
        : width(800), height(600), samples_per_pixel(100), max_depth(50),
          use_sky(true), background(0, 0, 0) {
        sky = Sky(Color(1, 1, 1), Color(0.5f, 0.7f, 1.0f));
    }
};

}

#endif
